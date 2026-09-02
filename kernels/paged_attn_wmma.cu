// ---------------------------------------------------------------------------
// Extracted verbatim from hetero-serve: heteroserve/model/paged_attn_wmma.py  (constant: _CUDA_SRC)
// by tools/extract_kernels.py -- do not hand-edit; edit the .cu and re-sync.
//
// Original build path: torch.utils.cpp_extension.load_inline(
//     name='heteroserve_prefill_wmma', functions=['paged_attention_prefill_wmma'],
//     extra_cuda_cflags=['-O3', '--use_fast_math'])
// Note: no -arch/-gencode was passed upstream; load_inline inferred the target
// from the live device or TORCH_CUDA_ARCH_LIST. The SM121 port makes it explicit.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#define HS_HAS_WMMA 1
#include <mma.h>
using namespace nvcuda;
#else
#define HS_HAS_WMMA 0
#endif

#define WARP_SIZE 32
#define TILE 16                 // WMMA M/N/K, and exactly one KV page
#define WARPS_PER_BLOCK 2
#define NEG_BIG (-3.0e38f)

// One warp per (sequence, 16-query tile, head).
//
// Shared memory per warp, head_dim 64:  Q 2K + K 2K + V 2K + S 1K + P 0.5K
// + O staging 1K = 8.5 KB. Two warps per block keeps a 128-dim head inside
// sm_75's 48 KB without needing the opt-in carveout.
template <int HEAD_DIM>
__global__ void prefill_wmma_kernel(
    float* __restrict__ out,                  // [B, S, H, D]
    const __half* __restrict__ k_cache,       // [NB, 16, HKV, D]
    const __half* __restrict__ v_cache,
    const float* __restrict__ q,              // [B, S, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_seqs,
    const int num_queries,
    const int num_heads,
    const int num_kv_heads,
    const int max_blocks,
    const float scale) {
#if HS_HAS_WMMA
  constexpr int NCHUNK = HEAD_DIM / TILE;     // head_dim slices of 16

  extern __shared__ char smem_raw[];
  const int warp = threadIdx.y;
  const int lane = threadIdx.x;

  // per-warp slab
  const size_t per_warp =
      sizeof(__half) * (3 * TILE * HEAD_DIM + TILE * TILE)   // Q, K, V, P
      + sizeof(float) * (TILE * TILE + TILE * TILE + 3 * TILE);  // S, O, m/l/corr
  char* base = smem_raw + warp * per_warp;

  __half* Qs = reinterpret_cast<__half*>(base);
  __half* Ks = Qs + TILE * HEAD_DIM;
  __half* Vs = Ks + TILE * HEAD_DIM;
  __half* Ps = Vs + TILE * HEAD_DIM;
  float*  Ss = reinterpret_cast<float*>(Ps + TILE * TILE);
  float*  Os = Ss + TILE * TILE;
  float*  ms = Os + TILE * TILE;
  float*  ls = ms + TILE;
  float*  cs = ls + TILE;

  const int q_tiles = (num_queries + TILE - 1) / TILE;
  const int flat = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (flat >= num_seqs * q_tiles * num_heads) return;

  const int h  = flat % num_heads;
  const int qt = (flat / num_heads) % q_tiles;
  const int b  = flat / (num_heads * q_tiles);

  const int kv_h = h / (num_heads / num_kv_heads);
  const int ctx_len = context_lens[b];
  const int q0 = qt * TILE;
  // absolute position of query row (q0 + r): a chunk is the trailing tokens
  const int q_abs0 = ctx_len - num_queries + q0;

  // ---- load the Q tile (fp32 -> half), zeroing rows past the chunk ----
  for (int i = lane; i < TILE * HEAD_DIM; i += WARP_SIZE) {
    const int r = i / HEAD_DIM, d = i % HEAD_DIM;
    const int qr = q0 + r;
    Qs[i] = (qr < num_queries)
        ? __float2half(q[((size_t)(b * num_queries + qr) * num_heads + h) * HEAD_DIM + d])
        : __float2half(0.0f);
  }
  if (lane < TILE) { ms[lane] = NEG_BIG; ls[lane] = 0.0f; }
  __syncwarp();

  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> o_frag[NCHUNK];
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) wmma::fill_fragment(o_frag[c], 0.0f);

  // Only tiles up to the last row's causal bound can contribute.
  const int last_q_abs = q_abs0 + TILE - 1;
  const int key_limit = min(ctx_len, last_q_abs + 1);
  const int n_tiles = (key_limit + TILE - 1) / TILE;

  const int* btab = block_tables + (size_t)b * max_blocks;

  for (int kt = 0; kt < n_tiles; ++kt) {
    const int blk = btab[kt];
    // One key tile is exactly one page -- the whole point of block_size == 16.
    const size_t page = (size_t)blk * TILE * num_kv_heads * HEAD_DIM
                        + (size_t)kv_h * HEAD_DIM;
    for (int i = lane; i < TILE * HEAD_DIM; i += WARP_SIZE) {
      const int t = i / HEAD_DIM, d = i % HEAD_DIM;
      const size_t off = page + (size_t)t * num_kv_heads * HEAD_DIM + d;
      Ks[i] = k_cache[off];
      Vs[i] = v_cache[off];
    }
    __syncwarp();

    // ---- S = Q * K^T  (col_major on K gives the transpose for free) ----
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> s_frag;
    wmma::fill_fragment(s_frag, 0.0f);
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::col_major> bfrag;
      wmma::load_matrix_sync(a, Qs + c * TILE, HEAD_DIM);
      wmma::load_matrix_sync(bfrag, Ks + c * TILE, HEAD_DIM);
      wmma::mma_sync(s_frag, a, bfrag, s_frag);
    }
    wmma::store_matrix_sync(Ss, s_frag, TILE, wmma::mem_row_major);
    __syncwarp();

    // ---- causal mask + online softmax, one row per lane ----
    if (lane < TILE) {
      const int r = lane;
      const int q_abs = q_abs0 + r;
      const bool row_live = (q0 + r) < num_queries && q_abs >= 0;

      float rmax = NEG_BIG;
      #pragma unroll
      for (int j = 0; j < TILE; ++j) {
        const int kpos = kt * TILE + j;
        const bool keep = row_live && kpos <= q_abs && kpos < ctx_len;
        const float v = keep ? Ss[r * TILE + j] * scale : NEG_BIG;
        Ss[r * TILE + j] = v;
        rmax = fmaxf(rmax, v);
      }
      const float m_new = fmaxf(ms[r], rmax);
      const float corr = __expf(ms[r] - m_new);
      float rsum = 0.0f;
      #pragma unroll
      for (int j = 0; j < TILE; ++j) {
        const float p = (Ss[r * TILE + j] <= NEG_BIG) ? 0.0f
                        : __expf(Ss[r * TILE + j] - m_new);
        Ps[r * TILE + j] = __float2half(p);
        rsum += p;
      }
      ls[r] = ls[r] * corr + rsum;
      ms[r] = m_new;
      cs[r] = corr;
    }
    __syncwarp();

    // ---- O = O*corr + P*V ----
    // The accumulator's register mapping is opaque, so the per-row rescale has
    // to round-trip through shared memory. This is the part CUTLASS avoids.
    #pragma unroll
    for (int c = 0; c < NCHUNK; ++c) {
      wmma::store_matrix_sync(Os, o_frag[c], TILE, wmma::mem_row_major);
      __syncwarp();
      if (lane < TILE) {
        const float k = cs[lane];
        #pragma unroll
        for (int j = 0; j < TILE; ++j) Os[lane * TILE + j] *= k;
      }
      __syncwarp();
      wmma::load_matrix_sync(o_frag[c], Os, TILE, wmma::mem_row_major);

      wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> pa;
      wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::row_major> vb;
      wmma::load_matrix_sync(pa, Ps, TILE);
      wmma::load_matrix_sync(vb, Vs + c * TILE, HEAD_DIM);
      wmma::mma_sync(o_frag[c], pa, vb, o_frag[c]);
      __syncwarp();
    }
  }

  // ---- normalise and write out ----
  #pragma unroll
  for (int c = 0; c < NCHUNK; ++c) {
    wmma::store_matrix_sync(Os, o_frag[c], TILE, wmma::mem_row_major);
    __syncwarp();
    if (lane < TILE) {
      const int r = lane;
      const int qr = q0 + r;
      if (qr < num_queries) {
        const float inv = (ls[r] > 0.0f) ? (1.0f / ls[r]) : 0.0f;
        float* o = out + ((size_t)(b * num_queries + qr) * num_heads + h) * HEAD_DIM
                   + c * TILE;
        #pragma unroll
        for (int j = 0; j < TILE; ++j) o[j] = Os[r * TILE + j] * inv;
      }
    }
    __syncwarp();
  }
#endif  // HS_HAS_WMMA
}

torch::Tensor paged_attention_prefill_wmma(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(q.dim() == 4, "q must be [B, S, H, D]");
  TORCH_CHECK(k_cache.scalar_type() == torch::kHalf,
              "the WMMA path needs an fp16 KV cache");
  TORCH_CHECK(k_cache.size(1) == 16,
              "the WMMA path needs block_size == 16 (one page per fragment)");

  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  block_tables = block_tables.to(torch::kInt32).contiguous();
  context_lens = context_lens.to(torch::kInt32).contiguous();

  const int B = q.size(0);
  const int S = q.size(1);
  const int H = q.size(2);
  const int D = q.size(3);
  const int HKV = k_cache.size(2);
  const int MB = block_tables.size(1);

  TORCH_CHECK(H % HKV == 0, "n_head must be divisible by n_kv_head");
  TORCH_CHECK(D % 16 == 0, "head_dim must be a multiple of 16");

  auto out = torch::empty({B, S, H, D}, q.options().dtype(torch::kFloat32));

  const int q_tiles = (S + 15) / 16;
  const int total = B * q_tiles * H;
  const dim3 threads(32, WARPS_PER_BLOCK);
  const dim3 grid((total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  #define LAUNCH(DIM)                                                          \
    {                                                                          \
      const size_t per_warp =                                                  \
          sizeof(__half) * (3 * 16 * (DIM) + 16 * 16)                          \
          + sizeof(float) * (16 * 16 + 16 * 16 + 3 * 16);                      \
      const size_t shmem = per_warp * WARPS_PER_BLOCK;                         \
      prefill_wmma_kernel<DIM><<<grid, threads, shmem>>>(                      \
          out.data_ptr<float>(),                                               \
          reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),       \
          reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),       \
          q.data_ptr<float>(), block_tables.data_ptr<int>(),                   \
          context_lens.data_ptr<int>(), B, S, H, HKV, MB, (float)scale);       \
    }

  switch (D) {
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 128: LAUNCH(128); break;
    default: TORCH_CHECK(false, "unsupported head_dim for the WMMA path: ", D);
  }
  #undef LAUNCH

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}
