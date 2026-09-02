// ---------------------------------------------------------------------------
// Extracted verbatim from hetero-serve: heteroserve/model/paged_attn_v3.py  (constant: _CUDA_SRC)
// by tools/extract_kernels.py -- do not hand-edit; edit the .cu and re-sync.
//
// Original build path: torch.utils.cpp_extension.load_inline(
//     name='heteroserve_paged_attn_v3', functions=['paged_attention_v3', 'choose_num_splits'],
//     extra_cuda_cflags=['-O3', '--use_fast_math'])
// Note: no -arch/-gencode was passed upstream; load_inline inferred the target
// from the live device or TORCH_CUDA_ARCH_LIST. The SM121 port makes it explicit.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu
#define NEG_BIG (-3.0e38f)

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}

// ---------------------------------------------------------------------------
// pass 1: each warp owns (sequence, head, split) and streams only its slice
// ---------------------------------------------------------------------------
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_split_kernel(
    float* __restrict__ partial_out,          // [B, H, S, D]
    float* __restrict__ partial_m,            // [B, H, S]
    float* __restrict__ partial_l,            // [B, H, S]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, H, D]
    const scalar_t* __restrict__ v_cache,
    const float* __restrict__ q,              // [B, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_blocks,
    const int num_splits,
    const float scale) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;

  const int lane = threadIdx.x;
  const int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= num_seqs * num_heads * num_splits) return;

  const int split = flat % num_splits;
  const int hb = flat / num_splits;
  const int h = hb % num_heads;
  const int b = hb / num_heads;

  // GQA: query head h reads the KV head it shares with its group.
  const int kv_h = h / (num_heads / num_kv_heads);

  const int ctx_len = context_lens[b];
  const int chunk = (ctx_len + num_splits - 1) / num_splits;
  const int start = split * chunk;
  const int end = min(start + chunk, ctx_len);

  float m = NEG_BIG;
  float l = 0.0f;
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  // An empty slice (short sequence, many splits) falls through with m = -big
  // and l = 0, which the merge weights to exactly zero. No special case needed.
  if (start < end) {
    const float* q_ptr = q + (size_t)(b * num_heads + h) * HEAD_DIM;
    float q_reg[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; ++i) q_reg[i] = q_ptr[lane * VPT + i];

    const int* btab = block_tables + (size_t)b * max_blocks;

    for (int j = start; j < end; ++j) {
      const int blk = btab[j / block_size];
      const int off = j % block_size;
      const size_t base = (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * HEAD_DIM;

      const scalar_t* k_ptr = k_cache + base + lane * VPT;
      float partial = 0.0f;
      #pragma unroll
      for (int i = 0; i < VPT; ++i) partial += q_reg[i] * static_cast<float>(k_ptr[i]);

      float s = warp_reduce_sum(partial);
      s = __shfl_sync(FULL_MASK, s, 0) * scale;

      const float m_new = fmaxf(m, s);
      const float correction = __expf(m - m_new);
      const float p = __expf(s - m_new);
      l = l * correction + p;

      const scalar_t* v_ptr = v_cache + base + lane * VPT;
      #pragma unroll
      for (int i = 0; i < VPT; ++i)
        acc[i] = acc[i] * correction + p * static_cast<float>(v_ptr[i]);

      m = m_new;
    }
  }

  const size_t pidx = (size_t)(b * num_heads + h) * num_splits + split;
  if (lane == 0) {
    partial_m[pidx] = m;
    partial_l[pidx] = l;
  }
  // Deliberately un-normalised: dividing by l here would lose the information
  // the merge needs to reweight this slice against the others.
  float* po = partial_out + pidx * HEAD_DIM + lane * VPT;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) po[i] = acc[i];
}

// ---------------------------------------------------------------------------
// pass 2: combine the per-split softmax states. Cheap: S values per (b, h).
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void merge_splits_kernel(
    float* __restrict__ out,                  // [B, H, D]
    const float* __restrict__ partial_out,    // [B, H, S, D]
    const float* __restrict__ partial_m,      // [B, H, S]
    const float* __restrict__ partial_l,      // [B, H, S]
    const int num_seqs,
    const int num_heads,
    const int num_splits) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;

  const int lane = threadIdx.x;
  const int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= num_seqs * num_heads) return;

  const size_t base = (size_t)flat * num_splits;

  float m_g = NEG_BIG;
  for (int s = 0; s < num_splits; ++s) m_g = fmaxf(m_g, partial_m[base + s]);

  float l_g = 0.0f;
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  for (int s = 0; s < num_splits; ++s) {
    const float w = __expf(partial_m[base + s] - m_g);
    l_g += partial_l[base + s] * w;
    const float* po = partial_out + (base + s) * HEAD_DIM + lane * VPT;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) acc[i] += po[i] * w;
  }

  const float inv = 1.0f / l_g;
  float* o = out + (size_t)flat * HEAD_DIM + lane * VPT;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) o[i] = acc[i] * inv;
}

// ---------------------------------------------------------------------------

int64_t choose_num_splits(int64_t num_seqs, int64_t num_heads, int64_t max_context) {
  // The first version of this targeted occupancy -- enough warps to give each SM
  // a few -- and it was wrong. Measured on a T4 at batch 16 / context 512 it
  // chose 2 splits (23.5% of peak) when 32 gave 48.8%, and at batch 32 /
  // context 2048 it chose 1, i.e. no split at all.
  //
  // Occupancy is not the binding constraint: the online softmax is *sequential*,
  // so a warp streaming 512 tokens carries a 512-long dependent chain of
  // exp/rescale. Splitting shortens that chain. So target a chain length
  // (~32 tokens per split) and only fall back to a device-fill argument when
  // the context is too short for that to produce enough work.
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t have = std::max<int64_t>(num_seqs * num_heads, 1);

  const int64_t by_chain = std::max<int64_t>(max_context / 32, 1);
  const int64_t to_fill_device =
      std::max<int64_t>(((int64_t)sms * 8 + have - 1) / have, 1);

  int64_t splits = std::max(by_chain, to_fill_device);
  return std::max<int64_t>(1, std::min<int64_t>(splits, 64));
}

torch::Tensor paged_attention_v3(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale,
    int64_t num_splits) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  q = q.contiguous();
  k_cache = k_cache.contiguous();
  v_cache = v_cache.contiguous();
  block_tables = block_tables.to(torch::kInt32).contiguous();
  context_lens = context_lens.to(torch::kInt32).contiguous();

  const int B = q.size(0);
  const int H = q.size(1);
  const int D = q.size(2);
  const int BS = k_cache.size(1);
  const int HKV = k_cache.size(2);
  const int MB = block_tables.size(1);

  TORCH_CHECK(H % HKV == 0, "n_head must be divisible by n_kv_head");
  TORCH_CHECK(D % 32 == 0, "head_dim must be a multiple of the warp size");

  if (num_splits <= 0) num_splits = choose_num_splits(B, H, (int64_t)MB * BS);

  auto fopts = q.options().dtype(torch::kFloat32);
  auto out = torch::empty({B, H, D}, fopts);
  auto partial_out = torch::empty({B, H, (int)num_splits, D}, fopts);
  auto partial_m = torch::empty({B, H, (int)num_splits}, fopts);
  auto partial_l = torch::empty({B, H, (int)num_splits}, fopts);

  const int warps_per_block = 4;
  const dim3 threads(WARP_SIZE, warps_per_block);

  const int split_warps = B * H * (int)num_splits;
  const dim3 split_grid((split_warps + warps_per_block - 1) / warps_per_block);
  const int merge_warps = B * H;
  const dim3 merge_grid((merge_warps + warps_per_block - 1) / warps_per_block);

  #define LAUNCH(DIM)                                                           \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                        \
        k_cache.scalar_type(), "paged_attention_v3", ([&] {                     \
          paged_attention_split_kernel<scalar_t, DIM>                           \
              <<<split_grid, threads>>>(                                        \
                  partial_out.data_ptr<float>(), partial_m.data_ptr<float>(),   \
                  partial_l.data_ptr<float>(), k_cache.data_ptr<scalar_t>(),    \
                  v_cache.data_ptr<scalar_t>(), q.data_ptr<float>(),            \
                  block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),   \
                  B, H, HKV, BS, MB, (int)num_splits, (float)scale);                 \
        }));                                                                    \
    merge_splits_kernel<DIM><<<merge_grid, threads>>>(                          \
        out.data_ptr<float>(), partial_out.data_ptr<float>(),                   \
        partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),               \
        B, H, (int)num_splits);

  switch (D) {
    case 32:  LAUNCH(32);  break;
    case 64:  LAUNCH(64);  break;
    case 128: LAUNCH(128); break;
    case 256: LAUNCH(256); break;
    default: TORCH_CHECK(false, "unsupported head_dim ", D);
  }
  #undef LAUNCH

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}
