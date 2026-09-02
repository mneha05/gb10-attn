// ---------------------------------------------------------------------------
// Extracted verbatim from hetero-serve: heteroserve/model/paged_attn_v2.py  (constant: _CUDA_SRC)
// by tools/extract_kernels.py -- do not hand-edit; edit the .cu and re-sync.
//
// Original build path: torch.utils.cpp_extension.load_inline(
//     name='heteroserve_paged_attn_v2', functions=['paged_attention_v2'],
//     extra_cuda_cflags=['-O3', '--use_fast_math'])
// Note: no -arch/-gencode was passed upstream; load_inline inferred the target
// from the live device or TORCH_CUDA_ARCH_LIST. The SM121 port makes it explicit.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(FULL_MASK, v, off));
  return v;
}

// One warp per (sequence, head). Each warp streams the whole context once,
// maintaining an online softmax so no score vector is ever materialised.
//
// Lane layout: head_dim is split across the warp, so lane i owns dims
// [i*VPT, (i+1)*VPT) where VPT = head_dim / 32. For head_dim 64 that is 2 dims
// per lane, read as one half2/float2.
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ out,                  // [B, H, D]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, H, D]
    const scalar_t* __restrict__ v_cache,     // [NB, BS, H, D]
    const float* __restrict__ q,              // [B, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_blocks,
    const float scale) {

  constexpr int VPT = HEAD_DIM / WARP_SIZE;   // values per thread

  const int warps_per_block = blockDim.y;
  const int warp_id = threadIdx.y;
  const int lane = threadIdx.x;

  // Flat warp id over (sequence, head). A 1-D grid only: pairing this with a
  // second grid dimension for batch would run every warp once per sequence.
  const int flat = blockIdx.x * warps_per_block + warp_id;
  if (flat >= num_seqs * num_heads) return;
  const int b = flat / num_heads;
  const int h = flat % num_heads;

  // GQA: query head h reads the KV head it shares with its group.
  const int kv_h = h / (num_heads / num_kv_heads);

  const int ctx_len = context_lens[b];
  if (ctx_len <= 0) return;

  const int* btab = block_tables + (size_t)b * max_blocks;
  const float* q_ptr = q + (size_t)(b * num_heads + h) * HEAD_DIM;

  // This lane's slice of q, held in registers for the whole stream.
  float q_reg[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) q_reg[i] = q_ptr[lane * VPT + i];

  // Online softmax state, plus this lane's slice of the output accumulator.
  float m = -3.0e38f;      // running max
  float l = 0.0f;          // running sum of exp
  float acc[VPT];
  #pragma unroll
  for (int i = 0; i < VPT; ++i) acc[i] = 0.0f;

  for (int j = 0; j < ctx_len; ++j) {
    const int blk = btab[j / block_size];
    const int off = j % block_size;
    const size_t base = (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * HEAD_DIM;

    // ---- score: warp-cooperative dot product, one shuffle reduction ----
    const scalar_t* k_ptr = k_cache + base + lane * VPT;
    float partial = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; ++i) partial += q_reg[i] * static_cast<float>(k_ptr[i]);

    float s = warp_reduce_sum(partial);
    s = __shfl_sync(FULL_MASK, s, 0) * scale;   // broadcast lane 0's total

    // ---- online softmax rescale ----
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

  const float inv_l = 1.0f / l;
  float* o_ptr = out + (size_t)(b * num_heads + h) * HEAD_DIM + lane * VPT;
  #pragma unroll
  for (int i = 0; i < VPT; ++i) o_ptr[i] = acc[i] * inv_l;
}

torch::Tensor paged_attention_v2(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    double scale) {

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

  auto out = torch::empty({B, H, D}, q.options().dtype(torch::kFloat32));

  const int warps_per_block = 4;
  const dim3 threads(WARP_SIZE, warps_per_block);
  const int total_warps = B * H;
  const dim3 grid((total_warps + warps_per_block - 1) / warps_per_block);

  #define LAUNCH(DIM)                                                          \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                       \
        k_cache.scalar_type(), "paged_attention_v2", ([&] {                    \
          paged_attention_v2_kernel<scalar_t, DIM><<<grid, threads>>>(         \
              out.data_ptr<float>(), k_cache.data_ptr<scalar_t>(),             \
              v_cache.data_ptr<scalar_t>(), q.data_ptr<float>(),               \
              block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),      \
              B, H, HKV, BS, MB, (float)scale);                                        \
        }));

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

