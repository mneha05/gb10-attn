// ---------------------------------------------------------------------------
// Extracted verbatim from hetero-serve: heteroserve/model/paged_attn.py  (constant: _CUDA_SRC)
// by tools/extract_kernels.py -- do not hand-edit; edit the .cu and re-sync.
//
// Original build path: torch.utils.cpp_extension.load_inline(
//     name='heteroserve_paged_attn', functions=['paged_attention'],
//     extra_cuda_cflags=['-O3', '--use_fast_math'])
// Note: no -arch/-gencode was passed upstream; load_inline inferred the target
// from the live device or TORCH_CUDA_ARCH_LIST. The SM121 port makes it explicit.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// One CUDA block per (sequence, head). Threads cooperate over the context.
//
// The block table indirection is the whole point: position j of sequence b
// lives at block_tables[b][j / block_size], offset j % block_size. Nothing is
// ever copied into a contiguous buffer first.
template <typename scalar_t>
__global__ void paged_attention_kernel(
    float* __restrict__ out,                  // [B, H, D]
    const scalar_t* __restrict__ k_cache,     // [NB, BS, H, D]
    const scalar_t* __restrict__ v_cache,     // [NB, BS, H, D]
    const float* __restrict__ q,              // [B, H, D]
    const int* __restrict__ block_tables,     // [B, MB]
    const int* __restrict__ context_lens,     // [B]
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int block_size,
    const int max_blocks,
    const int max_context,
    const float scale) {

  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // GQA: query head h reads the KV head it shares with its group.
  const int kv_h = h / (num_heads / num_kv_heads);

  const int ctx_len = context_lens[b];
  if (ctx_len <= 0) return;

  // scores[0..max_context) then a reduction scratchpad of nthreads floats.
  extern __shared__ float smem[];
  float* scores = smem;
  float* red = smem + max_context;

  const float* q_ptr = q + (size_t)(b * num_heads + h) * head_dim;
  const int* btab = block_tables + (size_t)b * max_blocks;

  // ---- 1. q . k for every cached position, straight out of the pages ----
  for (int j = tid; j < ctx_len; j += nthreads) {
    const int blk = btab[j / block_size];
    const int off = j % block_size;
    const scalar_t* k_ptr =
        k_cache + (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * head_dim;
    float acc = 0.f;
    for (int d = 0; d < head_dim; ++d) {
      acc += q_ptr[d] * static_cast<float>(k_ptr[d]);
    }
    scores[j] = acc * scale;
  }
  __syncthreads();

  // ---- 2. softmax, in two block-wide reductions ----
  float local = -3.0e38f;   // effectively -inf for fp32 scores
  for (int j = tid; j < ctx_len; j += nthreads) local = fmaxf(local, scores[j]);
  red[tid] = local;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();

  float partial = 0.f;
  for (int j = tid; j < ctx_len; j += nthreads) {
    const float e = __expf(scores[j] - mx);
    scores[j] = e;
    partial += e;
  }
  red[tid] = partial;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  const float denom = red[0];
  __syncthreads();

  // ---- 3. weighted sum over V, one thread per output dim ----
  for (int d = tid; d < head_dim; d += nthreads) {
    float acc = 0.f;
    for (int j = 0; j < ctx_len; ++j) {
      const int blk = btab[j / block_size];
      const int off = j % block_size;
      const scalar_t* v_ptr =
          v_cache + (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * head_dim;
      acc += scores[j] * static_cast<float>(v_ptr[d]);
    }
    out[(size_t)(b * num_heads + h) * head_dim + d] = acc / denom;
  }
}

torch::Tensor paged_attention(
    torch::Tensor q,             // [B, H, D] float32, contiguous
    torch::Tensor k_cache,       // [NB, BS, H, D]
    torch::Tensor v_cache,       // [NB, BS, H, D]
    torch::Tensor block_tables,  // [B, MB] int32
    torch::Tensor context_lens,  // [B] int32
    double scale) {

  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(k_cache.is_cuda() && v_cache.is_cuda(), "kv cache must be on CUDA");
  TORCH_CHECK(q.dim() == 3, "q must be [B, H, D]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [NB, BS, H, D]");

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

  const int max_context = MB * BS;
  auto out = torch::empty({B, H, D}, q.options().dtype(torch::kFloat32));

  const int threads = 128;
  const dim3 grid(B, H);
  const size_t shmem = (size_t)(max_context + threads) * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      k_cache.scalar_type(), "paged_attention", ([&] {
        paged_attention_kernel<scalar_t><<<grid, threads, shmem>>>(
            out.data_ptr<float>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            q.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            H, HKV, D, BS, MB, max_context, (float)scale);
      }));

  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

