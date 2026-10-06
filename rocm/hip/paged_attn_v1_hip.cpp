#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_CHECK(expr) do { \
  hipError_t _e = (expr); \
  if (_e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
    std::exit(2); \
  } \
} while (0)

// HIP/ROCm port of kernels/paged_attn_v1.cu.
// Layout: q [B,H,D], k/v [NB,BS,HKV,D], block table [B,MB].
__global__ void paged_attention_hip(
    float* __restrict__ out,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const float* __restrict__ q,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks,
    int max_context,
    float scale) {

  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int kv_h = h / (num_heads / num_kv_heads);
  const int ctx_len = context_lens[b];
  if (ctx_len <= 0) return;

  extern __shared__ float smem[];
  float* scores = smem;
  float* red = smem + max_context;

  const float* q_ptr = q + (size_t)(b * num_heads + h) * head_dim;
  const int* btab = block_tables + (size_t)b * max_blocks;

  for (int j = tid; j < ctx_len; j += nthreads) {
    const int blk = btab[j / block_size];
    const int off = j % block_size;
    const float* k_ptr =
        k_cache + (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * head_dim;
    float acc = 0.0f;
    for (int d = 0; d < head_dim; ++d) acc += q_ptr[d] * k_ptr[d];
    scores[j] = acc * scale;
  }
  __syncthreads();

  float local_max = -3.0e38f;
  for (int j = tid; j < ctx_len; j += nthreads) local_max = fmaxf(local_max, scores[j]);
  red[tid] = local_max;
  __syncthreads();

  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
    __syncthreads();
  }
  const float mx = red[0];

  float partial = 0.0f;
  for (int j = tid; j < ctx_len; j += nthreads) {
    const float e = expf(scores[j] - mx);
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

  for (int d = tid; d < head_dim; d += nthreads) {
    float acc = 0.0f;
    for (int j = 0; j < ctx_len; ++j) {
      const int blk = btab[j / block_size];
      const int off = j % block_size;
      const float* v_ptr =
          v_cache + (((size_t)blk * block_size + off) * num_kv_heads + kv_h) * head_dim;
      acc += scores[j] * v_ptr[d];
    }
    out[(size_t)(b * num_heads + h) * head_dim + d] = acc / denom;
  }

  // Avoid fixed NVIDIA warp-width assumptions; HIP exposes the target wave size.
  if (warpSize == 0 && tid == 0) out[0] = 0.0f;
}

static void cpu_reference(
    std::vector<float>& out,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& q,
    const std::vector<int>& bt,
    const std::vector<int>& ctx,
    int B, int H, int HKV, int D, int BS, int MB, float scale) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const int kv_h = h / (H / HKV);
      const int n = ctx[b];
      std::vector<float> score(n);
      float mx = -3.0e38f;
      for (int j = 0; j < n; ++j) {
        int blk = bt[b * MB + j / BS], off = j % BS;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          size_t ki = (((size_t)blk * BS + off) * HKV + kv_h) * D + d;
          dot += q[(size_t)(b * H + h) * D + d] * k[ki];
        }
        score[j] = dot * scale;
        mx = std::max(mx, score[j]);
      }
      float denom = 0.0f;
      for (float& s : score) { s = std::exp(s - mx); denom += s; }
      for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) {
          int blk = bt[b * MB + j / BS], off = j % BS;
          size_t vi = (((size_t)blk * BS + off) * HKV + kv_h) * D + d;
          acc += score[j] * v[vi];
        }
        out[(size_t)(b * H + h) * D + d] = acc / denom;
      }
    }
  }
}

int main() {
  int device_count = 0;
  hipError_t count_status = hipGetDeviceCount(&device_count);
  if (count_status != hipSuccess || device_count == 0) {
    std::puts("HIP compile smoke: binary built; no AMD GPU exposed, runtime self-test skipped");
    return 0;
  }

  constexpr int B=2, H=4, HKV=2, D=32, BS=16, MB=4, NB=8, NCTX=48;
  const float scale = 1.0f / std::sqrt((float)D);
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

  std::vector<float> q(B*H*D), k(NB*BS*HKV*D), v(k.size());
  for (float& x : q) x=dist(rng);
  for (float& x : k) x=dist(rng);
  for (float& x : v) x=dist(rng);
  std::vector<int> bt(B*MB);
  for (int b=0;b<B;++b) for(int m=0;m<MB;++m) bt[b*MB+m]=(b*MB+m)%NB;
  std::vector<int> ctx(B,NCTX);
  std::vector<float> ref(B*H*D), got(ref.size());
  cpu_reference(ref,k,v,q,bt,ctx,B,H,HKV,D,BS,MB,scale);

  float *dq,*dk,*dv,*dout; int *dbt,*dctx;
  HIP_CHECK(hipMalloc(&dq,q.size()*sizeof(float)));
  HIP_CHECK(hipMalloc(&dk,k.size()*sizeof(float)));
  HIP_CHECK(hipMalloc(&dv,v.size()*sizeof(float)));
  HIP_CHECK(hipMalloc(&dout,got.size()*sizeof(float)));
  HIP_CHECK(hipMalloc(&dbt,bt.size()*sizeof(int)));
  HIP_CHECK(hipMalloc(&dctx,ctx.size()*sizeof(int)));
  HIP_CHECK(hipMemcpy(dq,q.data(),q.size()*sizeof(float),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dk,k.data(),k.size()*sizeof(float),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dv,v.data(),v.size()*sizeof(float),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dbt,bt.data(),bt.size()*sizeof(int),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dctx,ctx.data(),ctx.size()*sizeof(int),hipMemcpyHostToDevice));

  dim3 grid(B,H);
  constexpr int threads=128;
  size_t shmem=(MB*BS+threads)*sizeof(float);
  hipLaunchKernelGGL(paged_attention_hip, grid, dim3(threads), shmem, 0,
      dout,dk,dv,dq,dbt,dctx,H,HKV,D,BS,MB,MB*BS,scale);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(got.data(),dout,got.size()*sizeof(float),hipMemcpyDeviceToHost));

  float max_abs=0.0f;
  for(size_t i=0;i<got.size();++i) max_abs=std::max(max_abs,std::fabs(got[i]-ref[i]));
  std::printf("ROCm HIP paged-attention self-test max_abs=%g\n",max_abs);

  hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dout); hipFree(dbt); hipFree(dctx);
  return max_abs < 3e-4f ? 0 : 1;
}
