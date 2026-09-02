// ---------------------------------------------------------------------------
// Extracted verbatim from hetero-serve: heteroserve/model/paged_attn_v3.py  (constant: _CPP_DECL)
// by tools/extract_kernels.py -- do not hand-edit; edit the .cu and re-sync.
//
// Original build path: torch.utils.cpp_extension.load_inline(
//     name='heteroserve_paged_attn_v3', functions=['paged_attention_v3', 'choose_num_splits'],
//     extra_cuda_cflags=['-O3', '--use_fast_math'])
// Note: no -arch/-gencode was passed upstream; load_inline inferred the target
// from the live device or TORCH_CUDA_ARCH_LIST. The SM121 port makes it explicit.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
torch::Tensor paged_attention_v3(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_tables, torch::Tensor context_lens, double scale,
    int64_t num_splits);
int64_t choose_num_splits(int64_t num_seqs, int64_t num_heads, int64_t max_context);
