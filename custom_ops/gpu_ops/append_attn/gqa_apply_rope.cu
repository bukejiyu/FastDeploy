// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "encoder_write_cache_with_rope_impl.cuh"
#include "helper.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "qwen3_rope.h"
#include "remote_cache_kv_ipc.h"
#include "gqa_rope_write_cache_impl.cuh"


std::vector<paddle::Tensor> GQAApplyRopeKernel(
  const paddle::Tensor &qkv,
  const paddle::Tensor &cu_seqlens_q,
  const paddle::Tensor &cu_seqlens_k,
  const paddle::Tensor &rotary_embs,
  const paddle::Tensor &seq_lens_encoder,
  const paddle::Tensor &seq_lens_decoder,
  const paddle::Tensor &batch_id_per_token,
  const paddle::optional<paddle::Tensor> &q_norm_weight,
  const paddle::optional<paddle::Tensor> &k_norm_weight,
  const int kv_token_num,
  const int attn_heads,
  const int kv_num_heads,
  const int head_dim,
  const int max_seq_len,
  const float rms_norm_eps,
  const bool use_neox_rotary_style,
  const bool rope_3d){
  typedef PDTraits<paddle::DataType::BFLOAT16> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
  const auto &qkv_dims = qkv.dims();
  const int token_num = qkv_dims[0];
  
  int rotary_dim = head_dim;
  
  if (!rope_3d) {
    PADDLE_ENFORCE_EQ(rotary_embs.dims().size(), 5);
    PADDLE_ENFORCE_EQ(rotary_embs.dims()[0], 2);
    PADDLE_ENFORCE_EQ(rotary_embs.dims()[1], 1);
    PADDLE_ENFORCE_EQ(rotary_embs.dims()[2], max_seq_len);
    PADDLE_ENFORCE_EQ(rotary_embs.dims()[3], 1);
    if (use_neox_rotary_style) {
      // Note(ZKK) Qwen3 like model
      // the [0,head_dim/2), [head_dim/2,head_dim) data are totally same!
      if (rotary_embs.dims()[4] == head_dim) {
        rotary_dim = head_dim;
      } else {
        // for glm partial rotary style
        PADDLE_ENFORCE_EQ(rotary_embs.dims()[4], head_dim / 4);
        rotary_dim = head_dim / 2;
      }
    } else {
      PADDLE_ENFORCE_EQ(rotary_embs.dims()[4], head_dim / 2);
    }
  }
  
  auto stream = qkv.stream();
  paddle::Tensor qkv_out = GetEmptyTensor(qkv.dims(), qkv.dtype(), qkv.place());
  paddle::Tensor q = GetEmptyTensor(
      {token_num, attn_heads, head_dim}, qkv.dtype(), qkv.place());
  paddle::Tensor k = GetEmptyTensor(
      {kv_token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());
  paddle::Tensor v = GetEmptyTensor(
      {kv_token_num, kv_num_heads, head_dim}, qkv.dtype(), qkv.place());

  if (use_neox_rotary_style) {
    if (rotary_dim == head_dim) {
      gqa_rotary_qk_split_variable_qwen3<data_t>(
          qkv_out.data<data_t>(),
          q.data<data_t>(),
          k.data<data_t>(),
          v.data<data_t>(),
          qkv.data<data_t>(),
          rotary_embs.data<float>(),
          batch_id_per_token.data<int>(),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          cu_seqlens_q.data<int>(),
          cu_seqlens_k.data<int>(),
          token_num,
          attn_heads,
          kv_num_heads,
          rope_3d ? rotary_embs.dims()[3] : rotary_embs.dims()[2],
          head_dim,
          rope_3d,
          stream);
    } else {
      gqa_neox_partial_rotary_qk_split_variable<data_t>(
          qkv_out.data<data_t>(),
          q.data<data_t>(),
          k.data<data_t>(),
          v.data<data_t>(),
          qkv.data<data_t>(),
          rotary_embs.data<float>(),
          batch_id_per_token.data<int>(),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          cu_seqlens_q.data<int>(),
          cu_seqlens_k.data<int>(),
          token_num,
          attn_heads,
          kv_num_heads,
          max_seq_len,
          head_dim,
          rotary_dim,
          stream);
    }
  } else {
    gqa_rotary_qk_split_variable<data_t>(
        qkv_out.data<data_t>(),
        q.data<data_t>(),
        k.data<data_t>(),
        v.data<data_t>(),
        qkv.data<data_t>(),
        rotary_embs.data<float>(),
        q_norm_weight ? q_norm_weight.get().data<float>() : nullptr,
        k_norm_weight ? k_norm_weight.get().data<float>() : nullptr,
        batch_id_per_token.data<int>(),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        cu_seqlens_q.data<int>(),
        cu_seqlens_k.data<int>(),
        token_num,
        attn_heads,
        kv_num_heads,
        max_seq_len,
        rope_3d ? rotary_embs.dims()[3] : rotary_embs.dims()[2],
        head_dim,
        rope_3d,
        rms_norm_eps,
        stream);
  }
  return {q, k, v, qkv_out};
}


PD_BUILD_STATIC_OP(gqa_apply_rope)
    .Inputs({"qkv",
             "cu_seqlens_q",
             "cu_seqlens_k",
             "rotary_embs",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "batch_id_per_token",
             paddle::Optional("q_norm_weight"),
             paddle::Optional("k_norm_weight")})
    .Outputs({"q", "k", "v", "qkv_out"})
    .Attrs({"kv_token_num: int",
            "attn_heads: int",
            "kv_num_heads: int",
            "head_dim: int",
            "max_seq_len: int",
            "rms_norm_eps: float",
            "use_neox_rotary_style: bool",
            "rope_3d: bool"})
    .SetKernelFn(PD_KERNEL(GQAApplyRopeKernel));