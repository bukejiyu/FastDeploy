// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "helper.h"
#include "utils.cuh"


template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthRotarySplitKernel(
    const T *qkv,
    const float *cos_emb,
    const float *sin_emb,
    const float *q_norm_weight,
    const float *k_norm_weight,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_k,
    T *qkv_out,
    T *q,
    T *k,
    T *v,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int max_model_len,
    const int head_dim,
    const bool rope_3d,
    const float rms_norm_eps) {
  using LoadT = AlignedVector<T, VecSize>;
  constexpr int HalfVecSize = VecSize / 2;
  using LoadEmbT = AlignedVector<float, HalfVecSize>;
  using LoadFloat = AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  LoadFloat tmp_vec;
  LoadFloat q_norm_vec, k_norm_vec;
  int64_t global_warp_idx = blockDim.y * blockIdx.x + threadIdx.y;
  int64_t all_warp_num = gridDim.x * blockDim.y;
  const int half_headdim = head_dim / 2;
  const int offset =
      (q_num_head + kv_num_head * 2) * head_dim;  // for all q,k,v
  const int all_head_num = elem_cnt / head_dim;
  for (int gloabl_hi = global_warp_idx; gloabl_hi < all_head_num;
       gloabl_hi += all_warp_num) {
    int64_t linear_index =
        gloabl_hi * head_dim + threadIdx.x * VecSize;  // 全局index
    const int token_idx =
        linear_index / offset;  // token id(第几个token,不分qkv)
    const int ori_bi = batch_id_per_token[token_idx];  // 第几个batch

    int cache_kv_len = seq_lens_decoder[ori_bi];
    // 这里其实是不需要处理的，但是由于FA3的bug，所以必须！
    if (seq_lens_encoder[ori_bi] == 0) cache_kv_len = 0;

    const int bias = linear_index % offset;
    const int hi = bias / head_dim;
    const int h_bias = bias % head_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) +
        cache_kv_len;  // 在当前seq中的id(拼接了seq到一个batch的情况下有效)
    const int64_t emb_idx =
        ori_seq_id * half_headdim + h_bias / 2;  // embedding的id
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * head_dim + hi * head_dim +
        h_bias;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;
    int64_t base_split_idx;
    T *out_p = nullptr;
    if (hi < q_num_head) {
      base_split_idx =
          token_idx * q_num_head * head_dim + hi * head_dim + h_bias;
      out_p = q;
    } else if (hi < q_num_head + kv_num_head) {
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head) * head_dim + h_bias;
      out_p = k;
    } else {
      out_p = v;
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head - kv_num_head) * head_dim + h_bias;
    }

    // TODO check this correct or not
    int64_t new_emb_idx =
        rope_3d ? emb_idx + ori_bi * head_dim * max_model_len : emb_idx;
    float thread_m2 = 0.0f;
    float warp_m2 = 0.0f;

    if (q_norm_weight && k_norm_weight) {
      if (hi < q_num_head + kv_num_head) {  // only q and k need rope
        Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
        Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
        for (int i = 0; i < HalfVecSize; i++) {
          const float input_left = static_cast<float>(src_vec[2 * i]);
          const float input_right = static_cast<float>(src_vec[2 * i + 1]);
          const float cos_tmp = cos_emb_vec[i];
          const float sin_tmp = sin_emb_vec[i];
          float tmp1 = input_left * cos_tmp - input_right * sin_tmp;
          float tmp2 = input_right * cos_tmp + input_left * sin_tmp;
          tmp_vec[2 * i] = tmp1;
          tmp_vec[2 * i + 1] = tmp2;
          thread_m2 += tmp1 * tmp1 + tmp2 * tmp2;
        }
      }
      WelfordWarpAllReduce<float, 32>(thread_m2, &warp_m2);  // 单个head的标准差

      if (hi < q_num_head + kv_num_head) {  // only q and k need norm
        float row_variance = max(warp_m2 / head_dim, 0.0f);
        float row_inv_var = Rsqrt(row_variance + rms_norm_eps);
        if (hi < q_num_head) {
          Load<float, VecSize>(&q_norm_weight[threadIdx.x * VecSize],
                               &q_norm_vec);
#pragma unroll
          for (int i = 0; i < VecSize; i++) {
            src_vec[i] =
                static_cast<T>(tmp_vec[i] * row_inv_var * q_norm_vec[i]);
          }
        } else {
          Load<float, VecSize>(&k_norm_weight[threadIdx.x * VecSize],
                               &k_norm_vec);
          for (int i = 0; i < VecSize; i++) {
            src_vec[i] =
                static_cast<T>(tmp_vec[i] * row_inv_var * k_norm_vec[i]);
          }
        }
      }
    } else {
      if (hi < q_num_head + kv_num_head) {
        Load<float, HalfVecSize>(&cos_emb[new_emb_idx], &cos_emb_vec);
        Load<float, HalfVecSize>(&sin_emb[new_emb_idx], &sin_emb_vec);
#pragma unroll
        for (int i = 0; i < HalfVecSize; i++) {
          const float input_left = static_cast<float>(src_vec[2 * i]);
          const float input_right = static_cast<float>(src_vec[2 * i + 1]);
          const float cos_tmp = cos_emb_vec[i];
          const float sin_tmp = sin_emb_vec[i];
          src_vec[2 * i] =
              static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          src_vec[2 * i + 1] =
              static_cast<T>(input_right * cos_tmp + input_left * sin_tmp);
        }
      }
    }
    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
    Store<T, VecSize>(src_vec, &out_p[base_split_idx]);
  }
}

template <typename T>
void gqa_rotary_qk_split_variable(
    T *qkv_out,  // [token_num, 3, num_head, head_dim]
    T *q,
    T *k,
    T *v,
    const T *qkv_input,
    const float *rotary_emb,  // [2, 1, seq_len, 1, head_dim / 2]
    const float *q_norm_weight,
    const float *k_norm_weight,
    const int *batch_id_per_token,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int max_model_len,
    const int input_output_len,
    const int head_dim,
    const bool rope_3d,
    const float rms_norm_eps,
    const cudaStream_t &stream) {
  assert(head_dim == 128 && "head_dim must be 128");
  int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * head_dim;

  constexpr int HEAD_DIM = 128;
  constexpr int PackSize = HEAD_DIM / kWarpSize;
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  dim3 block_size(kWarpSize, blocksize / kWarpSize);

  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + input_output_len * head_dim / 2;
  launchWithPdlWhenEnabled(GQAVariableLengthRotarySplitKernel<T, PackSize>,
                           grid_size,
                           block_size,
                           0,
                           stream,
                           qkv_input,
                           cos_emb,
                           sin_emb,
                           q_norm_weight,
                           k_norm_weight,
                           batch_id_per_token,
                           cu_seqlens_q,
                           seq_lens_encoder,
                           seq_lens_decoder,
                           cu_seqlens_k,
                           qkv_out,
                           q,
                           k,
                           v,
                           elem_nums,
                           num_heads,
                           kv_num_heads,
                           max_model_len,
                           head_dim,
                           rope_3d,
                           rms_norm_eps);
}

template <typename T, int VecSize = 1>
__global__ void GQAVariableLengthNeoxPartialRotarySplitKernel(
    const T *qkv,
    const float *cos_emb,
    const float *sin_emb,
    const int *batch_id_per_token,
    const int *cu_seqlens_q,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_k,
    T *qkv_out,
    T *q,
    T *k,
    T *v,
    const int64_t elem_cnt,
    const int q_num_head,
    const int kv_num_head,
    const int max_model_len,
    const int head_dim,
    const int rotary_dim) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadEmbT = AlignedVector<float, VecSize>;
  LoadT src_vec;
  LoadT src_vec_right;
  LoadEmbT cos_emb_vec;
  LoadEmbT sin_emb_vec;
  int64_t global_warp_idx = blockDim.y * blockIdx.x + threadIdx.y;
  int64_t all_warp_num = gridDim.x * blockDim.y;
  const int half_rotary_dim = rotary_dim / 2;
  const int half_headdim = head_dim / 2;
  const int offset =
      (q_num_head + kv_num_head * 2) * head_dim;  // for all q,k,v
  const int all_head_num = elem_cnt / head_dim;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int gloabl_hi = global_warp_idx; gloabl_hi < all_head_num;
       gloabl_hi += all_warp_num) {
    int64_t linear_index =
        gloabl_hi * head_dim + threadIdx.x * VecSize;  // 全局index
    const int token_idx =
        linear_index / offset;  // token id(第几个token,不分qkv)
    const int ori_bi = batch_id_per_token[token_idx];  // 第几个batch

    int cache_kv_len = seq_lens_decoder[ori_bi];
    // 这里其实是不需要处理的，但是由于FA3的bug，所以必须！
    if (seq_lens_encoder[ori_bi] == 0) cache_kv_len = 0;

    const int bias = linear_index % offset;
    const int hi = bias / head_dim;
    const int h_bias = bias % head_dim;

    const int ori_seq_id =
        (token_idx - cu_seqlens_q[ori_bi]) +
        cache_kv_len;  // 在当前seq中的id(拼接了seq到一个batch的情况下有效)
    const int64_t base_idx =
        token_idx * (q_num_head + 2 * kv_num_head) * head_dim + hi * head_dim +
        h_bias;
    Load<T, VecSize>(&qkv[base_idx], &src_vec);
    const int kv_write_idx = cu_seqlens_k[ori_bi] + ori_seq_id;
    int64_t base_split_idx;
    T *out_p = nullptr;
    if (hi < q_num_head) {
      base_split_idx =
          token_idx * q_num_head * head_dim + hi * head_dim + h_bias;
      out_p = q;
    } else if (hi < q_num_head + kv_num_head) {
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head) * head_dim + h_bias;
      out_p = k;
    } else {
      out_p = v;
      base_split_idx = kv_write_idx * kv_num_head * head_dim +
                       (hi - q_num_head - kv_num_head) * head_dim + h_bias;
    }

    if (hi < q_num_head + kv_num_head) {
      if (h_bias < rotary_dim) {
        int64_t emb_idx = ori_seq_id * half_rotary_dim;
        if (h_bias < half_rotary_dim) {
          Load<T, VecSize>(&qkv[base_idx + half_rotary_dim], &src_vec_right);
          emb_idx += h_bias;
        } else {
          Load<T, VecSize>(&qkv[base_idx - half_rotary_dim], &src_vec_right);
          emb_idx += h_bias - half_rotary_dim;
        }
        Load<float, VecSize>(&cos_emb[emb_idx], &cos_emb_vec);
        Load<float, VecSize>(&sin_emb[emb_idx], &sin_emb_vec);
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          const float input_left = static_cast<float>(src_vec[i]);
          const float input_right = static_cast<float>(src_vec_right[i]);
          const float cos_tmp = cos_emb_vec[i];
          const float sin_tmp = sin_emb_vec[i];
          if (h_bias < half_rotary_dim) {
            src_vec[i] =
                static_cast<T>(input_left * cos_tmp - input_right * sin_tmp);
          } else {
            src_vec[i] =
                static_cast<T>(input_left * cos_tmp + input_right * sin_tmp);
          }
        }
      }
    }

    Store<T, VecSize>(src_vec, &qkv_out[base_idx]);
    Store<T, VecSize>(src_vec, &out_p[base_split_idx]);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T>
void gqa_neox_partial_rotary_qk_split_variable(
    T *qkv_out,  // [token_num, 3, num_head, head_dim]
    T *q,
    T *k,
    T *v,
    const T *qkv_input,
    const float *rotary_emb,  // [2, 1, seq_len, 1, head_dim / 4]
    const int *batch_id_per_token,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int *cu_seqlens_q,
    const int *cu_seqlens_k,
    const int token_num,
    const int num_heads,
    const int kv_num_heads,
    const int max_model_len,
    const int head_dim,
    const int rotary_dim,
    const cudaStream_t &stream) {
  assert(head_dim == 128 && "head_dim must be 128");
  int64_t elem_nums = token_num * (num_heads + 2 * kv_num_heads) * head_dim;

  constexpr int HEAD_DIM = 128;
  constexpr int PackSize = HEAD_DIM / kWarpSize;
  assert(rotary_dim / 2 % PackSize == 0);
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  dim3 block_size(kWarpSize, blocksize / kWarpSize);

  const float *cos_emb = rotary_emb;
  const float *sin_emb = rotary_emb + max_model_len * rotary_dim / 2;
  launchWithPdlWhenEnabled(
      GQAVariableLengthNeoxPartialRotarySplitKernel<T, PackSize>,
      grid_size,
      block_size,
      0,
      stream,
      qkv_input,
      cos_emb,
      sin_emb,
      batch_id_per_token,
      cu_seqlens_q,
      seq_lens_encoder,
      seq_lens_decoder,
      cu_seqlens_k,
      qkv_out,
      q,
      k,
      v,
      elem_nums,
      num_heads,
      kv_num_heads,
      max_model_len,
      head_dim,
      rotary_dim);
}

