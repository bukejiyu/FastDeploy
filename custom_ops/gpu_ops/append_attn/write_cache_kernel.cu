// // Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.


// #include "encoder_write_cache_with_rope_impl.cuh"
// #include "helper.h"
// #include "paddle/extension.h"
// #include "paddle/phi/backends/context_pool.h"
// #include "paddle/phi/core/memory/memcpy.h"
// #include "qwen3_rope.h"
// #include "remote_cache_kv_ipc.h"
// #include "gqa_rope_write_cache_impl.cuh"


// void WriteCacheKernel()
// // write cache
//   if (cache_quant_type == "none") {
//     CascadeAppendWriteCacheKVQKV<data_t>(
//         meta_data,
//         qkv_out,
//         block_tables,
//         batch_id_per_token,
//         cu_seqlens_q,
//         seq_lens_encoder,
//         seq_lens_decoder,
//         max_seq_len,
//         stream,
//         const_cast<paddle::Tensor *>(&key_cache),
//         const_cast<paddle::Tensor *>(&value_cache));
//   } else if (cache_quant_type == "cache_int8" ||
//              cache_quant_type == "cache_fp8" ||
//              cache_quant_type == "block_wise_fp8") {
//     CascadeAppendWriteCacheKVC8QKV<data_t, 128, 64>(
//         meta_data,
//         *const_cast<paddle::Tensor *>(&key_cache),
//         *const_cast<paddle::Tensor *>(&value_cache),
//         qkv_out,
//         cache_k_quant_scales.get(),
//         cache_v_quant_scales.get(),
//         seq_lens_this_time,
//         seq_lens_decoder,
//         batch_id_per_token,
//         cu_seqlens_q,
//         block_tables,
//         kv_batch_ids,
//         kv_tile_ids,
//         kv_num_blocks_data,
//         max_seq_len,
//         false,  // is_scale_channel_wise
//         cache_quant_type,
//         stream,
//         const_cast<paddle::Tensor *>(&key_cache),
//         const_cast<paddle::Tensor *>(&value_cache));
//   } else if (cache_quant_type == "cache_int4_zp") {
//     CascadeAppendWriteCacheKVC4QKV<data_t, 128, 64>(
//         meta_data,
//         *const_cast<paddle::Tensor *>(&key_cache),
//         *const_cast<paddle::Tensor *>(&value_cache),
//         qkv_out,
//         cache_k_quant_scales.get(),
//         cache_v_quant_scales.get(),
//         cache_k_zp.get(),
//         cache_v_zp.get(),
//         seq_lens_this_time,
//         seq_lens_decoder,
//         batch_id_per_token,
//         cu_seqlens_q,
//         block_tables,
//         kv_batch_ids,
//         kv_tile_ids,
//         kv_num_blocks_data,
//         max_seq_len,
//         stream,
//         const_cast<paddle::Tensor *>(&key_cache),
//         const_cast<paddle::Tensor *>(&value_cache));
//   } else {
//     PD_THROW(
//         "cache_quant_type_str should be one of [none, cache_int8, cache_fp8, "
//         "cache_int4_zp]");
//   }