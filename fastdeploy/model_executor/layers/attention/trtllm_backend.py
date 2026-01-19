"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import paddle



# try:
#     from paddle.nn.functional.flash_attention import flash_attention_v3_varlen
# except:
#     flash_attention_v3_varlen = None

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    init_kv_signal_per_query,
    init_signal_layerwise,
    open_shm_and_get_meta_signal,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.platforms import current_platform

if current_platform.is_cuda():
    from fastdeploy.model_executor.ops.gpu import merge_prefill_decode_output
    from fastdeploy.model_executor.ops.gpu import decoder_write_cache_with_rope

else:
    merge_prefill_decode_output = None

import os
from fastdeploy.model_executor.layers.attention.utils import  split_decodes_and_prefills


@dataclass
class TrtllmAttentionMetadata(AttentionMetadata):
    """
    TrtllmAttentionMetadata
    """
    num_decodes:int = 0
    num_prefills:int = 0
    num_decode_tokens:int = 0
    num_prefill_tokens:int = 0
    enable_ids_reorder:bool = True

    cu_seqlens_k: paddle.Tensor = None

    pre_cache_batch_ids = None
    pre_cache_tile_ids_per_batch = None
    pre_cache_num_blocks_cpu = None
    kv_token_num_cpu = None

    _fuse_kernel_compute_dtype: str = "bf16"
    _dtype: paddle.dtype = paddle.bfloat16

    active_seq_lens_decoder: paddle.Tensor=None
    active_seq_lens_encoder: paddle.Tensor=None
    active_seq_lens_this_time: paddle.Tensor=None
    active_total_seq_len: paddle.Tensor=None
    active_block_kv_indptr_gpu: paddle.Tensor=None
    active_block_tables:paddle.Tensor = None

    num_running_requests:int = 0
    num_decodes:int = 0
    num_prefills:int = 0
    num_decode_tokens:int = 0
    num_prefill_tokens:int = 0

    #为了过滤掉seq this time 等于0的情况
    filter_mask: paddle.Tensor = None



save_step_id = 1

class TrtllmAttentionBackend(AttentionBackend):
    """
    TrtllmAttentionBackend backend implementation
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: TrtllmAttentionMetadata
    flash_attn_func: callable = None
    use_output:bool =True
    enable_ids_reorder:bool = True

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
        sliding_window:int|None = None,
    ):
        """
        TrtllmAttentionBackend __init__
        """
        super().__init__()
        self.attention_metadata: TrtllmAttentionMetadata = None
        self.max_seq_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.attn_outputsize_tp = self.num_heads * self.head_dim
        self.block_size = fd_config.cache_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate = self.speculative_method is not None
        self.speculate_max_draft_token_num = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )

        # if self.flash_attn_func is None:
        #     prop = paddle.device.cuda.get_device_properties()
        #     cc = prop.major * 10 + prop.minor
        #     is_current_sm_supported = cc >= 90
        #     is_paddle_supported = any(num >= 90 for num in paddle.version.cuda_archs())
        #     if is_current_sm_supported and is_paddle_supported:
        #         self.flash_attn_func = flash_attention_v3_varlen
        #         print("The current platform supports Flash Attention V3.")
        #         self.flash_attn_kwargs = {}
        #     else:
        #         self.flash_attn_func = flash_attn_unpadded
        #         self.flash_attn_kwargs = {"scale": self.head_dim**-0.5, "training": False}
        #         print(
        #             "The current platform does not support Flash Attention V3, so Flash Attention V2 will be used instead."
        #         )
        self.rope_3d: bool = getattr(fd_config.model_config, "rope_3d", False)
        # Note(ZKK): here must be consistent with append_attn_backend.py
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 1024))
        self.zero_seq_enc_lens_for_decode = paddle.zeros(
            shape=[fd_config.scheduler_config.max_num_seqs, 1], dtype=paddle.int32
        )
        self.block_kv_indptr_gpu = paddle.zeros(shape=[fd_config.scheduler_config.max_num_seqs+1, 1], dtype=paddle.int32)
        

    def get_attention_meta(self):
        """get_attention_meta"""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """
        Calculate kv cache shape
        """
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        value_cache_shape = key_cache_shape
        return key_cache_shape, value_cache_shape

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        metadata = TrtllmAttentionMetadata()
        metadata.num_decodes, metadata.num_prefills, metadata.num_decode_tokens, metadata.num_prefill_tokens = (
            split_decodes_and_prefills(
                forward_meta,
            )
        )
        metadata.num_running_requests = metadata.num_decodes+metadata.num_prefills
        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            forward_meta.decoder_num_blocks_cpu,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            forward_meta.encoder_batch_ids,
            forward_meta.encoder_tile_ids_per_batch,
            forward_meta.encoder_num_blocks_x_cpu,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            self.encoder_block_shape_q,
            self.decoder_block_shape_q,
            self.group_size,
            self.block_size,
        )
        # pd_disaggregation
        # print("forward_meta.max_len_tensor_cpu",forward_meta.max_len_tensor_cpu)
        # if forward_meta.max_len_tensor_cpu[1] > 0:
        (
            metadata.cu_seqlens_k,
            metadata.pre_cache_batch_ids,
            metadata.pre_cache_tile_ids_per_batch,
            metadata.pre_cache_num_blocks_cpu,
            metadata.kv_token_num_cpu,
        ) = pre_cache_len_concat(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.max_len_tensor_cpu[2],
            self.block_size,
        )
        
        metadata.kv_signal_data_list = [None] * self.num_layers

        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"
        
        metadata.active_seq_lens_decoder = forward_meta.seq_lens_decoder[:metadata.num_running_requests]
        metadata.active_seq_lens_encoder = forward_meta.seq_lens_encoder[:metadata.num_running_requests]
        metadata.active_seq_lens_this_time = forward_meta.seq_lens_this_time[:metadata.num_running_requests]
        metadata.active_total_seq_len = metadata.active_seq_lens_decoder + metadata.active_seq_lens_this_time
        num_blocks = ( metadata.active_total_seq_len+ (self.block_size - 1)) // self.block_size
        #for trtllm prefill
        metadata.active_block_kv_indptr_gpu = self.block_kv_indptr_gpu[1:metadata.num_running_requests+1]
        metadata.active_block_kv_indptr_gpu = paddle.cumsum(num_blocks)
        metadata.active_block_tables = forward_meta.block_tables[:metadata.num_running_requests]
        metadata.filter_mask = metadata.active_seq_lens_this_time.squeeze(-1) != 0
        self.attention_metadata: AttentionMetadata = metadata


    
    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
        **kwargs,
    ):
        metadata = self.attention_metadata
        # if layer.layer_id == 0:
            
        #     print("forward_meta.seq_lens_this_time:",forward_meta.seq_lens_this_time)
        #     print("forward_meta.seq_lens_encoder:",forward_meta.seq_lens_encoder)
        #     print("forward_meta.seq_lens_decoder:",forward_meta.seq_lens_decoder)
        #     print("forward_meta.batch_id_per_token:",forward_meta.batch_id_per_token)
        #     print("forward_meta.kv_batch_ids:",forward_meta.kv_batch_ids)
        #     print("forward_meta.kv_tile_ids_per_batch:",forward_meta.kv_tile_ids_per_batch)
        #     print("forward_meta.kv_num_blocks_x_cpu:",forward_meta.kv_num_blocks_x_cpu)
        #     print("metadata.kv_token_num_cpu",metadata.kv_token_num_cpu)
        #     print("block_tables:",forward_meta.block_tables)
        # q, _, _, _ = gqa_rope_write_cache(
        #         qkv,
        #         forward_meta.caches[2 * layer.layer_id],
        #         forward_meta.caches[2 * layer.layer_id + 1],
        #         forward_meta.cu_seqlens_q,
        #         metadata.cu_seqlens_k,
        #         forward_meta.rotary_embs,
        #         forward_meta.seq_lens_this_time,
        #         forward_meta.seq_lens_encoder,
        #         forward_meta.seq_lens_decoder,
        #         forward_meta.batch_id_per_token,
        #         forward_meta.block_tables,
        #         forward_meta.kv_batch_ids,
        #         forward_meta.kv_tile_ids_per_batch,
        #         forward_meta.kv_num_blocks_x_cpu,
        #         metadata.pre_cache_batch_ids,
        #         metadata.pre_cache_tile_ids_per_batch,
        #         metadata.pre_cache_num_blocks_cpu,
        #         getattr(layer, "q_norm_weight", None),
        #         getattr(layer, "k_norm_weight", None),
        #         getattr(layer, "cache_k_scale", None),
        #         getattr(layer, "cache_v_scale", None),
        #         getattr(layer, "cache_k_out_scale", None),
        #         getattr(layer, "cache_v_out_scale", None),
        #         getattr(layer, "cache_k_zp", None),
        #         getattr(layer, "cache_v_zp", None),
        #         None,
        #         metadata.kv_token_num_cpu[0].item(),
        #         self.max_seq_len,
        #         getattr(layer, "rms_norm_eps", 1e-6),
        #         layer.use_neox_rotary_style,
        #         getattr(layer, "cache_quant_type_str", "none"),
        #         self.rope_3d,
        #     )
        # if layer.layer_id == 17:
        #     print("qkv:",qkv)
        
        output = kwargs.get("output")
        attention_sink = kwargs.get("sinks", None)
        workspace_buffer = paddle.empty(394 * 1024 * 1024, dtype=paddle.int8)
        self.bmm1_scale = float(1.0 / (self.head_dim**0.5))
        paddle.compat.enable_torch_proxy()
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache
        from flashinfer.decode import  trtllm_batch_decode_with_kv_cache
        if layer.layer_id == 0 :
            print(f"prefill_token_nums {metadata.num_prefill_tokens},decode_token_nums {metadata.num_decode_tokens} ,num_prefills {metadata.num_prefills},num_decodes {metadata.num_decodes} ")
        if metadata.num_prefill_tokens > 0:
            if layer.layer_id==0:
                print("++++++++++++runing prefill!!++++++++++++++")
            prefill_qkv = qkv.clone()
            q, _, _, _ = gqa_rope_write_cache(
                prefill_qkv,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                forward_meta.cu_seqlens_q,
                metadata.cu_seqlens_k,
                forward_meta.rotary_embs,
                forward_meta.seq_lens_this_time,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.block_tables,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                metadata.pre_cache_batch_ids,
                metadata.pre_cache_tile_ids_per_batch,
                metadata.pre_cache_num_blocks_cpu,
                getattr(layer, "q_norm_weight", None),
                getattr(layer, "k_norm_weight", None),
                getattr(layer, "cache_k_scale", None),
                getattr(layer, "cache_v_scale", None),
                getattr(layer, "cache_k_out_scale", None),
                getattr(layer, "cache_v_out_scale", None),
                getattr(layer, "cache_k_zp", None),
                getattr(layer, "cache_v_zp", None),
                None,
                metadata.kv_token_num_cpu[0].item(),
                self.max_seq_len,
                getattr(layer, "rms_norm_eps", 1e-6),
                layer.use_neox_rotary_style,
                getattr(layer, "cache_quant_type_str", "none"),
                self.rope_3d,
            )
            prefill_start = metadata.num_decodes
            prefill_q = q[metadata.num_decode_tokens:]
            # seq_lens_prefill = forward_meta.seq_lens_encoder[metadata.num_decode_tokens:]
            seq_lens_prefill =metadata.active_seq_lens_encoder[prefill_start:][metadata.filter_mask[prefill_start:]]
            cu_seqlens_q = forward_meta.cu_seqlens_q[metadata.num_decode_tokens:]
            cu_seqlens_q = cu_seqlens_q-metadata.num_decode_tokens
            #cu_seqlens_k = bs + 1
            # if layer.layer_id == 0:
            #     print("metadata.cu_seqlens_k:",metadata.cu_seqlens_k)
            # seq_lens_decoder = forward_meta.seq_lens_decoder[:metadata.num_running_requests][prefill_start:]
            # seq_lens_decoder = metadata.active_seq_lens_decoder[prefill_start:]
            # print("seq_lens_decoder:",seq_lens_decoder)
            # seq_lens_this_time = forward_meta.seq_lens_this_time[prefill_start:]
            seq_lens_this_time = metadata.active_seq_lens_this_time[prefill_start:][metadata.filter_mask[prefill_start:]]

            # print("prefill 的 seq_lens_this_time:",seq_lens_this_time)
            # print("prefill 的 seq_lens_decoder:",seq_lens_decoder)
            # total_seq_len = seq_lens_decoder + seq_lens_this_time
            # print("total_seq_len:",total_seq_len)
            # num_blocks = ( total_seq_len+ (self.block_size - 1)) // self.block_size
            # self.block_kv_indptr_gpu[1:metadata.num_running_requests+1]=paddle.cumsum(num_blocks)
            out = output[metadata.num_decode_tokens:]
            # mock_block_tables_prefill = forward_meta.block_tables[:metadata.num_running_requests][prefill_start:]
            mock_block_tables_prefill = metadata.active_block_tables[prefill_start:][metadata.filter_mask[prefill_start:]]
            # if layer.layer_id == 0:
            #     print("block_kv_indptr_gpu:",self.block_kv_indptr_gpu)
            # assert self.block_kv_indptr_gpu.shape[0] == metadata.num_prefills + 1

            
            prefill_q = prefill_q.contiguous()
            seq_lens_prefill=seq_lens_prefill.contiguous()
            cu_seqlens_q=cu_seqlens_q.contiguous()

            assert prefill_q.is_contiguous()
            assert forward_meta.caches[2 * layer.layer_id].is_contiguous()
            assert forward_meta.caches[2 * layer.layer_id + 1].is_contiguous()
            assert workspace_buffer.is_contiguous()
            assert mock_block_tables_prefill.is_contiguous()
            assert seq_lens_prefill.is_contiguous()


            #num_blocks, num_kv_heads, page_size, head_dim

            if layer.layer_id == 0:
                print("prefill_q:",prefill_q)
                print("max_q_len:",forward_meta.max_len_tensor_cpu[0])
                print("max_kv_len:",forward_meta.max_len_tensor_cpu[3])
                print("cum_seq_lens_q:",cu_seqlens_q)
                print("cum_seq_lens_kv:",self.block_kv_indptr_gpu)
                print("block_table:",forward_meta.block_tables)
                print("seq_lens_prefill:",seq_lens_prefill)
                # print("cache_k:",forward_meta.caches[2 * layer.layer_id])
                # print("cache_v:",forward_meta.caches[2 * layer.layer_id+1])
            #     paddle.save({"q":q},"/workspace3/tbh/FastDeploy/prefill/q.pdparams")
            #     paddle.save({"qkv":qkv},"/workspace3/tbh/FastDeploy/prefill/qkv.pdparams")
            #     paddle.save({"cache_k":forward_meta.caches[2 * layer.layer_id]},"/workspace3/tbh/FastDeploy/prefill/cache_k.pdparams")
            #     paddle.save({"cache_v":forward_meta.caches[2 * layer.layer_id+1]},"/workspace3/tbh/FastDeploy/prefill/cache_v.pdparams")
            trtllm_batch_context_with_kv_cache(
                query=prefill_q,
                kv_cache=(forward_meta.caches[2 * layer.layer_id], forward_meta.caches[2 * layer.layer_id + 1]),
                workspace_buffer=workspace_buffer,
                block_tables=mock_block_tables_prefill,
                seq_lens=seq_lens_prefill,
                max_q_len=forward_meta.max_len_tensor_cpu[0].item(),
                max_kv_len=forward_meta.max_len_tensor_cpu[3].item(),
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=1.0,
                batch_size=metadata.num_prefills,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=self.block_kv_indptr_gpu,
                # window_left=-1,
                out=out,
                # sinks=paddle.to_tensor([]),
                # nvfp4 
                # o_sf_scale/o_sf_vec_size
            )
        
        # if metadata.num_decode_tokens>0:
        #     print("++++++++++++runing decoder!!++++++++++++++")
            
            # res_decoder = append_attention(
            #     qkv,
            #     forward_meta.caches[2 * layer.layer_id],
            #     forward_meta.caches[2 * layer.layer_id + 1],
            #     forward_meta.seq_lens_encoder,
            #     forward_meta.seq_lens_decoder,
            #     forward_meta.seq_lens_this_time,
            #     forward_meta.batch_id_per_token,
            #     forward_meta.cu_seqlens_q,
            #     forward_meta.block_tables,
            #     forward_meta.encoder_batch_ids,
            #     forward_meta.encoder_tile_ids_per_batch,
            #     forward_meta.encoder_num_blocks_x_cpu,
            #     forward_meta.kv_batch_ids,
            #     forward_meta.kv_tile_ids_per_batch,
            #     forward_meta.kv_num_blocks_x_cpu,
            #     forward_meta.decoder_batch_ids,
            #     forward_meta.decoder_tile_ids_per_batch,
            #     forward_meta.decoder_num_blocks_cpu,
            #     forward_meta.max_len_tensor_cpu,
            #     forward_meta.rotary_embs,
            #     forward_meta.attn_mask,
            #     layer.qkv_bias,
            #     layer.qkv_scale,
            #     getattr(layer, "cache_k_scale", None),
            #     getattr(layer, "cache_v_scale", None),
            #     getattr(layer, "cache_k_out_scale", None),
            #     getattr(layer, "cache_v_out_scale", None),
            #     getattr(layer, "cache_k_zp", None),
            #     getattr(layer, "cache_v_zp", None),
            #     layer.linear_shift,
            #     layer.linear_smooth,
            #     forward_meta.attn_mask_offsets,
            #     metadata.kv_signal_data_list[layer.layer_id],
            #     getattr(layer, "q_norm_weight", None),
            #     getattr(layer, "k_norm_weight", None),
            #     getattr(layer, "sinks", None),
            #     getattr(layer, "rms_norm_eps", 1e-6),
            #     metadata._fuse_kernel_compute_dtype,
            #     getattr(layer, "cache_quant_type_str", "none"),
            #     layer.use_neox_rotary_style,
            #     self.rope_3d,
            #     self.max_seq_len,
            #     getattr(layer, "quant_max_bound", 0.0),
            #     getattr(layer, "quant_min_bound", 0.0),
            #     getattr(layer, "out_scale", -1.0),
            #     self.encoder_block_shape_q,
            #     self.decoder_block_shape_q,
            #     self.max_partition_size,
            #     self.max_seq_len,
            #     self.speculate_max_draft_token_num + 1,
            #     self.causal,
            #     self.speculative_method is not None,
            # )
            # output[:metadata.num_decode_tokens] = res_decoder.reshape([metadata.num_decode_tokens,layer.num_heads,self.head_dim])
            # print("res_decoder.shape:",res_decoder.shape)
        global save_step_id
        if metadata.num_decode_tokens>0:
            if layer.layer_id==0:
                print("++++++++++++runing decoder!!++++++++++++++")
            # if layer.layer_id==0:
                # paddle.save({"qkv":qkv},f"/workspace3/tbh/FastDeploy/decoder/qkv_{save_step_id}.pdparams")
            decoder_qkv = qkv.clone()
            qkv_out=decoder_write_cache_with_rope(
                decoder_qkv,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                forward_meta.block_tables,
                forward_meta.max_len_tensor_cpu,
                forward_meta.rotary_embs,  # rotary_embs
                None,  # qkv_bias
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # kv_signal_data
                None,  # q_norm_weight
                None,  # k_norm_weight
                1e-6,
                "none",
                False,  # use_neox_rotary_style
                False,
                self.max_seq_len,
                0.0,  # quant_max_bound
                0.0,  # quant_min_bound
                False,  # speculate_decoder
            )
            decode_end = metadata.num_decodes
            decode_q = qkv_out[:,:self.num_heads*self.head_dim][:metadata.num_decode_tokens].reshape([-1,self.num_heads,self.head_dim])
            decode_q=decode_q.contiguous()
            mock_block_tables_decode = metadata.active_block_tables[:decode_end][metadata.filter_mask[:decode_end]]
            total_seq_len=metadata.active_total_seq_len[:decode_end][metadata.filter_mask[:decode_end]]
            out = output[:metadata.num_decode_tokens]
            if layer.layer_id == 0:
                print("seq_lens:",total_seq_len)
                print("max_seq_len:",forward_meta.max_len_tensor_cpu[3].item())
                print("block_tables:",mock_block_tables_decode)
                print("forward_meta.seq_lens_encoder:",forward_meta.seq_lens_encoder)
                print("forward_meta.seq_lens_decoder:",forward_meta.seq_lens_decoder)
                print("forward_meta.seq_lens_this_time:",forward_meta.seq_lens_this_time)
                print("forward_meta.cu_seqlens_q:",forward_meta.cu_seqlens_q)
                print("forward_meta.batch_id_per_token:",forward_meta.batch_id_per_token)
                print("forward_meta.kv_num_blocks_x_cpu:",forward_meta.kv_num_blocks_x_cpu)
                print("decoder 的 max_len_tensor_cpu:",forward_meta.max_len_tensor_cpu)
                # print("输入进trtllm的 seq_lens_decoder:",seq_lens_decoder)
                # print("输入进trtllm的 seq_lens_this_time:",seq_lens_this_time)
                print("输入进trtllm的 total_seq_len:",total_seq_len)
                # paddle.save({"q":decode_q},f"/workspace3/tbh/FastDeploy/decoder/q_{save_step_id}.pdparams")
                # paddle.save({"cache_k":forward_meta.caches[2 * layer.layer_id]},f"/workspace3/tbh/FastDeploy/decoder/cache_k.pdparams")
                # paddle.save({"cache_v":forward_meta.caches[2 * layer.layer_id+1]},f"/workspace3/tbh/FastDeploy/decoder/cache_v.pdparams")
                save_step_id+=1



            # assert decode_q.shape[0] == metadata.num_decode_tokens

            trtllm_batch_decode_with_kv_cache(
                query=decode_q,
                kv_cache=(forward_meta.caches[2 * layer.layer_id], forward_meta.caches[2 * layer.layer_id + 1]),
                workspace_buffer=workspace_buffer,
                block_tables=mock_block_tables_decode,
                seq_lens=total_seq_len,
                max_seq_len=forward_meta.max_len_tensor_cpu[3].item(),
                # max_q_len=forward_meta.max_len_tensor_cpu[3],
                # max_kv_len=self.max_context_len,
                bmm1_scale=self.bmm1_scale,
                bmm2_scale=1.0,
                # cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                # cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                window_left=-1,
                # TODO: add attention_sink operation or nvfp4 scale factor if needed
                sinks=None,
                # max_q_len=forward_meta.max_len_tensor_cpu[0].item(),
                # cum_seq_lens_q=cu_seqlens_q,
                out=out,  
            )
        if layer.layer_id == 0:
            print("第0层最终的输出:",output.reshape([output.shape[0],-1]))
        return output.reshape([output.shape[0],-1])
            # trtllm_batch_decode_with_kv_cache(
            #     query=decode_q,
            #     kv_cache=(forward_meta.caches[2 * layer.layer_id], forward_meta.caches[2 * layer.layer_id + 1]),
            #     workspace_buffer=workspace_buffer,
            #     block_tables=mock_block_tables_decode,
            #     seq_lens=total_seq_len,
            #     max_seq_len=forward_meta.max_len_tensor_cpu[3],
            #     bmm1_scale=1.0,
            #     bmm2_scale=1.0,
            #     window_left=-1,
            #     sinks=None,
            #     # nvfp4 
            #     # o_sf_scale/o_sf_vec_size
            #     out=out  
            # )


        