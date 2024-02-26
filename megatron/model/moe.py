from megablocks.layers import common
from megablocks.layers import dmlp_registry
from megablocks.layers import mpu
from megablocks.layers import router
from megablocks.layers.all_to_all import all_to_all
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np

import torch.nn.init as init

from megatron.neox_arguments.arguments import NeoXArgs
from megatron import mpu

import torch

from megatron.model.activations import get_activation
from megatron.model.moe_mlp import GroupedMLP

from .router import LearnedRouter

def promote_scalar(x):
    return x.view(1) if not len(x.size()) else x


_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def batched_load_balancing_loss(args: Arguments):
    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} token_per_experts "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}"
        )
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} expert_scores "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}"
        )

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all(
        [x.ndim == 1 and x.numel() == args.moe_num_experts for x in tokens_per_expert]
    )

    tokens = expert_scores[0].shape[0]
    assert all(
        [
            (
                x.ndim == 2
                and x.shape[1] == args.moe_num_experts
                and x.shape[0] == tokens
            )
            for x in expert_scores
        ]
    )

    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    if args.moe_lbl_in_fp32:
        expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
    else:
        expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = args.moe_num_experts * args.moe_loss_weight
    scale_denominator = args.num_layers * tokens * args.moe_top_k
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)


# COPIED FROM moe.py above ParallelMLP class
# NOTE: This class defines MoE expert computation, using tensor (model) parallel size as the expert parallel size
# The implication of this is that the expert weights will only be sharded within a single node
class ParallelDroplessMLP(torch.nn.Module):
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method=init.xavier_normal_,
        output_layer_init_method=init.xavier_normal_,
        parallel_output=False,
    ):
        super(ParallelDroplessMLP, self).__init__()

        self.activation_func = get_activation(neox_args)

        # calculate feedforward size
        ff_mult = 4
        ff_dim = ff_mult * neox_args.hidden_size
        rows_per_rank = ff_dim * neox_args.moe_num_experts

        # Calculate the number of experts in total and the number of experts
        # owned by this rank.
        world_size = mpu.get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.top_k = neox_args.moe_top_k

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # start original ParallelDroplessMLP init
        self.hidden_size = neox_args.hidden_size
        self.ffn_hidden_size = neox_args.hidden_size
        self.blocking = 128
        self.mlp = GroupedMLP()

        # Calculate the number of bits needed to represent the column indices
        # in the intermediate sparse matrix.
        max_column_index = (self.ffn_hidden_size * self.num_experts) // self.blocking
        self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

        if self.args.bias:
            # Note that the output bias is not parallelized with expert
            # model parallelism.
            self.bias = torch.nn.Parameter(
                torch.empty(
                    args.hidden_size, device=args.device, dtype=common.dtype(args)
                )
            )
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def load_balancing_loss(self, tokens_per_expert, expert_scores):
        """Calculate the load balancing loss contribution."""
        assert len(expert_scores.size()) == 2
        tokens, num_experts = expert_scores.size()
        assert num_experts == self.num_experts
        assert len(tokens_per_expert.size()) == 1
        (num_experts,) = tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.top_k)
        return scale * torch.dot(
            tokens_per_expert.to(expert_scores.dtype), expert_scores.mean(dim=0)
        )

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
        self,
        x,
        tokens_per_expert,
        indices,
        bin_ids,
        expert_weights,
        bins,
        top_k,
    ):
        """
        grouped_permute_and_compute
        """
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.gather(x, indices, bin_ids, bins, top_k)

        # TODO: mask x here to this rank's experts only

        # Perform the expert computation, replacing this rank's experts results
        x = self.mlp(x, tokens_per_expert)

        # TODO: all gather masked results from across Tensor parallel ranks here

        # Un-route the data for the MoE output, so each rank has now has a complete, un-masked input
        # tensor and can continue independently
        return ops.scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
            self.args.quantize_scatter_num_bits,
        )

    def forward_once(self, x, expert_weights, expert_indices):
        """
        grouped_forward_once

            x: [sl, bs, hs]
            expert_weights: [sl * bs, top-k]
            expert_indices: [sl * bs, top-k]
        """
        
        # both are now (sl * bs * top_k)
        expert_weights = expert_weights.flatten()
        expert_indices = expert_indices.flatten()
        
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(
                expert_indices
            )

        out = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            self.args.moe_top_k,
        )
        return out, tokens_per_expert

    def forward(self, x, scores, expert_weights, expert_indices):
        in_shape = x.size()

        # Compute the experts.
        x, tokens_per_expert = self.forward_once(x, expert_weights, expert_indices)
        if self.training:
            save_load_balancing_loss((tokens_per_expert, scores))
        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias
        return x


def cast_if_autocast_enabled(tensor: torch.Tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class dMoE(torch.nn.Module):
    def __init__(self, neox_args: NeoXArgs, init_method, output_layer_init_method,):
        super(dMoE, self).__init__()

        # Token router.
        self.router = LearnedRouter(neox_args)

        # Expert computation helper.
        self.experts = ParallelDroplessMLP(neox_args,
            init_method,
            output_layer_init_method,
            parallel_output=False,
        )

    def forward(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments.
        scores, expert_weights, expert_indices = self.router(x)

        # Compute the experts.
        # return value should be 
        return self.experts(x, scores, expert_weights, expert_indices)











    # def parallel_forward_once(self, x, expert_weights, expert_indices):
    #     # NOTE: This function implements the same computation as forward_once
    #     # but with expert model parallelism.
    #     #
    #     # 1. Permute the tokens locally so that they are grouped by their
    #     # expert assignments. This allows us to transfer all of the tokens
    #     # for a remote device in one communication primitive.
    #     #
    #     # 2. Permute the tokens across the expert parallel devices. After
    #     # this is completed each device has all of the tokens assigned to
    #     # its set of experts in its local HBM.
    #     #
    #     # 3. Permute the tokens locally so that they are grouped by their
    #     # expert assignement. After the distributed permutation the tokens
    #     # are grouped by which device they came from. We re-order them
    #     # locally to allow for efficient computation.
    #     #
    #     # After this series of permutations we compute the linear layers
    #     # and then repeat these three steps in reverse to produce the final
    #     # output.
    #     #
    #     # Compute the mapping of local tokens to experts.
    #     expert_weights = expert_weights.flatten()
    #     expert_indices = expert_indices.flatten()
    #     with torch.no_grad():
    #         indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(
    #             expert_indices
    #         )

    #         # If we're sharding the experts along the hidden dimension
    #         # multiple devices own parts of the same sets of experts.
    #         # Replicate the token counts so every device gets the counts.
    #         repeated_tokens_per_expert = ops.repeat(
    #             tokens_per_expert, (mpu.hidden_sharding_degree(self.args),)
    #         )

    #         # Pass token count information to the device on which the
    #         # target expert resides.
    #         parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)
    #         tpe_handle = torch.distributed.all_to_all_single(
    #             parallel_tokens_per_expert,
    #             repeated_tokens_per_expert,
    #             group=self.args.expert_parallel_group,
    #             async_op=True,
    #         )

    #     # Permute locally and without any padding so that tokens for each
    #     # parallel device are stored contiguously.
    #     #
    #     # This view updates the shape of the tensor from [sl, bs, hs] to
    #     # [sl * bs, hs] prior to the permutation.
    #     x = x.view(-1, x.shape[-1])
    #     x = ops.gather(x, indices, bin_ids, bins, self.top_k)

    #     # Compute the number of tokens that will be received from each
    #     # device and permute the input data across the devices.
    #     with torch.no_grad():
    #         tpe_handle.wait()
    #         experts_per_rank = mpu.experts_per_rank(self.args)

    #         # Reshape to [world_size, num_experts_per_rank].
    #         world_size = mpu.get_expert_parallel_world_size(self.args)
    #         repeated_tokens_per_expert = repeated_tokens_per_expert.view(
    #             world_size, experts_per_rank
    #         )
    #         parallel_tokens_per_expert = parallel_tokens_per_expert.view(
    #             world_size, experts_per_rank
    #         )

    #         # TODO(tgale): It might be faster to do this on the GPU and
    #         # then communicate the results back to the host.
    #         send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)
    #         parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
    #         recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

    #         # Convert the send/recv counts to lists.
    #         send_counts = send_counts.tolist()
    #         recv_counts = recv_counts.tolist()
    #         tokens_received = sum(recv_counts)

    #     # If we're sharding the experts along the hidden dimension
    #     # multiple devices own parts of the same sets of experts.
    #     # Replicate the token counts so devices that share experts
    #     # get all of the tokens assigned to them.
    #     #
    #     # TODO(tgale): Fuse this into the prior, local permutation.
    #     x = ops.repeat(x, (mpu.hidden_sharding_degree(self.args), 1))

    #     # Start the cross-device permutation asynchronously so we can
    #     # overlap communication with computation.
    #     parallel_x, parallel_x_handle = all_to_all(
    #         x, recv_counts, send_counts, self.args.expert_parallel_group, async_op=True
    #     )

    #     with torch.no_grad():
    #         # After we do the cross-device permutation we have the tokens on the
    #         # correct device but not yet grouped by expert because we received
    #         # tokens from each device as contiguous chunks. To group the tokens
    #         # for expert computation we'll do one more local permutation. The
    #         # rest of this torch.no_grad() scope sets up the indices and bins
    #         # for this permutation.
    #         replicate_bins = ops.inclusive_cumsum(
    #             parallel_tokens_per_expert.flatten(), 0
    #         )
    #         replicate_bins = (
    #             replicate_bins.view(1)
    #             if not len(replicate_bins.size())
    #             else replicate_bins
    #         )

    #         # Construct the expert indices for the permuted tokens.
    #         parallel_top_expert = torch.remainder(
    #             torch.arange(
    #                 self.num_experts * mpu.hidden_sharding_degree(self.args),
    #                 dtype=torch.int32,
    #                 device=indices.device,
    #             ),
    #             mpu.experts_per_rank(self.args),
    #         )
    #         parallel_top_expert = ops.replicate(
    #             parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
    #         ).flatten()

    #         # TODO(tgale): The sort_end_bit here can be reduced.
    #         parallel_bin_ids, parallel_indices = ops.sort(
    #             parallel_top_expert, self.sort_end_bit
    #         )

    #         # Calculate the bins boundaries from the token counts.
    #         parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
    #             dim=0, dtype=torch.int
    #         )
    #         parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
    #         parallel_bins = (
    #             parallel_bins.view(1)
    #             if not len(parallel_bins.size())
    #             else parallel_bins
    #         )

    #     # Locally permute the tokens and perform the expert computation.
    #     # Block to make sure that the cross-device permutation is complete.
        
    #     # GroupedMLP requires counts on CPU. We can use the tensor already
    #     # moved to CPU for the prior all_to_all, which avoids an extra
    #     # device synchronization.
    #     parallel_tokens_per_expert = parallel_tokens_per_expert_cpu.sum(
    #         dim=0, dtype=torch.int
    #     )
    #     parallel_x_handle.wait()
    #     parallel_x = self.permute_and_compute(
    #         parallel_x,
    #         parallel_tokens_per_expert,
    #         parallel_indices,
    #         parallel_bin_ids,
    #         None,  # expert_weights
    #         parallel_bins,
    #         top_k=1,
    #     )

    #     # Un-permute the tokens across the devices.
    #     x, _ = all_to_all(
    #         parallel_x, send_counts, recv_counts, self.args.expert_parallel_group
    #     )

    #     # Reduce along the hidden sharding to get the final outputs.
    #     #
    #     # TODO(tgale): Fuse this into the following local permutation.
    #     shape = (mpu.hidden_sharding_degree(self.args), -1, self.args.hidden_size)
    #     x = ops.sum(x.view(shape), dim=0)

    #     # Un-permute locally to setup for the next series of operations.
    #     x = ops.scatter(
    #         x,
    #         indices,
    #         bin_ids,
    #         expert_weights,
    #         bins,
    #         self.top_k,
    #         self.args.quantize_scatter_num_bits,
    #     )
    #     return x, tokens_per_expert.flatten()