# TODO: add copyright that includes Megablocks reference

import megablocks.ops
import numpy as np

from megatron.mpu.initialize import get_model_parallel_rank
from megatron.mpu.initialize import get_model_parallel_world_size

from megatron.neox_arguments.arguments import NeoXArgs
from megatron import mpu
from megatron.mpu import copy_to_model_parallel_region
from megatron.mpu import gather_from_model_parallel_region

import torch

from megatron.model.moe_mlp import ParallelGroupedLLaMAMLP, ParallelGroupedMLP

from .router import LearnedRouter

def promote_scalar(x: torch.Tensor):
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


# def batched_load_balancing_loss():
#     # tokens_per_expert[i].shape = (num_experts)
#     # expert_scores[i].shape = (tokens, num_experts)
#     tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
#     num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
#     if args.num_layers_per_virtual_pipeline_stage is not None:
#         num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

#     if len(tokens_per_expert) != num_layers_per_pipeline_stage:
#         raise ValueError(
#             f"Expected {num_layers_per_pipeline_stage} token_per_experts "
#             f"but found {len(tokens_per_expert)}.\nnum_layers = "
#             f"{args.num_layers}\npipeline_model_parallel_size = "
#             f"{args.pipeline_model_parallel_size}\n"
#             "num_layers_per_virtual_pipeline_stage"
#             f" = {args.num_layers_per_virtual_pipeline_stage}"
#         )
#     if len(expert_scores) != num_layers_per_pipeline_stage:
#         raise ValueError(
#             f"Expected {num_layers_per_pipeline_stage} expert_scores "
#             f"but found {len(tokens_per_expert)}.\nnum_layers = "
#             f"{args.num_layers}\npipeline_model_parallel_size = "
#             f"{args.pipeline_model_parallel_size}\n"
#             "num_layers_per_virtual_pipeline_stage"
#             f" = {args.num_layers_per_virtual_pipeline_stage}"
#         )

#     # Verify the shape of the tokens_per_expert and expert_scores tensors.
#     assert all(
#         [x.ndim == 1 and x.numel() == args.moe_num_experts for x in tokens_per_expert]
#     )

#     tokens = expert_scores[0].shape[0]
#     assert all(
#         [
#             (
#                 x.ndim == 2
#                 and x.shape[1] == args.moe_num_experts
#                 and x.shape[0] == tokens
#             )
#             for x in expert_scores
#         ]
#     )

#     # Concatenate the contributions of each layer and convert to
#     # the correct types and formats for the dot product.
#     if args.moe_lbl_in_fp32:
#         expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
#     else:
#         expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)
#     tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

#     expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
#     assert tokens_per_expert.numel() == expected_values
#     assert expert_scores.numel() == expected_values

#     # Calculate the total scale across all factors.
#     #
#     # loss_weight * num_experts / (num_layers * tokens * top_k)
#     scale_numerator = args.moe_num_experts * args.moe_loss_weight
#     scale_denominator = args.num_layers * tokens * args.moe_top_k
#     scale = scale_numerator / scale_denominator
#     return scale * torch.dot(tokens_per_expert, expert_scores)

def get_expert_tokens_for_rank(routed_tokens: torch.Tensor, tokens_per_expert: torch.Tensor):
    # Calculate cumulative sums of tokens_per_expert, ensure the shapes are correct
    world_size = get_model_parallel_world_size()
    rank = get_model_parallel_rank()

    # TODO: is this check necessary here/what does it cost us to redundantly do it in multiple places?
    assert tokens_per_expert.shape[0] % world_size == 0
    
    cumulative_sums = torch.cumsum(tokens_per_expert, dim=0)
    assert cumulative_sums[-1] == routed_tokens.shape[0]

    # select the right starting and ending indices from the cumsum to figure out what tokens to select
    rank_expert_indices = cumulative_sums.chunk(world_size)
    start_index = rank_expert_indices[rank - 1][-1] if rank > 0 else 0
    end_index = rank_expert_indices[rank][-1]

    # Use indices to select the chunk of the tokens matrix
    selected_experts = routed_tokens[start_index:end_index]

    return selected_experts

def get_expert_token_counts_for_rank(tokens_per_expert: torch.Tensor):
    # TODO: add bounds checking of size is 1D for tokens_per_expert
    # should be (num_experts) long
    world_size = get_model_parallel_world_size()
    rank = get_model_parallel_rank()

    return tokens_per_expert.chunk(world_size)[rank]


class ParallelDroplessMLP(torch.nn.Module):
    """
    This class defines MoE expert computation, using tensor (model) parallel size as the expert parallel size

    The implication of this parallelism decision is that the expert weights can only be sharded within a single node
    """
    def __init__(
        self,
        neox_args: NeoXArgs,
        init_method,
        output_layer_init_method,
    ):
        """
        
        Bias is currently not supported
        """
        super(ParallelDroplessMLP, self).__init__()

        # Calculate the number of experts to allocate on this rank
        world_size = mpu.get_model_parallel_world_size()
        assert neox_args.moe_num_experts % world_size == 0
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = self.num_experts // world_size
        self.top_k = neox_args.moe_top_k

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # decide which parallel grouped MLP implementation to use
        if neox_args.mlp_type == "regular":
            self.mlp = ParallelGroupedMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        elif neox_args.mlp_type == "llama":
            self.mlp = ParallelGroupedLLaMAMLP(
                neox_args=neox_args,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
            )
        else:
            raise KeyError(neox_args.mlp_type)

    def load_balancing_loss(self, tokens_per_expert: torch.Tensor, expert_scores: torch.Tensor):
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

    def indices_and_bins(self, top_expert: torch.Tensor):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = megablocks.ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = megablocks.ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = megablocks.ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
        self,
        input_: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        """
        grouped_permute_and_compute

        torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)
        
        NOTE: Megablocks sets up all MLP tensors as column parallel and uses transposes on some of the grouped_gemm calls for the ops that would be row parallel. This seems to be fine and since we aren't using the underlying NeoX ColumnParallelLinear and RowParallelLinear classes, there doesn't seem to be a reason to change it...because that'd introduce a lot of additional complexity.

        column parallel linear forward

        ```python
        def forward(self, input_):
            if self.use_mup and self.mup_rescale_parameters:
                input_ /= self.width_mult()
            # Set up backprop all-reduce.
            input_parallel = copy_to_model_parallel_region(input_)
            # Matrix multiply.

            bias = self.bias if not self.skip_bias_add else None
            output_parallel = F.linear(input_parallel, self.weight, bias)
            if self.gather_output:
                # All-gather across the partitions.
                output = gather_from_model_parallel_region(output_parallel)
            else:
                output = output_parallel
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias
        ```
        """
        # Route the tokens for MoE computation.
        ## stack (sl, bs, hs) into (sl * bs, hs)
        input_ = input_.view(-1, input_.shape[-1])

        ## repeat each token top_k times and shuffle tokens to group them by their respective experts
        input_ = megablocks.ops.gather(input_, indices, bin_ids, bins, top_k)

        # QUESTION: is this sufficient for backprop/how does input_parallel actually get used?
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)

        # get tokens routed to this rank's experts only
        input_parallel = get_expert_tokens_for_rank(input_parallel, tokens_per_expert)
        # get tokens_per_expert for this rank's experts only
        local_tokens_per_expert = get_expert_token_counts_for_rank(tokens_per_expert)

        # Perform the expert computation for this rank's experts
        output_parallel = self.mlp(input_parallel, local_tokens_per_expert)

        # all gather masked results from across Tensor parallel ranks here and cat them together
        # this will replicate the calculation of each expert across all ranks
        # NOTE: this combined all_gather and torch.cat operation is performed by gather_from_model_parallel_region(output_parallel)
        # Unlike ColumnParallelLinear, it is nonsensical in the MoE world
        # to optionally return the output_parallel result...we still have to scatter the tokens back to their original positions
        output = gather_from_model_parallel_region(output_parallel)

        # Un-route the data for the MoE output
        return megablocks.ops.scatter(
            output,
            indices,
            bin_ids,
            expert_weights,
            bins,
            top_k,
        )
        

    def forward_once(self, x: torch.Tensor, expert_weights: torch.Tensor, expert_indices: torch.Tensor):
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
            self.top_k,
        )
        return out, tokens_per_expert

    def forward(self, x, scores, expert_weights, expert_indices):
        # save shape so we can re-shape the outputs later
        in_shape = x.size()

        # Compute the experts.
        x, tokens_per_expert = self.forward_once(x, expert_weights, expert_indices)
        
        # save load balancing loss
        # TODO: this is based on megatron-only...how does this work in the megatron-deepspeed world?
        if self.training:
            save_load_balancing_loss((tokens_per_expert, scores))
        
        # restore input shape
        x = x.view(in_shape)
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

class ParallelDroplessMoE(torch.nn.Module):
    """
    TODO: add check to ensure world_size >= num experts...otherwise you end up cutting experts across ranks. which is bad
    """
    def __init__(
            self,
            neox_args: NeoXArgs,
            init_method,
            output_layer_init_method,
        ):
        super(ParallelDroplessMoE, self).__init__()

        self.router = LearnedRouter(
            neox_args,
            init_method
        )

        self.experts = ParallelDroplessMLP(
            neox_args,
            init_method,
            output_layer_init_method,
        )

    def forward(self, x):
        # we expect inputs as (sl, bs, hs)
        # neox also provides inputs as (sl, bs, hs)

        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth
        x = cast_if_autocast_enabled(x)

        # Compute the expert scores and assignments
        scores, expert_weights, expert_indices = self.router(x)

        # return value should be
        return self.experts(x, scores, expert_weights, expert_indices), None
