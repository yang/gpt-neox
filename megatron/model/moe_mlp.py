# TODO: add neox and megablocks copyright notices

import torch
from megatron.model.activations import get_activation

from megatron.mpu.layers import _initialize_affine_weight_gpu
from megatron.mpu.initialize import get_model_parallel_world_size
from megatron.mpu.utils import divide

from megatron.neox_arguments.arguments import NeoXArgs

from megablocks import grouped_gemm_util as gg


class ScaleGradient(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        return grad * ctx.scale, None


scale_gradient = ScaleGradient.apply


class ParallelGroupedMLP(torch.nn.Module):
    def __init__(
            self,
            neox_args: NeoXArgs,
            init_method,
            output_layer_init_method,
            stride=1,
            multiple_of=256,
        ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedMLP, self).__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size
        

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        # Otherwise, 4 x hidden size, padded to multiple of 256
        else:
            per_expert_ff_dim = 4 * self.hidden_size
            per_expert_ff_dim = self.multiple_of * ((per_expert_ff_dim + multiple_of - 1) // multiple_of)

        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension
        self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        # input
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w1, init_method, partition_dim=0, stride=stride
        )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w2, output_layer_init_method, partition_dim=0, stride=stride
        )

        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs
        w1 = w1.view(self.experts_per_rank, -1, self.hidden_size)
        w2 = w2.view(self.experts_per_rank, -1, self.hidden_size)

        # Compute the MLP
        x = gg.ops.gmm(x, w1, grouped_gemm_batch_sizes, trans_b=True)
        x = self.activation_func(x)
        return gg.ops.gmm(x, w2, grouped_gemm_batch_sizes)


class ParallelGroupedLLaMAMLP(torch.nn.Module):
    def __init__(
            self,
            neox_args: NeoXArgs,
            init_method,
            output_layer_init_method,
            stride=1,
            multiple_of=256,
        ):
        """
        Copied from SparseMLP
        """
        super(ParallelGroupedMLP, self).__init__()

        self.activation_func = get_activation(neox_args)
        self.activation_type = neox_args.activation

        self.multiple_of = multiple_of

        world_size = get_model_parallel_world_size()
        self.num_experts = neox_args.moe_num_experts
        self.experts_per_rank = divide(self.num_experts, world_size)

        self.hidden_size = neox_args.hidden_size
        

        # Allow custom intermediate size
        if neox_args.intermediate_size is not None:
            per_expert_ff_dim = neox_args.intermediate_size
        # Otherwise, 4 x hidden size, padded to multiple of 256
        else:
            per_expert_ff_dim = int(2 * neox_args.hidden_size * 4 / 3)
            per_expert_ff_dim = self.multiple_of * ((per_expert_ff_dim + multiple_of - 1) // multiple_of)

        self.per_expert_ff_dim = per_expert_ff_dim
        # number of rows per rank is the number of experts * ff dimension per expert
        self.num_rows_per_rank = self.experts_per_rank * per_expert_ff_dim

        # input
        self.w1 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w1, init_method, partition_dim=0, stride=stride
        )

        # gate
        self.w3 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w3, init_method, partition_dim=0, stride=stride
        )

        # output
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self.num_rows_per_rank,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=neox_args.params_dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.w2, output_layer_init_method, partition_dim=0, stride=stride
        )

        # TODO: why do we need this? was in original megablocks code
        self.gradient_scale = None
        if world_size > 1:
            self.gradient_scale = 1 / world_size

    def scale_grad(self, w: torch.Tensor):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor):
        grouped_gemm_batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w3, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w3), self.scale_grad(self.w2))

        w1 = self.w1.view(self.num_experts, -1, self.hidden_size)
        w3 = w3.view(self.num_experts, -1, self.hidden_size)

        w2 = w2.view(self.num_experts, -1, self.hidden_size)
        
        llama_x_w1T = gg.ops.gmm(
            x, # x
            w1, # w1
            grouped_gemm_batch_sizes,
            trans_b=True
        )

        llama_x_w3T = gg.ops.gmm(
            x, # x
            w3, # w3
            grouped_gemm_batch_sizes,
            trans_b=True
        )

        llama_act_x_w1T = self.activation_func(llama_x_w1T)
        
        # self.w2(self.activation_func(w1_out) * w3_out)
        llama_mlp_out = gg.ops.gmm(
            llama_act_x_w1T * llama_x_w3T, # activation results gated (element-wise) with w3
            w2, # w2
            grouped_gemm_batch_sizes, # batch_sizes
        )

        return llama_mlp_out