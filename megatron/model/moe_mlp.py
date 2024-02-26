from megablocks.layers import common
from megablocks.layers import gelu
from megablocks.layers.activation_fn import act_fn
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments, InitFn, DEFAULT_ACTIVATION_FN
from megablocks import turbo_util as turbo
from megablocks import grouped_gemm_util as gg
import torch
import torch.nn.functional as F
from packaging import version

from megatron.mpu.utils import divide
from megatron.mpu.initialize import get_model_parallel_rank
from megatron.mpu.initialize import get_model_parallel_world_size

from megatron.neox_arguments.arguments import NeoXArgs


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


class GroupedMLP(torch.nn.Module):
    def __init__(self, neox_args: NeoXArgs):
        """
        Copied from SparseMLP
        """
        super().__init__()
        self._experts_per_rank = divide(neox_args.moe_num_experts, get_model_parallel_world_size())
        self._num_rows_per_rank = self._experts_per_rank * args.ffn_hidden_size

        self.w1 = torch.nn.Parameter(
            torch.empty(
                self._num_rows_per_rank,
                args.hidden_size,
                device=args.device,
                dtype=neox_args.params_dtype,
            )
        )
        self.w2 = torch.nn.Parameter(
            torch.empty(
                self._num_rows_per_rank,
                args.hidden_size,
                device=args.device,
                dtype=common.dtype(args),
            )
        )

        # Initialize the parameters for the MLP.
        #
        # NOTE: It is important that we create the weight tensors prior
        # to creating the master weights and slicing our the piece for
        # this rank. If the master weights are created first the PyTorch
        # caching allocator appears to use the same memory block for these
        # and the slice which causes large increases in our peak memory
        # usage.
        with torch.no_grad():
            self.w1.copy_(
                create_dmoe_expert_weights(
                    args,
                    args.moe_num_experts,
                    args.ffn_hidden_size,
                    args.hidden_size,
                    args.init_method,
                )
            )
            self.w2.copy_(
                create_dmoe_expert_weights(
                    args,
                    args.moe_num_experts,
                    args.ffn_hidden_size,
                    args.hidden_size,
                    args.output_layer_init_method,
                )
            )

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

    def scale_grad(self, w):
        """
        Copied from SparseMLP
        """
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, x, tokens_per_expert):
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1, w2 = (self.scale_grad(self.w1), self.scale_grad(self.w2))

        # Re-shape the weights for the grouped GEMMs.
        ne = mpu.experts_per_rank(self.args)
        w1 = w1.view(ne, -1, self.args.hidden_size)
        w2 = w2.view(ne, -1, self.args.hidden_size)

        # Compute the MLP.
        x = gg.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = self.args.activation_fn(x)
        return gg.ops.gmm(x, w2, batch_sizes)
