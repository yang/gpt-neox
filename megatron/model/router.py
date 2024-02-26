# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.

# Code adapted from Megablocks
# repo: https://github.com/stanford-futuredata/megablocks

import torch

from megatron.neox_arguments.arguments import NeoXArgs


class LearnedRouter(torch.nn.Module):
    # TODO: how do we ensure that all data parallel copies get the same
    # initializations? Or is this handled by RNG seeding?

    def __init__(self, neox_args: NeoXArgs):
        super().__init__()
        self.jitter_eps = neox_args.moe_jitter_eps
        self.top_k = neox_args.moe_top_k

        # Learned router parameters.
        #
        # NOTE: This weight matrix is not parallelized with expert tensor
        # parallelism. Each device needs the entire router weight matrix
        # so that it can route its batch of data correctly.
        self.layer = torch.nn.Linear(
            neox_args.hidden_size,
            neox_args.moe_num_experts,
            bias=False,
            dtype=neox_args.params_dtype,
            device=torch.cuda.current_device(),
        )
        neox_args.init_method(self.layer.weight)

    def jitter(self, x):
        """
        Apply jittering to the input tensor during training.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Jittered input tensor.
        """
        low = 1.0 - self.args.moe_jitter_eps
        high = 1.0 + self.args.moe_jitter_eps
        noise = torch.rand(x.size(), dtype=x.dtype, device=x.device)
        return low + noise * (high - low)

    def _top_k(self, scores):
        """
        Select the top-k experts based on input scores.

        Args:
            scores (torch.Tensor): Input scores from the router.
                (sl * bs, num_experts)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing expert weightings and indices of selected experts.


        """
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)

    def forward(self, x):
        """
        Forward pass through the Learned Router.

        Args:
            x (torch.Tensor): Input tensor to be routed.
                (sl, bs, hs)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing
                - scores (sl * bs, num_experts): Output scores from the router
                - expert_weights (sl * bs, top_k): Weights assigned to the selected experts
                - expert_indices (sl * bs, top_k): Indices of the selected experts
        """
        if self.training and self.jitter_eps is not None:
            x = x * self.jitter(x)

        # x.view shape: (sl * bs, hs)...every token as a row
        # scores (float) shape: (sl * bs, num_experts)...expert rankings for every token
        scores = self.layer(x.view(-1, x.shape[-1])).softmax(dim=-1)
        
        # expert_weights (float) shape: (sl * bs, top_k)...value(s) from scores corresponding to the top_k experts
        # expert_indices (int) shape: (sl * bs, top_k)...index(indices) from scores corresponding to the top_k experts
        expert_weights, expert_indices = self._top_k(scores)
        # expert_weights probability mass won't add up to 1 because we took
        # the topk scores from the softmax
        # TODO: placeholder for moe_normalize_expert_weights if necessary

        return scores, expert_weights, expert_indices