# Copyright (c) 2024, EleutherAI
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

import pytest
import torch

from megatron.model.router import LearnedRouter
from copy import deepcopy
from unittest.mock import patch
from megatron.neox_arguments import NeoXArgs
from tests.common import simulate_deepy_env, BASE_CONFIG

def get_args_moe_k1():
    neox_args = NeoXArgs.from_dict(BASE_CONFIG)
    neox_args.moe_num_experts=8
    neox_args.moe_jitter_eps=0.8
    neox_args.moe_top_k=1
    return neox_args

def test_output_shapes():
    neox_args = get_args_moe_k1()
    router = LearnedRouter(neox_args)

    input_tensor = torch.randn((neox_args.train_micro_batch_size_per_gpu, neox_args.seq_len, neox_args.hidden_size))
    output_scores, expert_weights, expert_indices = router(input_tensor)

    # Assert output shapes
    pytest.assert_equal(output_scores.shape, (neox_args.train_micro_batch_size_per_gpu * neox_args.seq_len, neox_args.moe_num_experts))
    pytest.assert_equal(expert_weights.shape, (neox_args.train_micro_batch_size_per_gpu * neox_args.seq_len, neox_args.moe_top_k))
    pytest.assert_equal(expert_indices.shape, (neox_args.train_micro_batch_size_per_gpu * neox_args.seq_len, neox_args.moe_top_k))

def test_jitter_function(self):
    router = LearnedRouter(self.neox_args)

    input_tensor = torch.randn((self.neox_args.batch_size, self.neox_args.seq_len, self.neox_args.hidden_size))

    # Set training mode and apply jitter
    router.train()
    input_tensor_jittered = router.jitter(input_tensor)

    # Assert the shape remains the same
    self.assertEqual(input_tensor.shape, input_tensor_jittered.shape)

def test_top_k_function(self):
    router = LearnedRouter(self.neox_args)

    num_experts = self.neox_args.moe_num_experts
    scores = torch.randn((self.batch_size * self.sequence_length, num_experts))

    expert_weights, expert_indices = router._top_k(scores)

    # Assert top_k output shapes
    self.assertEqual(expert_weights.shape, (self.batch_size * self.sequence_length, router.top_k))
    self.assertEqual(expert_indices.shape, (self.batch_size * self.sequence_length, router.top_k))
