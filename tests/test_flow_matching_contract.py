from __future__ import annotations

import torch

from models.modeling_smolvlm_vla import (
    _euler_integrate_flow,
    _flow_interpolate,
    _flow_reconstruct_action,
    _flow_target_velocity,
)


def test_oracle_constant_velocity_recovers_action_for_any_euler_step_count():
    torch.manual_seed(7)
    action = torch.randn(4, 10, 6)
    noise = torch.randn_like(action)
    oracle_velocity = _flow_target_velocity(action, noise)

    for steps in (1, 5, 20):
        recovered = _euler_integrate_flow(
            noise.clone(),
            steps,
            lambda x_t, t: oracle_velocity,
        )
        torch.testing.assert_close(recovered, action, atol=2e-6, rtol=2e-6)


def test_one_point_oracle_reconstruction_recovers_action_at_all_times():
    torch.manual_seed(11)
    action = torch.randn(7, 10, 6)
    noise = torch.randn_like(action)
    velocity = _flow_target_velocity(action, noise)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    x_t = _flow_interpolate(action, noise, t)
    recovered = _flow_reconstruct_action(x_t, velocity, t)
    torch.testing.assert_close(recovered, action, atol=1e-6, rtol=1e-6)
