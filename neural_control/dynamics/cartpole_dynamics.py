from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import casadi as ca
import numpy as np
import torch
import torch.nn as nn

from neural_control.dynamics._typing import (
    ActionTensor,
    DiscreteTimeDynamics,
    ImageTensor,
    SequenceBufferTensor,
    StateTensor,
)


from neural_control.dynamics.learnt_dynamics import (
    LearntDynamics, LearntDynamicsMPC
)

# target state means that theta is zero --> only third position matters
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.81


class CartpoleDynamics(DiscreteTimeDynamics):

    def __init__(
        self,
        modified_params: dict[str, float] | None = None,
        test_time: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        self.batch_size = batch_size
        with open(
            os.path.join(
                Path(__file__).parent.absolute(), "config_cartpole.json"
            ), "r"
        ) as infile:
            self.cfg = json.load(infile)

        self.test_time = test_time
        if modified_params:
            self.cfg.update(modified_params)
        self.cfg["friction"] = .5
        self.cfg["total_mass"] = self.cfg["masspole"] + self.cfg["masscart"]
        self.cfg["polemass_length"] = self.cfg["masspole"] * self.cfg["length"]
        self.timestamp = 0
        # delay of 2
        if self.cfg["delay"] > 0:
            self.action_buffer = np.zeros(
                (batch_size, int(self.cfg["delay"]), 1)
            )
        self.enforce_contact = -1

    def reset_buffer(self) -> None:
        self.action_buffer = np.zeros(
            (self.batch_size, int(self.cfg["delay"]), 1)
        )

    def __call__(
        self,
        state: StateTensor,
        action: ActionTensor,
        dt: float,
    ) -> StateTensor:
        return self.simulate_cartpole(state, action, dt)

    def simulate_cartpole(
        self,
        state: StateTensor,
        action: ActionTensor,
        delta_t: float,
    ) -> StateTensor:
        """
        Compute new state from state and action
        """
        self.timestamp += .05

        # pylint: disable=no-member,unused-argument
        action = action[..., 0] * self.cfg["max_force_mag"] * 0.5

        sin_theta = torch.sin(state[..., 2])
        cos_theta = torch.cos(state[..., 2])

        xdot_update = self._calculate_xdot_update(
            state, action, sin_theta, cos_theta
        )
        thetadot_update = self._calculate_thetadot_update(
            state, action, sin_theta, cos_theta
        )

        new_x = state[..., 0] + state[..., 1] * delta_t
        new_xdot = state[..., 1] + xdot_update * delta_t
        new_costheta, new_sintheta = self._calculate_theta_update(
            state, delta_t, sin_theta, cos_theta
        )
        new_theta = torch.atan2(new_sintheta, new_costheta)
        new_thetadot = state[..., 3] + thetadot_update * delta_t

        next_state = torch.stack(
            [new_x, new_xdot, new_theta, new_thetadot], dim=-1
        )
        return next_state
        # next_state.expand(torch.Size(sample_shape) + state.shape)

    def _calculate_xdot_update(
        self,
        state: StateTensor,
        action: ActionTensor,
        sin_theta: StateTensor,
        cos_theta: StateTensor,
    ) -> StateTensor:
        # pylint: disable=no-member
        x_dot = state[..., 1]
        theta_dot = state[..., 3]
        return (
            -2 * self.cfg["polemass_length"] * (theta_dot**2) * sin_theta +
            3 * self.cfg["masspole"] * gravity * sin_theta * cos_theta +
            4 * action - 4 * self.cfg["friction"] * x_dot
        ) / (
            4 * self.cfg["total_mass"] -
            3 * self.cfg["masspole"] * cos_theta**2
        )

    def _calculate_thetadot_update(
        self,
        state: StateTensor,
        action: ActionTensor,
        sin_theta: StateTensor,
        cos_theta: StateTensor,
    ) -> StateTensor:
        # pylint: disable=no-member
        x_dot = state[..., 1]
        theta_dot = state[..., 3]
        return (
            -3 * self.cfg["polemass_length"] *
            (theta_dot**2) * sin_theta * cos_theta +
            6 * self.cfg["total_mass"] * gravity * sin_theta + 6 *
            (action - self.cfg["friction"] * x_dot) * cos_theta
        ) / (
            4 * self.cfg["length"] * self.cfg["total_mass"] -
            3 * self.cfg["polemass_length"] * cos_theta**2
        )

    @staticmethod
    def _calculate_theta_update(
        state: StateTensor,
        delta_t: float,
        sin_theta: StateTensor,
        cos_theta: StateTensor,
    ) -> tuple[StateTensor, StateTensor]:
        sin_theta_dot = torch.sin(state[..., 3] * delta_t)
        cos_theta_dot = torch.cos(state[..., 3] * delta_t)
        new_sintheta = sin_theta * cos_theta_dot + cos_theta * sin_theta_dot
        new_costheta = cos_theta * cos_theta_dot - sin_theta * sin_theta_dot
        return new_costheta, new_sintheta


class LearntCartpoleDynamics(LearntDynamics, CartpoleDynamics):

    def __init__(
        self,
        modified_params: dict[str, float] | None = None,
        not_trainable: Sequence[str] | str | None = None,
    ) -> None:
        CartpoleDynamics.__init__(self, modified_params=modified_params)
        super(LearntCartpoleDynamics, self).__init__(4, 1)

        dict_pytorch = {}
        for key, val in self.cfg.items():
            requires_grad = True
            # # code to avoid training the parameters
            if not_trainable == "all" or (
                isinstance(not_trainable, Sequence) and key in not_trainable
            ):
                requires_grad = False
            dict_pytorch[key] = torch.nn.Parameter(
                torch.tensor([val]), requires_grad=requires_grad
            )
        self.cfg = torch.nn.ParameterDict(dict_pytorch)

    def simulate(
        self, state: StateTensor, action: ActionTensor, dt: float
    ) -> StateTensor:
        return self.simulate_cartpole(state, action, dt)


class SequenceCartpoleDynamics(LearntDynamicsMPC, CartpoleDynamics):

    def __init__(self, buffer_length: int = 3) -> None:
        CartpoleDynamics.__init__(self)
        super(SequenceCartpoleDynamics,
              self).__init__(5 * buffer_length, 1, out_state_size=4)

    def simulate(
        self, state: StateTensor, action: ActionTensor, dt: float
    ) -> StateTensor:
        return self.simulate_cartpole(state, action, dt)

    def forward(
        self,
        state: StateTensor,
        state_action_buffer: SequenceBufferTensor,
        action: ActionTensor,
        dt: float,
    ) -> StateTensor:
        # run through normal simulator f hat
        new_state = self.simulate(state, action, dt)
        # run through residual network delta
        added_new_state = self.state_transformer(state_action_buffer, action)
        return new_state + added_new_state


class ImageCartpoleDynamics(torch.nn.Module, CartpoleDynamics):

    def __init__(
        self,
        img_width: int,
        img_height: int,
        nr_img: int = 5,
        state_size: int = 4,
        action_dim: int = 1,
    ) -> None:
        CartpoleDynamics.__init__(self)
        super(ImageCartpoleDynamics, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        # conv net
        self.conv1 = nn.Conv2d(nr_img * 2 - 1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10 + 2, 20, 3, padding=1)
        self.conv4 = nn.Conv2d(20, 1, 3, padding=1)

        # residual network
        self.flat_img_size = 10 * (img_width) * (img_height)

        self.linear_act = nn.Linear(action_dim, 32)
        self.act_to_img = nn.Linear(32, img_width * img_height)

        self.linear_state_1 = nn.Linear(self.flat_img_size + 32, 64)
        self.linear_state_2 = nn.Linear(64, state_size, bias=False)

    def conv_head(self, image: torch.Tensor) -> torch.Tensor:
        cat_all = [image]
        for i in range(image.size()[1] - 1):
            cat_all.append(
                torch.unsqueeze(image[:, i + 1] - image[:, i], dim=1)
            )
        sub_images = torch.cat(cat_all, dim=1)
        conv1 = torch.relu(self.conv1(sub_images.float()))
        conv2 = torch.relu(self.conv2(conv1))
        return conv2

    def action_encoding(self, action: torch.Tensor) -> torch.Tensor:
        ff_act = torch.relu(self.linear_act(action))
        return ff_act

    def state_transformer(
        self, image_conv: torch.Tensor, act_enc: torch.Tensor
    ) -> torch.Tensor:
        flattened = image_conv.reshape((-1, self.flat_img_size))
        state_action = torch.cat((flattened, act_enc), dim=1)

        ff_1 = torch.relu(self.linear_state_1(state_action))
        ff_2 = self.linear_state_2(ff_1)
        return ff_2

    def image_prediction(
        self,
        image_conv: torch.Tensor,
        act_enc: torch.Tensor,
        prior_img: torch.Tensor,
    ) -> torch.Tensor:
        act_img = torch.relu(self.act_to_img(act_enc))
        act_img = act_img.reshape((-1, 1, self.img_width, self.img_height))
        # concat channels
        with_prior = torch.cat((image_conv, prior_img, act_img), dim=1)
        # conv
        conv3 = torch.relu(self.conv3(with_prior))
        conv4 = torch.sigmoid(self.conv4(conv3))
        # return the single channel that we have (instead of squeeze)
        return conv4[:, 0]

    def forward(
        self,
        state: StateTensor,
        image: ImageTensor,
        action: ActionTensor,
        dt: float,
    ) -> StateTensor:
        # run through normal simulator f hat
        new_state = self.simulate_cartpole(state, action, dt)
        # encode image and action (common head)
        img_conv = self.conv_head(image)
        act_enc = self.action_encoding(action)
        # run through residual network delta
        added_new_state = self.state_transformer(img_conv, act_enc)
        # # Predict next image
        # prior_img = torch.unsqueeze(image[:, 0], 1).float()
        # next_img = self.image_prediction(img_conv, act_enc, prior_img)
        return new_state + added_new_state  # , next_img


class CartpoleDynamicsMPC(CartpoleDynamics):

    def __init__(
        self, modified_params: dict[str, float] | None = None
    ) -> None:
        CartpoleDynamics.__init__(self, modified_params=modified_params)

    def simulate_cartpole(self, dt: float) -> ca.Function:
        (x, x_dot, theta, theta_dot) = (
            ca.SX.sym("x"), ca.SX.sym("x_dot"), ca.SX.sym("theta"),
            ca.SX.sym("theta_dot")
        )
        action = ca.SX.sym("action")
        x_state = ca.vertcat(x, x_dot, theta, theta_dot)

        # helper variables
        force = self.cfg["max_force_mag"] * action * 0.5
        costheta = ca.cos(theta)
        sintheta = ca.sin(theta)

        # xdot update
        x_acc = (
            -2 * self.cfg["polemass_length"] *
            (theta_dot**2) * sintheta + 3 * self.cfg["masspole"] * gravity *
            sintheta * costheta + 4 * force - 4 * self.cfg["friction"] * x_dot
        ) / (
            4 * self.cfg["total_mass"] - 3 * self.cfg["masspole"] * costheta**2
        )

        # thetadot_update
        thetaacc = (
            -3 * self.cfg["polemass_length"] *
            (theta_dot**2) * sintheta * costheta +
            6 * self.cfg["total_mass"] * gravity * sintheta + 6 *
            (force - self.cfg["friction"] * x_dot) * costheta
        ) / (
            4 * self.cfg["length"] * self.cfg["total_mass"] -
            3 * self.cfg["polemass_length"] * costheta**2
        )

        wind_drag = self.cfg["wind"] * costheta

        x_state_dot = ca.vertcat(x_dot, x_acc, theta_dot, thetaacc + wind_drag)
        X = x_state + dt * x_state_dot

        F = ca.Function('F', [x_state, action], [X], ['x', 'u'], ['ode'])
        return F


if __name__ == "__main__":
    state_test_np = np.array([0.5, 1.3, 0.1, 0.4])
    state_test = torch.unsqueeze(torch.from_numpy(state_test_np), 0).float()
    action_test_np = np.array([0.4])
    action_test = torch.unsqueeze(torch.from_numpy(action_test_np), 0).float()

    normal_dyn = CartpoleDynamics()
    next_state = normal_dyn(state_test, action_test, 0.02)
    print("------------")
    print(next_state[0])

    # test: compare to mpc
    # if test doesnt work, remove clamp!!
    mpc_dyn = CartpoleDynamicsMPC()
    F = mpc_dyn.simulate_cartpole(0.02)
    mpc_state = F(state_test_np, action_test_np)
    print("--------------------")
    print(mpc_state)
