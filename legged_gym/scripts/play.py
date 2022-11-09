# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
# import os
import matplotlib as plt
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import pandas as pd

import numpy as np
import torch
import logging



def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.sim_device = torch.device("cuda")
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # create a
    for i in range(1000*int(env.max_episode_length)):

        # print the contact forces for all robots and all feet
        # print('contact_forces on each leg for each robot', env.contact_forces[:, env.feet_indices].cpu().numpy()) # works!
  
        # get the height of the terrain below each feet of each robot using the _get_height function
        # print('height field around each robot', env._get_heights().cpu().numpy()) # works!

        # joint positions for each leg for each environment (50,12) (joint angle in rad)
        # print('Joint positions in rad for each leg for each robot', env.dof_pos.cpu().numpy()) # works!
        
        # joint velocities for each leg for each environment (50,12) (joint velocity in rad/s)
        # print('Joint velocities in rad/s for each leg for each robot', env.dof_vel.cpu().numpy()) # works!

        # joint torques for each leg for each environment (50,12) (joint torque in Nm)
        # print('Joint torques in Nm for each leg for each robot', env.torques.cpu().numpy()) # works!
        
        # get the linear velocity of each robot (50,3) (velocity in m/s)
        # print(env.base_lin_vel.cpu().numpy()) # works!

        # get the linear acceleration of each robot (50,3) (acceleration in m/s^2)
        # print(env.base_acc.cpu().numpy()) # works!

        # get the angular velocity of each robot (50,3) (angular velocity in rad/s)
        # print(env.base_ang_vel.cpu().numpy()) # works!

        # get the position of each robot (50,3) (position in m)
        # print(env.base_pos.cpu().numpy()) # works!

        # print orientation of each robot (50,4) (quaternion)
        # print(env.base_quat.cpu().numpy()) # works!

        # print the velocity and acceleration of each rigod body
        # print('velocity of each rigid body', env.rigid_body_lin_vel.cpu().numpy()) # not working
        # print('acceleration of each rigid body', env.rigid_body_ang_vel.cpu().numpy()) # not working

        # for each booleam value in env.contact, if the value is true, create a vector of the corresponding feet incdices that are true

        # get the friction coefficient of the terrain
        # print(env.terrain.friction_coeff)
        # print("Running episode: ", i)
        # for j in range(env.num_envs):
            # for each contact in the environment, get the indices of the feet where the corresponding contact is true
            # feet_in_contact = env.feet_indices.cpu().numpy()[env.contact[j].nonzero().cpu().numpy()]
            # print("friction of env", i, ':', env.friction_coeffs[i])
            # contact = env.get_contact_rigid_body(env.envs[i])
            # print(contact)
            # print('velocity of each feet in contact', env.rigid_body_lin_vel[i, feet_in_contact,:].cpu().numpy())
            # print('base vel',env.base_lin_vel[i,:].cpu().numpy())
            # print('acceleration of each feet in contact', env.rigid_body_ang_vel[i, feet_in_contact,:].cpu().numpy())


        # print('velocity of each feet', env.rigid_body_lin_vel[:, env.feet_indices,:].cpu().numpy()) # not working
        # print('acceleration of each feet', env.rigid_body_ang_vel[:, env.feet_indices,:].cpu().numpy()) # not working

        # Put the position, velocity, 
        # print("Static friction of env:", env.friction_coeffs)
        # print("Ground plane friction:", env.plane_params.dynamic_friction, env.plane_params.static_friction)

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # if RECORD_FRAMES:
        #     if i % 2:
        #         filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
        #         env.gym.write_viewer_image_to_file(env.viewer, filename)
        #         img_idx += 1
        # if MOVE_CAMERA:
        #     camera_position += camera_vel * env.dt
        #     env.set_camera(camera_position, camera_position + camera_direction)

        # if i < stop_state_log:
        # print("contact_forces_world", env.contact_forces_world[robot_index, :,:].cpu().numpy())

        # print("contact_forces", env.sensor_forces[robot_index, :,:].cpu().numpy())

        logger.log_states(
            {
                # 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                # 'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                # 'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                # 'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'base_acc_x': env.base_acc[robot_index, 0].item(),
                'base_acc_y': env.base_acc[robot_index, 1].item(),
                'base_acc_z': env.base_acc[robot_index, 2].item(),
                'base_ang_vel_x': env.base_ang_vel[robot_index, 0].item(),
                'base_ang_vel_y': env.base_ang_vel[robot_index, 1].item(),
                'base_ang_vel_z': env.base_ang_vel[robot_index, 2].item(),
                'base_pos_x': env.base_pos[robot_index, 0].item(),
                'base_pos_y': env.base_pos[robot_index, 1].item(),
                'base_pos_z': env.base_pos[robot_index, 2].item(),
                'base_quat_x': env.base_quat[robot_index, 0].item(),
                'base_quat_y': env.base_quat[robot_index, 1].item(),
                'base_quat_z': env.base_quat[robot_index, 2].item(),
                'base_quat_w': env.base_quat[robot_index, 3].item(),
                'robot_static_friction': env.friction_coeffs[robot_index].item(),
                'ground_plane_dynamic_friction': env.plane_params.dynamic_friction,
                'ground_plane_static_friction': env.plane_params.static_friction,

                # 'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
            }
        )
        # for sensors in env.sensors_list:
        #     for sensor in sensors:
        #         sensor_data = sensor.get_forces()
        #         print("Force of leg: ",sensor_data.force)   # force as Vec3
        #         print("Torque of leg: ",sensor_data.torque)  # torque as Vec3

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
