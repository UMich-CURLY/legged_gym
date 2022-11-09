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
from datetime import datetime



def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False

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
    # get the date and time as a string
    now = datetime.now()

    # create a folder for the log file if it does not exist with the date and time
    log_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', now.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_path, exist_ok=True)

    # create a log file using logging
    logging.basicConfig(filename=os.path.join(log_path, 'robot' + str(robot_index) + '.log'), level=logging.DEBUG,format='%(message)s')

    print('Feet order: FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
    time = 0

    for i in range(1000*int(env.max_episode_length)):


        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        time += env.dt

        # if i < stop_state_log:
        logger.log_states(
            {
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
                'contact_force_x_FL': env.contact_forces[robot_index, env.feet_indices[0], 0].cpu().numpy(),
                'contact_force_y_FL': env.contact_forces[robot_index, env.feet_indices[0], 1].cpu().numpy(),
                'contact_force_z_FL': env.contact_forces[robot_index, env.feet_indices[0], 2].cpu().numpy(),
                'contact_force_x_FR': env.contact_forces[robot_index, env.feet_indices[1], 0].cpu().numpy(),
                'contact_force_y_FR': env.contact_forces[robot_index, env.feet_indices[1], 1].cpu().numpy(),
                'contact_force_z_FR': env.contact_forces[robot_index, env.feet_indices[1], 2].cpu().numpy(),
                'contact_force_x_RL': env.contact_forces[robot_index, env.feet_indices[2], 0].cpu().numpy(),
                'contact_force_y_RL': env.contact_forces[robot_index, env.feet_indices[2], 1].cpu().numpy(),
                'contact_force_z_RL': env.contact_forces[robot_index, env.feet_indices[2], 2].cpu().numpy(),
                'contact_force_x_RR': env.contact_forces[robot_index, env.feet_indices[3], 0].cpu().numpy(),
                'contact_force_y_RR': env.contact_forces[robot_index, env.feet_indices[3], 1].cpu().numpy(),
                'contact_force_z_RR': env.contact_forces[robot_index, env.feet_indices[3], 2].cpu().numpy(),
                "contact_forces_world_x_FL": env.sensor_forces[robot_index, 0,0].cpu().numpy(),
                "contact_forces_world_y_FL": env.sensor_forces[robot_index, 0,1].cpu().numpy(),
                "contact_forces_world_z_FL": env.sensor_forces[robot_index, 0,2].cpu().numpy(),
                "contact_forces_world_x_FR": env.sensor_forces[robot_index, 1,0].cpu().numpy(),
                "contact_forces_world_y_FR": env.sensor_forces[robot_index, 1,1].cpu().numpy(),
                "contact_forces_world_z_FR": env.sensor_forces[robot_index, 1,2].cpu().numpy(),
                "contact_forces_world_x_RL": env.sensor_forces[robot_index, 2,0].cpu().numpy(),
                "contact_forces_world_y_RL": env.sensor_forces[robot_index, 2,1].cpu().numpy(),
                "contact_forces_world_z_RL": env.sensor_forces[robot_index, 2,2].cpu().numpy(),
                "contact_forces_world_x_RR": env.sensor_forces[robot_index, 3,0].cpu().numpy(),
                "contact_forces_world_y_RR": env.sensor_forces[robot_index, 3,1].cpu().numpy(),
                "contact_forces_world_z_RR": env.sensor_forces[robot_index, 3,2].cpu().numpy(),
                'contact_FL': env.contact[robot_index][0].cpu().numpy().item()*1,
                'contact_FR': env.contact[robot_index][1].cpu().numpy().item()*1,
                'contact_RL': env.contact[robot_index][2].cpu().numpy().item()*1,
                'contact_RR': env.contact[robot_index][3].cpu().numpy().item()*1,
            }
        )
        # print(env.contact_forces[robot_index, env.feet_indices[0], 0].cpu().numpy())
        # print(env.contact_forces[robot_index, env.feet_indices[1], 0].cpu().numpy())
        # print(env.contact_forces[robot_index, env.feet_indices[2], 0].cpu().numpy())
        # print(env.contact_forces[robot_index, env.feet_indices[3], 0].cpu().numpy())


        # if this is the first iteration of the episode, create a .log file using logging module
        if i == 20:
            # create a string of all the keys in logger.state_log.items()
            keys = 'time '
            for key in logger.state_log.items(): # with good formatting separating each key with a ||
                keys += str(key[0]) + ' '
            logging.info(keys)
            # create a dashed line to separate the keys from the values where the || are aligned
            # create a string of dashes with the same length as the number of characters in 'time'
            dashes = '----'
            for key in logger.state_log.items():
                # create a string of dashes with the same length as the key
                dashes += '-' * len(str(key[0])) + ' '
            logging.info(dashes)

        elif i>20:
            # sum the time from env.dt and the previous time
            
            # create a string of all the values in logger.state_log.items() with a precision of 3
            values = str(round(time, 3)) + ' '
            # get the last value in the list of values for each key with a precision of 3 
            for key in logger.state_log.items():
                values += str(np.round(key[1][-1], 3)) + ' '
            logging.info(values)



if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
    # close the log file
    logging.shutdown()

