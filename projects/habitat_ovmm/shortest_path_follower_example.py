#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil

import magnum as mn
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from utils.config_utils import (
    create_agent_config,
    create_env_config,
    get_habitat_config,
    get_omega_config,
)
from utils.env_utils import create_ovmm_env_fn

from home_robot.core.interfaces import DiscreteNavigationAction
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)

cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


# class SimpleRLEnv(habitat.RLEnv):
#     def get_reward_range(self):
#         return [-1, 1]

#     def get_reward(self, observations):
#         return 0

#     def get_done(self, observations):
#         return self.habitat_env.episode_over

#     def get_info(self, observations):
#         return self.habitat_env.get_metrics()


os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

DISCRETE_ACTION_MAP = {
    0: DiscreteNavigationAction.STOP,
    1: DiscreteNavigationAction.MOVE_FORWARD,
    2: DiscreteNavigationAction.TURN_LEFT,
    3: DiscreteNavigationAction.TURN_RIGHT,
}


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(info["top_down_map"], output_size)


def shortest_path_example(config):
    # config = habitat.get_config(
    #     config_path="benchmark/ovmm/ovmm.yaml",
    #     overrides=[
    #         "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
    #     ],
    # )

    ovmm_env = create_ovmm_env_fn(config)
    # env = ovmm_env.habitat_env.env._env
    print(f"Total number of episodes in env: {ovmm_env.number_of_episodes}")

    # goal_radius = env.episodes[0].goals[0].radius
    goal_radius = 0.05
    if goal_radius is None:
        goal_radius = config.habitat.simulator.forward_step_size

    print(f"Current env: {ovmm_env}")  # HabitatOpenVocabManipEnv
    print(
        f"Current env: {ovmm_env.habitat_env}"
    )  # <GymHabitatEnv<HabGymWrapper instance>>
    print(f"Current env: {ovmm_env.habitat_env.env}")  # HabGymWrapper instance>
    print(f"Current env: {ovmm_env.habitat_env.env._env}")  # <RLTaskEnv instance>
    print(f"Current env: {ovmm_env.habitat_env.env._env._env}")  # habitat.core.env.Env
    print(f"Current simulator: {ovmm_env.habitat_env.env._env._env.sim}")  # OVMMSim
    follower = ShortestPathFollower(
        ovmm_env.habitat_env.env._env.habitat_env.sim, goal_radius, False
    )

    for _ in range(ovmm_env.number_of_episodes):
        ovmm_env.reset()
        episode_id = ovmm_env.get_current_episode().episode_id
        dirname = os.path.join(
            IMAGE_DIR,
            "shortest_path_example_ovmm",
            f"{episode_id}",
        )
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(
            os.path.join(os.getcwd(), dirname, f"logs_ep_{episode_id}.txt"), "w"
        ) as f:
            f.write("Environment creation successful\n")
            f.write("Agent stepping around inside environment.\n")
            images_third_person = []
            images_instance_map = []
            images_top_down_map = []
            steps, max_steps = 0, 1000
            info = None
            object_pos = ovmm_env.habitat_env.env._env.habitat_env.current_episode.candidate_goal_receps[
                0
            ].position
            goal = (
                ovmm_env.habitat_env.env._env.habitat_env.current_episode.start_position
            )
            goal[0] = (
                ovmm_env.habitat_env.env._env.habitat_env.current_episode.start_position[
                    0
                ]
                + 1
            )
            while (
                not ovmm_env.habitat_env.env._env.habitat_env.episode_over
                and steps < max_steps
            ):
                if steps != 0:
                    #     print(f"OVMM observations GPS: {observations.gps} Compass: {observations.compass}")
                    f.write(
                        f"info['ovmm_dist_to_pick_goal']:\t{info['ovmm_dist_to_pick_goal']}\n"
                    )
                #     print(f"OVMM distance to place goal: {info['ovmm_dist_to_place_goal']}")

                f.write(f"\nTimestep: {steps}\n")
                print(f"Timestep: {steps}")
                best_action = DISCRETE_ACTION_MAP[
                    follower.get_next_action(mn.Vector3(goal), f)
                ]
                f.write(f"Agent action: {best_action}\n")
                if best_action is None:
                    break

                observations, done, info = ovmm_env.apply_action(best_action, info)
                steps += 1
                info["timestep"] = steps
                third_person_image = observations.third_person_image
                instance_map = observations.task_observations["instance_map"]

                # top_down_map = draw_top_down_map(info, im1.shape[0])
                # output_im = np.concatenate((im1, im2, info["top_down_map.map"]), axis=1)
                images_third_person.append(third_person_image)
                images_instance_map.append(instance_map)
                # images_top_down_map.append(info["top_down_map.map"])
            images_to_video(images_third_person, dirname, "trajectory_third_person")
            images_to_video(images_instance_map, dirname, "trajectory_instance_map")
            # images_to_video(images_top_down_map, dirname, "trajectory_top_down_map")
            if steps >= max_steps:
                f.write("Max steps reached! Aborting episode...")
            else:
                f.write("Episode finished succesfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=["local", "local_vectorized", "remote"],
        default="local",
    )
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/agent/oracle_agent.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--env_config_path",
        type=str,
        default="projects/habitat_ovmm/configs/env/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "random"],
        help="Agent to evaluate",
    )
    parser.add_argument(
        "overrides",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # get habitat config
    habitat_config, _ = get_habitat_config(
        args.habitat_config_path,
        overrides=args.overrides
        + [
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
        ],
    )

    # get baseline config
    baseline_config = get_omega_config(args.baseline_config_path)

    # get env config
    env_config = get_omega_config(args.env_config_path)

    # merge habitat and env config to create env config
    env_config = create_env_config(
        habitat_config, env_config, evaluation_type=args.evaluation_type
    )

    # merge env config and baseline config to create agent config
    agent_config = create_agent_config(env_config, baseline_config)

    shortest_path_example(env_config)