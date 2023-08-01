import os
from typing import TYPE_CHECKING, Union

import habitat
import matplotlib.pyplot as plt
import numpy as np
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.agent import Agent

# from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

# from habitat.utils.visualizations import maps
# from habitat.utils.visualizations.utils import (
#     images_to_video,
#     observations_to_image
# )
# from habitat_sim.utils import viz_utils as vut
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    ContinuousNavigationAction,
    DiscreteNavigationAction,
    Observations,
)

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self):
        self.env = None
        self.shortest_path_follower = None
        self.goal_coordinates = None
        self.discrete_action_map = {
            0: DiscreteNavigationAction.STOP,
            1: DiscreteNavigationAction.MOVE_FORWARD,
            2: DiscreteNavigationAction.TURN_LEFT,
            3: DiscreteNavigationAction.TURN_RIGHT,
        }

    def set_oracle_info(self, env, goal_type: str = "NAV_TO_OBJ"):
        """Instantiate shortest path follower

        Args:
            env (_type_): Habitat env
            goal_type (str, optional): One of NAV_TO_OBJ pr NAV_TO_REC. Defaults to "NAV_TO_OBJ".
        """
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=env.habitat_env.sim,
            goal_radius=0.05,
            return_one_hot=False,
        )
        if goal_type == "NAV_TO_REC":
            self.goal_coordinates = (
                env.habitat_env.current_episode.candidate_goal_receps[0].position
            )
            print(f"Goal co-ordinates for NAV_TO_REC set to {self.goal_coordinates}")
        else:  # default to NAV_TO_OBJ
            self.goal_coordinates = env.habitat_env.current_episode.candidate_objects[
                0
            ].position
            print(f"Goal co-ordinates for NAV_TO_OBJ set to {self.goal_coordinates}")

    def act(self, observations, info) -> Union[int, np.ndarray]:
        # import time
        # print(f"Observations: {observations}")
        # time.sleep(3)
        # print(f"Info: {info}")
        # time.sleep(3)
        # print(f"Env: {self.env}")
        # time.sleep(3)
        # print(f"Follower: {self.shortest_path_follower}")
        # time.sleep(3)
        # print(f"Episode: {self.env.habitat_env.current_episode}")
        # time.sleep(3)

        action = self.discrete_action_map[
            self.shortest_path_follower.get_next_action(self.goal_coordinates)
        ]
        terminate = False
        if action == DiscreteNavigationAction.STOP:
            terminate = True

        return action, terminate

    def reset(self) -> None:
        self.env = None
        self.shortest_path_follower = None
        self.goal_coordinates = None

    def reset_vectorized(self) -> None:
        self.reset()  # or NotImplementedError, really.
