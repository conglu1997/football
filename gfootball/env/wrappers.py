# coding=utf-8
# Copyright 2019 Google LLC
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


"""Environment that can be used with OpenAI Baselines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import cv2
from functools import partial
from gfootball.env import observation_preprocessing
import gym
import numpy as np


class PeriodicDumpWriter(gym.Wrapper):
  """A wrapper that only dumps traces/videos periodically."""

  def __init__(self, env, dump_frequency):
    gym.Wrapper.__init__(self, env)
    self._dump_frequency = dump_frequency
    self._original_render = env._config['render']
    self._original_dump_config = {
        'write_video': env._config['write_video'],
        'dump_full_episodes': env._config['dump_full_episodes'],
        'dump_scores': env._config['dump_scores'],
    }
    self._current_episode_number = 0

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    if (self._dump_frequency > 0 and
        (self._current_episode_number % self._dump_frequency == 0)):
      self.env._config.update(self._original_dump_config)
      self.env._config.update({'render': True})
    else:
      self.env._config.update({'render': self._original_render,
                               'write_video': False,
                               'dump_full_episodes': False,
                               'dump_scores': False})
    self._current_episode_number += 1
    return self.env.reset()


class Simple115StateWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to 115-features state."""

  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    shape = (self.env.unwrapped._config.number_of_players_agent_controls(), 115)
    self.observation_space = gym.spaces.Box(
        low=-1, high=1, shape=shape, dtype=np.float32)

  def observation(self, observation):
    """Converts an observation into simple115 format.

    Args:
      observation: observation that the environment returns

    Returns:
      (N, 115) shaped representation, where N stands for the number of players
      being controlled.
    """
    final_obs = []
    for obs in observation:
      o = []
      o.extend(obs['left_team'].flatten())
      o.extend(obs['left_team_direction'].flatten())
      o.extend(obs['right_team'].flatten())
      o.extend(obs['right_team_direction'].flatten())

      # If there were less than 11vs11 players we backfill missing values with
      # -1.
      # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
      if len(o) < 88:
        o.extend([-1] * (88 - len(o)))

      # ball position
      o.extend(obs['ball'])
      # ball direction
      o.extend(obs['ball_direction'])
      # one hot encoding of which team owns the ball
      if obs['ball_owned_team'] == -1:
        o.extend([1, 0, 0])
      if obs['ball_owned_team'] == 0:
        o.extend([0, 1, 0])
      if obs['ball_owned_team'] == 1:
        o.extend([0, 0, 1])

      active = [0] * 11
      if obs['active'] != -1:
        active[obs['active']] = 1
      o.extend(active)

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)
      final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)


class PixelsStateWrapper(gym.ObservationWrapper):
  """A wrapper that extracts pixel representation."""

  def __init__(self, env, grayscale=True,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._grayscale = grayscale
    self._channel_dimensions = channel_dimensions
    self.observation_space = gym.spaces.Box(
        low=0, high=255,
        shape=(self.env.unwrapped._config.number_of_players_agent_controls(),
               channel_dimensions[1], channel_dimensions[0],
               1 if grayscale else 3),
        dtype=np.uint8)

  def observation(self, obs):
    o = []
    for observation in obs:
      frame = observation['frame']
      if self._grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      frame = cv2.resize(frame, (self._channel_dimensions[0],
                                 self._channel_dimensions[1]),
                         interpolation=cv2.INTER_AREA)
      if self._grayscale:
        frame = np.expand_dims(frame, -1)
      o.append(frame)
    return np.array(o, dtype=np.uint8)


class SMMWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to a minimap."""

  def __init__(self, env,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._channel_dimensions = channel_dimensions
    shape = (self.env.unwrapped._config.number_of_players_agent_controls(),
             channel_dimensions[1], channel_dimensions[0],
             len(observation_preprocessing.get_smm_layers(
                 self.env.unwrapped._config)))
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=shape, dtype=np.uint8)

  def observation(self, obs):
    return observation_preprocessing.generate_smm(
        obs, channel_dimensions=self._channel_dimensions,
        config=self.env.unwrapped._config)


class SingleAgentObservationWrapper(gym.ObservationWrapper):
  """A wrapper that returns an observation only for the first agent."""

  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        dtype=env.observation_space.dtype)

  def observation(self, obs):
    return obs[0]


class SingleAgentRewardWrapper(gym.RewardWrapper):
  """A wrapper that returns a reward only for the first agent."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    return reward[0]


class CheckpointRewardWrapper(gym.RewardWrapper):
  """A wrapper that adds a dense checkpoint reward."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)
    self._collected_checkpoints = {True: 0, False: 0}
    self._num_checkpoints = 10
    self._checkpoint_reward = 0.1

  def reset(self):
    self._collected_checkpoints = {True: 0, False: 0}
    return self.env.reset()

  def reward(self, reward):
    if self.env.unwrapped.last_observation is None:
      return reward

    assert len(reward) == len(self.env.unwrapped.last_observation)

    for rew_index in range(len(reward)):
      o = self.env.unwrapped.last_observation[rew_index]
      is_left_to_right = o['is_left']

      if reward[rew_index] == 1:
        reward[rew_index] += self._checkpoint_reward * (
            self._num_checkpoints -
            self._collected_checkpoints[is_left_to_right])
        self._collected_checkpoints[is_left_to_right] = self._num_checkpoints
        continue

      # Check if the active player has the ball.
      if ('ball_owned_team' not in o or
          o['ball_owned_team'] != (0 if is_left_to_right else 1) or
          'ball_owned_player' not in o or
          o['ball_owned_player'] != o['active']):
        continue

      if is_left_to_right:
        d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
      else:
        d = ((o['ball'][0] + 1) ** 2 + o['ball'][1] ** 2) ** 0.5

      # Collect the checkpoints.
      # We give reward for distance 1 to 0.2.
      while (self._collected_checkpoints[is_left_to_right] <
             self._num_checkpoints):
        if self._num_checkpoints == 1:
          threshold = 0.99 - 0.8
        else:
          threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
                       self._collected_checkpoints[is_left_to_right])
        if d > threshold:
          break
        reward[rew_index] += self._checkpoint_reward
        self._collected_checkpoints[is_left_to_right] += 1
    return reward


class FrameStack(gym.Wrapper):
  """Stack k last observations."""

  def __init__(self, env, k):
    gym.Wrapper.__init__(self, env)
    self.obs = collections.deque([], maxlen=k)
    low = env.observation_space.low
    high = env.observation_space.high
    low = np.concatenate([low] * k, axis=-1)
    high = np.concatenate([high] * k, axis=-1)
    self.observation_space = gym.spaces.Box(
        low=low, high=high, dtype=env.observation_space.dtype)

  def reset(self):
    observation = self.env.reset()
    self.obs.extend([observation] * self.obs.maxlen)
    return self._get_observation()

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    self.obs.append(observation)
    return self._get_observation(), reward, done, info

  def _get_observation(self):
    return np.concatenate(list(self.obs), axis=-1)

class MAPOSimple115StateWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to 115-features state.

     Each Observation is converted to coordinates relative to the respective player's absolute position (ego-frame)

     In addition, each observation is modified so as to respect partial observability constraints resulting
     from:
      - restricted view cone
      - depth noise
      - view obstruction
  """

  def __init__(self,
               env,
               po_view_cone_xy_opening=160,
               po_view_cone_z_opening=70,
               po_player_width=0.060,
               po_player_height=0.033,
               po_player_view_radius=-1,
               po_depth_noise="default"):
    gym.ObservationWrapper.__init__(self, env)
    self.po_view_cone_xy_opening = po_view_cone_xy_opening
    self.po_view_cone_z_opening = po_view_cone_z_opening
    self.po_player_width = po_player_width
    self.po_player_height = po_player_height
    self.po_player_view_radius = po_player_view_radius
    self.po_depth_noise = {"type":"gaussian", "sigma":0.1, "attenuation_type": "fixed_angular_resolution", "angular_resolution_degrees":0.2}\
        if po_depth_noise == "default" else po_depth_noise
    shape = (self.env.unwrapped._config.number_of_players_agent_controls(), 115)
    self.observation_space = gym.spaces.Box(
        low=-1, high=1, shape=shape, dtype=np.float32)

  def observation(self, observation):
    """Converts an observation into simple115 format.

    Args:
      observation: observation that the environment returns

    Returns:
      (N, 115) shaped representation, where N stands for the number of players
      being controlled.
    """

    # initialise / update player view directions
    player_view_directions = getattr(self, "player_view_directions", None)
    if player_view_directions is None:
        # set player view directions to players facing forward at the beginning
        self.player_view_directions = {}
        self.player_view_directions["right"] = np.zeros((len(observation), 2))
        self.player_view_directions["right"][:, 0] = -1
        self.player_view_directions["left"] = np.zeros((len(observation), 2))
        self.player_view_directions["left"][:, 0] = 1

    for player_id, obs in enumerate(observation):
        if np.sum(obs["left_team_direction"]) != 0.0:
            self.player_view_directions["left"][player_id] = obs["left_team_direction"] / np.linalg.norm(obs["left_team_direction"])
        if np.sum(obs["right_team_direction"]) != 0.0:
            self.player_view_directions["right"][player_id] = obs["left_team_direction"] / np.linalg.norm(obs["left_team_direction"])

    _add_zero = lambda vec: np.concatenate([vec, np.array([0])], -1)  # helper function used to extend 2d to 3d coords

    # encapsulate into objects
    class _gobj():
        def __init__(self, label, type, location, attrs):
            self.label = label
            self.type = type
            self.location = location
            self.distance = np.linalg.norm(location)
            self.attrs = attrs
            self.is_visible = True

        def rep(self):
            lst = []
            lst.extend([1.0 if self.is_visible else 0.0])

            if self.type in ["player"]:
                if self.is_visible:
                    lst.extend(self.location[:2])
                    norm = np.linalg.norm(self.attrs["view_direction"])
                    if norm != 0.0:
                        lst.extend([self.attrs["move_direction"][0]/norm,
                                    self.attrs["move_direction"][1]/norm,
                                    norm])
                    else:
                        lst.extend([0]*2)
                    norm = np.linalg.norm(self.attrs["view_direction"])
                    if norm != 0.0:
                        lst.extend([self.attrs["view_direction"][0]/norm,
                                    self.attrs["view_direction"][1]/norm])
                    else:
                        lst.extend([0,0])
                else:
                    lst.extend([0]*7)
            elif self.type in ["ball"]:
                if self.is_visible:
                    norm = np.linalg.norm(self.attrs["move_direction"])
                    xy_norm = np.linalg.norm(self.attrs["move_direction"][:2])
                    if norm != 0.0 and xy_norm != 0.0:
                        lst.extend([self.attrs["move_direction"][0]/xy_norm,
                                    self.attrs["move_direction"][1]/xy_norm,
                                    self.attrs["move_direction"][2]/norm,
                                    norm])
                    else:
                        lst.extend([0]*4)
                    lst.extend({-1:[1,0,0],0:[0,1,0],1:[0,0,1]}[self.attrs["owned_team"]])
                else:
                    lst.extend([0]*7)

            return lst


    final_obs = []
    for player_id, obs in enumerate(observation):
      player_location = _add_zero(obs['left_team'][player_id])  # [x,y] of active player
      player_view_direction = _add_zero(self.player_view_directions[player_id])
      obj_lst = []

      # encapsulate left players
      left_player_list = [_gobj(label="left_player__{}".format(i),
                           type="player",
                           location=_add_zero(loc)-player_location,
                           attr=dict(view_direction=self.player_view_directions["left"][i],
                                     move_direction=dir)) for i, (loc, dir) in enumerate(zip(obs['left_team'], obs['left_team_direction']))
                           ]
      obj_lst.extend(left_player_list)

      # encapsulate right players
      right_player_list = [_gobj(label="right_player__{}".format(i),
                                   type="player",
                                   location=_add_zero(loc)-player_location,
                                   attr=dict(view_direction=self.player_view_directions["right"][i],
                                             move_direction=dir)) for i, (loc, dir) in enumerate(zip(obs['right_team'], obs['right_team_direction']))
                                   ]
      obj_lst.extend(right_player_list)

      # encapsulate ball
      ball = _gobj(label="ball",
                   type="ball",
                   location=obs["ball"]-player_location,
                   attr=dict(owned_team=obs["ball_owned_team"],
                             move_direction=obs["ball_direction"]))

      # update visibilities wrt player view radius
      if self.player_view_radius != -1:
        for obj in obj_lst:
          obj.is_visible = True if obj.distance <= self.player_view_radius else False

      # update visibilities wrt player view cone
      def _signed_angle(a, b):
        t = np.degrees(np.arctan2(a[0] * b[1] - b[0] * a[1], a[0] * a[1] + b[0] * b[1]))
        return (-(180 + t) if t < 0 else t)

      def _is_visible(player_view_direction, rel_obj_location, view_cone_xy_opening, view_cone_z_opening):
        if np.linalg.norm(rel_obj_location) == 0.0:
            return True

        rel_angle = _signed_angle(rel_obj_location[:2], player_view_direction[:2])
        rel_z_angle = np.arctan(rel_obj_location[2] / np.linalg.norm(rel_obj_location)) * 180.0 / np.pi if sum(rel_obj_location) != 0.0 else 0.0

        # Determine whether obj location is visible wrt view cone
        if view_cone_xy_opening < 360:
          if (rel_angle < -view_cone_xy_opening // 2) or (rel_angle > view_cone_xy_opening // 2):
            return False

        if view_cone_z_opening < 90:
          if rel_z_angle > view_cone_z_opening:
            return False

        return True

      for obj in obj_lst:
        obj.is_visible = _is_visible(player_view_direction, obj.location, self.view_cone_xy_opening, self.view_cone_z_opening)

      # update visibilities wrt occlusion
      obj_dist_sorted = [o for o in obj_lst if o.is_visible and o.type=="player"].sort(key=lambda obj: obj.distance)
      while len(obj_dist_sorted) > 1:
        curr_obj = obj_dist_sorted[0]
        for obj in obj_dist_sorted[1:]:
          # relative location wrt curr_obj
          if np.linalg.norm(curr_obj.location[:2]) == 0.0 or np.linalg.norm(curr_obj.location) == 0.0:
              continue

          blocked_xy_angle = 2*np.degrees(np.arctan(self.player_width/np.linalg.norm(curr_obj.location[:2])))
          blocked_z_angle = np.arctan(self.player_height/np.linalg.norm(curr_obj.location))
          obj.is_visible = not _is_visible(player_view_direction,
                                           obj.location,
                                           blocked_xy_angle,
                                           blocked_z_angle)
          # update
          obj_dist_sorted = [o for o in obj_dist_sorted if o.is_visible]

      # noise visible object coordinates according to their distance
      if self.depth_noise is not None:
        visible_objs = [obj for obj in obj_lst if obj.is_visible]
        for obj in visible_objs:
          if self.depth_noise.get("type", None) == "gaussian":
            if self.depth_noise.get("attenuation_type", None) == "fixed_angular_resolution":
              angular_resolution = self.depth_noise.get("angular_resolution_degrees", 2) * np.pi / 180.0
              sigma = obj.distance * angular_resolution
              # noise relevant quantities
              obj.location += np.random.normal(0, sigma, (3,))
              obj.attrs["move_direction"] += np.random.normal(0, sigma, (3,))
              if obj.type == "player":
                obj.location[2] = 0.0  # it is commonly known that players are confined to xy plane
                obj.attrs["view_direction"] += np.random.normal(0, sigma, (3,))

      # collate observation from object representations
      o = []
      for obj in obj_lst:
        o.extend(obj.rep())

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)
      final_obs.append(o)

    return np.array(final_obs, dtype=np.float32)
