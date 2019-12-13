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

def _apply_partial_observability(value,
                                 value_pos,
                                 player_pos,
                                 player_view_direction,
                                 depth_noise,
                                 view_obstruction,
                                 view_cone_xy_opening,
                                 view_cone_z_opening,
                                 view_cone_distortion,
                                 value_type="noisable",  # objects of different type are not noised
                                 invisible_value=-2
                                 ):
  """
  Apply partial observability-induced transformations on physical quantity value, with respect to the player_pos and the
  value_pos.

  :param value:
  :param value_pos:
  :param player_pos:
  :param player_view_direction:
  :param depth_noise:
  :param view_obstruction:
  :param view_cone_xy_opening:
  :param view_cone_z_opening:
  :param value_type:
  :return:
  """

  # Convert to coordinates relative to player position
  rel_pos = value_pos - player_pos[:value_pos.shape[0]]
  rel_dist = np.linalg.norm(rel_pos)
  po_value = value.copy() if isinstance(value, np.ndarray) else value

  def signed_angle(a, b):
      t = np.degrees(np.arctan2(a[0]*b[1]-b[0]*a[1],a[0]*a[1]+b[0]*b[1]))
      return (-(180+t) if t < 0 else t)


  rel_angle = signed_angle(rel_pos[:2], player_view_direction[:2])
  rel_z_angle = np.arctan(rel_pos[2]/np.linalg.norm(rel_pos)) * 180.0 / np.pi if sum(rel_pos) != 0.0 else 0.0

  # Determine whether value location is visible wrt view cone
  if view_cone_xy_opening < 360:
      if (rel_angle < -view_cone_xy_opening // 2) or (rel_angle > view_cone_xy_opening // 2):
          is_visible = False
          po_value.fill(invisible_value)
      else:
        is_visible = True

  if view_cone_z_opening < 90 and is_visible:
      if rel_z_angle > view_cone_z_opening:
          is_visible = False
          po_value.fill(invisible_value)
      else:
        is_visible = True


  # If value is visible and of type "noisable", apply suitable noising
  if is_visible:
      if value_type == "noisable":
          if depth_noise is not None:
              if depth_noise.get("type", None) == "gaussian":
                  if depth_noise.get("attenuation_type", None) == "fixed_angular_resolution":
                      angular_resolution = depth_noise.get("angular_resolution_degrees", 2) * np.pi / 180.0
                      sigma = rel_dist * angular_resolution
                      # noise along tangential direction
                      if np.linalg.norm(po_value) != 0.0:
                          po_value += np.random.normal(0, sigma) * np.array([po_value[1] / np.linalg.norm(po_value),
                                                                             -po_value[0] / np.linalg.norm(po_value),
                                                                             po_value[2] / np.linalg.norm(po_value)])

  return po_value, is_visible


class MAPOSimple115StateWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to 115-features state.

     Each Observation is converted to coordinates relative to the respective player's absolute position (ego-frame)

     In addition, each observation is modified so as to respect partial observability constraints resulting
     from:
      - restricted view cone
      - depth noise
      - view obstruction
  """

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

    def add_zero(vec):
        return np.concatenate([vec, np.array([0])], -1)

    player_view_directions = getattr(self, "player_view_directions", None)
    if player_view_directions is None:
        # set player view directions to players facing forward at the beginning
        self.player_view_directions = np.zeros((len(observation), 2))
        self.player_view_directions[:, 0] = 1

    # update player view directions
    for player_id, obs in enumerate(observation):
        if np.sum(obs["left_team_direction"]) != 0.0:
            self.player_view_directions[player_id] = obs["left_team_direction"] / np.linalg.norm(obs["left_team_direction"])

    final_obs = []
    for player_id, obs in enumerate(observation):
      player_pos = add_zero(obs['left_team'][player_id])  # [x,y] of active player
      player_view_direction = add_zero(self.player_view_directions[player_id])

      apply_po = partial(_apply_partial_observability,
                         player_pos=player_pos,
                         player_view_direction=player_view_direction,
                         depth_noise=getattr(self, "po_noise", {"type":"gaussian", "sigma":0.1, "attenuation_type": "fixed_angular_resolution", "angular_resolution_degrees":0.2}),
                         view_obstruction=getattr(self, "po_view_obstruction", True),
                         view_cone_xy_opening=getattr(self, "po_view_cone_xy_opening", 160),  # 120 degrees corresponds to 2/3 obstructed
                         view_cone_z_opening=getattr(self, "po_view_cone_z_opening", 70),  # cannot see objects coming in at very steep angles
                         view_cone_distortion=getattr(self, "view_cone_distortion", None),  # TODO: distortion in view cone fringes
                         invisible_value=0
                         )

      left_team = obs['left_team']
      left_team = np.stack([apply_po(add_zero(o), value_pos=add_zero(o))[0][:2] for o in left_team])
      left_team_direction = obs['left_team_direction']
      left_team_direction = np.stack([apply_po(add_zero(o), value_pos=add_zero(o))[0][:2] for o in left_team_direction])
      right_team = obs['right_team']
      right_team = np.stack([apply_po(add_zero(o), value_pos=add_zero(o))[0][:2] for o in right_team])
      right_team_direction = obs['right_team_direction']
      right_team_direction = np.stack([apply_po(add_zero(o), value_pos=add_zero(o))[0][:2] for o in right_team_direction])

      # If there were less than 11vs11 players we backfill missing values with
      # -1.
      # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)

      # ball position
      ball, ball_is_visible = apply_po(obs['ball'],
                                       value_pos=obs['ball'])

      ######## determine if there are any obstructions ########

      ######## determine other object properties ##############

      if ball_is_visible:
          # ball direction
          ball_direction, _ = apply_po(obs['ball_direction'],
                                       value_pos=obs['ball'])

          # one hot encoding of which team owns the ball
          ball_owned, _ = apply_po(obs['ball_owned_team'],
                                   value_type="discrete",
                                   value_pos=obs['ball'],
                                   invisible_value=-2)
          if ball_owned == -1:
            ball_owned = [1, 0, 0]
          elif ball_owned == 0:
            ball_owned = [0, 1, 0]
          elif ball_owned == 1:
            ball_owned = [0, 0, 1]
      else:
        ball_direction = np.zeros_like(obs['ball_direction'])
        ball_owned = [0, 0, 0]

      ######## now extend observations #########################

      o = []
      o.extend(left_team.flatten().tolist())
      o.extend(left_team_direction.flatten().tolist())
      o.extend(right_team.flatten().tolist())
      o.extend(right_team_direction.flatten().tolist())

      if len(o) < 88:
          o.extend([-1] * (88 - len(o)))

      o.extend(ball.flatten().tolist())
      o.extend(ball_direction.flatten().tolist())
      o.extend(ball_owned)

      active = [0] * 11

      if obs['active'] != -1:
        active[obs['active']] = 1
      o.extend(active)

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)

      # Check whether any of the elements are view-obstructed!


      final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)
