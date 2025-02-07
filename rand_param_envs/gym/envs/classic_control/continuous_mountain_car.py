# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from 
https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
"""

import math
import random

from rand_param_envs import gym
from rand_param_envs.gym import spaces
from rand_param_envs.gym.utils import seeding
import numpy as np


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_position=0.45, verbose=True):
        self.verbose = verbose
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self._goal_position = goal_position  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.power = 0.0015*100
        self.cur_step = 0
        self.max_step = 999  # useless here, specified in main.py args.num_initial_steps

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Box(self.min_action, self.max_action, shape = (1,))
        self.observation_space = spaces.Box(self.low_state, self.high_state)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self._goal_position)
        time_out = bool(self.cur_step >= self.max_step)
        self.cur_step += 1

        reward = 0
        if self.verbose:
            if random.Random().random() < 0.001: # or done or time_out:
                print(f'step: {self.cur_step}/{self.max_step}, position: {position:.2f}, velocity: {velocity:.2f}, '
                      f'action: {action[0]:.2f}, done: {done}, goal: {self._goal_position:.2f}')
            # if done:
            #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ done')
            # if time_out:
            #     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ time out')
        if done:
            reward = 100.0
        reward -= math.pow(action[0], 2)*0.1
        # reward -= math.pow((self._goal_position - position), 2)*0.01

        self.state = np.array([position, velocity])
        if time_out:
            done = True
        return self.state, reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        self.cur_step = 0
        return np.array(self.state)

#    def get_state(self):
#        return self.state

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from rand_param_envs.gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self._goal_position-self.min_position)*scale
            flagy1 = self._height(self._goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


class Continuous_MountainCarEnv_Rand(Continuous_MountainCarEnv):
    '''
    Randomized goal version of Continuous_MountainCarEnv
    '''
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_position = task.get('goal_position', 0.4)
        self._goal = self._goal_position
        # goal_position
        super(Continuous_MountainCarEnv_Rand, self).__init__(goal_position=self._goal_position)

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        goal_positions = np.random.uniform(0.3, 0.5, size=(num_tasks,))
        tasks = [{'goal_position': goal_position} for goal_position in goal_positions]
        # print(goal_positions)
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_position = self._task['goal_position']
        self._goal = self._goal_position
        self._reset()

