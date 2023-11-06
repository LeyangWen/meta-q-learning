import math
from rand_param_envs import gym
from rand_param_envs.gym import spaces
from rand_param_envs.gym.utils import seeding
import numpy as np
import os


class KukaHumanResponse(gym.Env):
    def __init__(self, verbose=False):
        '''
        :param max_steps: maximum number of steps
        :param verbose: whether to print out information
        Currently relies on the Kuka..._Rand class to load the human response data
        '''
        super(KukaHumanResponse, self).__init__()
        self.verbose = verbose
        self.continous_change_speed = 3
        # Define action and observation spaces
        # Action: 2 continuous values, 3 binary values, merge into one continuous box
        # first two elements (movement speed change; arm swing speed change), between -1 and 1
        # three random binary elements (proximity, level of autonomy, leader of collaboration)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        # Observation: 4 continuous values, 2 binary values, merge into one continuous box
        # first two elements human response (valence, arousal), between -inf and inf
        # second two elements (movement speed; arm swing speed) between 27.8, 23.8 max speed and 143.8, 109.1
        # three random binary elements (proximity, level of autonomy, leader of collaboration)
        human_res_low_bnd, human_res_high_bnd = [-20, 20]
        move_spd_low_bnd, move_spd_high_bnd = [27.8, 143.8]
        arm_spd_low_bnd, arm_spd_high_bnd = [23.8, 109.1]
        low_binary, high_binary = [-1.0, 1.0]
        self.observation_space = spaces.Box(low=np.array([human_res_low_bnd, human_res_low_bnd, move_spd_low_bnd, arm_spd_low_bnd, low_binary, low_binary, low_binary]),
                                            high=np.array([human_res_high_bnd, human_res_high_bnd, move_spd_high_bnd, arm_spd_high_bnd, high_binary, high_binary, high_binary])
                                            )
        # todo: add to init input after testing
        self.max_steps = 1000
        self.max_brick_time = 4*60  # 4 hour
        self.learning_steps = 20  # can not reach don condition
        self._seed()
        self.reset()

    def _seed(self, seed=None):  # todo: not really integrated at the moment
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_human_response(self, curr_state, simulated=True):
        ''' compute human response from current state
        :param curr_state: {}
        :param simulated: whether to use simulated human response in csv file
        :return: np.array(valence, arousal)
        '''
        if simulated:
            this_state = curr_state[2:]
            currStateMat = np.array(
                [1, this_state[0] ** 2, this_state[0], this_state[1] ** 2, this_state[1], this_state[2],
                 this_state[3], this_state[4], 1])
            valence = np.matmul(currStateMat, self.val_coeffs) * self.val_std + self.val_mean
            arousal = np.matmul(currStateMat, self.aro_coeffs) * self.aro_std + self.aro_mean
            return np.array([valence, arousal])
        else:
            raise NotImplementedError

    def reset(self):
        self.state = self.observation_space.sample()
        self.state[:2] = self.compute_human_response(self.state)

        self.done = False
        self.steps_taken = 0
        self.total_time = 0
        self.total_brick = 0
        return np.array(self.state)

    def compute_reward(self):
        state = self.state
        steps = self.steps_taken
        # from human response
        valence, arousal = state[:2]
        movement_speed, arm_swing_speed = state[2:4]
        proximity, level_of_autonomy, leader_of_collaboration = state[4:]

        aWeights = [0, 1, 0]
        done_option = 3
        if done_option == 1:  # big reward after reached goal and end
            if valence > self.val_mean and arousal > self.aro_mean:
                reward_from_human_response = 1000
                self.done = True
        elif done_option == 2:  # done if we fail to maintain good human response after n step
            if valence < self.val_mean or arousal < self.aro_mean:
                if steps > self.learning_steps:  # eeg at 0.5 hz, 2 brick per minute. 10 (max 20) brick to learn
                    self.done = True
                # else:
                #   aWeights = [0.001, 0.0001, 0]
                #   # leyang changed to very small until after learning steps
        elif done_option == 3:  # force 5 min break if we fail to maintain good human response, done at max brick time
            if valence < self.val_mean or arousal < self.aro_mean:
                self.total_time += 5 # 5 min break

        reward_from_human_response = (valence - self.val_mean) + (arousal - self.aro_mean)

        # reward from prod
        # t = s / v   (v = s / t)
        MoveDistance = 305  # cm
        ArmDistance = 2 * 60  # cm
        travelTime = 2 * (ArmDistance / arm_swing_speed + MoveDistance / movement_speed)
        if proximity > 0:
            travelTime += 3
        if level_of_autonomy > 0:
            travelTime += 2
        if leader_of_collaboration > 0:
            travelTime += 1

        # bricks per hour
        #
        self.total_time += travelTime / 60
        self.total_brick += 1

        prod_reward_options = 2
        if prod_reward_options == 0:  # reward by current brick productivity (speed)
            reward_from_prod = 3600 / travelTime
        elif prod_reward_options == 1:  # cumulative brick productivity
            reward_from_prod = self.total_brick / self.total_time
        elif prod_reward_options == 2:  # +1 per brick, end by max time
            reward_from_prod = 1

        # reward from steps
        reward_from_steps = 0.1 * steps

        aReward = [reward_from_human_response, reward_from_prod, reward_from_steps]
        weighted_reward = np.matmul(aReward, aWeights)

        if self.steps_taken % 10 == 1 and self.verbose:  # change to True to print output
            print(f"Step: {self.steps_taken}")
            print(f"State: {state}")
            print(
                f"Reward from Human Response: {reward_from_human_response:.2f}*{aWeights[0]} = {reward_from_human_response * aWeights[0]:.2f}")
            print(f"Reward from Prod: {reward_from_prod:.2f}*{aWeights[1]} = {reward_from_prod * aWeights[1]:.2f}")
            print(
                f"Reward from Steps: {reward_from_steps:.2f}*{aWeights[2]} = {reward_from_steps * aWeights[2]:.2f}")
            print(f"Weighted Reward: {weighted_reward:.2f}")
        return weighted_reward

    def step(self, action):
        self.action = action
        self.steps_taken += 1
        movement_speed, arm_swing_speed = self.state[2:4]

        # update movement speed and arm swing speed
        movement_speed += self.action[0] * self.continous_change_speed
        arm_swing_speed += self.action[1] * self.continous_change_speed
        # check out of bnd
        movement_speed = min(max(movement_speed, self.observation_space.low[2]), self.observation_space.high[2])
        arm_swing_speed = min(max(arm_swing_speed, self.observation_space.low[3]), self.observation_space.high[3])
        self.state[2:4] = movement_speed, arm_swing_speed

        # update proximity, level_of_autonomy, leader_of_collaboration
        self.state[4:] = action[2:]

        # update human response
        self.state[:2] = self.compute_human_response(self.state)

        reward = self.compute_reward()
        done = self.done
        if self.total_time > self.max_brick_time:
            done = True
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.steps_taken}, State: {self.state}, "
              f"Action,{self.action}, Done: {self.done}, "
              f"Reward: {self.compute_reward()}")


class KukaHumanResponse_Rand(KukaHumanResponse):
    def __init__(self, task={}, n_tasks=18, randomize_tasks=False):
        '''
        :param task: task is a dictionary with key 'goal_position'
        :param n_tasks: number of tasks, 18 subjects in total, 13 for training, 5 for testing
        :param randomize_tasks: whether to randomize tasks
        '''
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self.load_from_task(self.tasks[0])
        # goal_position
        super(KukaHumanResponse_Rand, self).__init__()

    def load_response(self, file, index):
        data = np.loadtxt(file, delimiter=',')
        return data[index, :9], data[index, 9], data[index, 10]

    def sample_tasks(self, num_tasks):
        BASE_FOLDER = 'rand_param_envs/gym/envs/HRC/human_response/'
        valence_file = os.path.join(BASE_FOLDER, 'valence_merge.csv')
        arousal_file = os.path.join(BASE_FOLDER, 'arousal_merge.csv')
        tasks = []
        for i in range(num_tasks):
            this_task = {}
            this_task['sub_id'] = i
            this_task['val_coeffs'], this_task['val_mean'], this_task['val_std'] = self.load_response(valence_file, i)
            this_task['aro_coeffs'], this_task['aro_mean'], this_task['aro_std'] = self.load_response(arousal_file, i)
            tasks.append(this_task)
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self.load_from_task(self.tasks[idx])
        self.reset()

    def load_from_task(self, task):
        self._task = task
        self.sub_id = task['sub_id']
        self.val_coeffs = task['val_coeffs']
        self.val_mean = task['val_mean']
        self.val_std = task['val_std']
        self.aro_coeffs = task['aro_coeffs']
        self.aro_mean = task['aro_mean']
        self.aro_std = task['aro_std']


