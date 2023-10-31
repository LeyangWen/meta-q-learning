import math
from rand_param_envs import gym
from rand_param_envs.gym import spaces
from rand_param_envs.gym.utils import seeding
import numpy as np
import os


class KukaHumanResponse(gym.Env):  # todo: add another rand task subclass
    def __init__(self, max_sbj=13, max_steps=1000):
        super(KukaHumanResponse, self).__init__()
        self.verbose = True  # todo: change verbose to self.verbose
        # Define action and observation spaces
        # Action: 2 continuous values, 3 binary values
        # first two elements between 27.8, 23.8 max speed and 143.8, 109.1
        # three random binary elements (0 or 1)
        self.continous_change_speed = 50  # 20 #5  # ([0,1]-0.5)*speed*2  #leyang: try different value
        self.action_space = {  # todo: change action and observation from dict to spaces
            'continuous': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            # movement speed; arm swing speed
            'discrete': spaces.MultiDiscrete([2, 2, 2])  # proximity, level of autonomy, leader of collaboration
        }
        # leyang: changed continuous to change the state, not directly set value
        low1, low2, high1, high2 = [27.8, 23.8, 143.8, 109.1]
        self.observation_space = {
            'human_response': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'robot_continuous': spaces.Box(low=np.array([low1, low2]), high=np.array([high1, high2]), shape=(2,),
                                           dtype=np.float32),
            # spaces.Box(low=1.0, high=143.8, shape=(2,), dtype=np.float32),
            'robot_discrete': spaces.MultiDiscrete([2, 2, 2])
        }

        # Initialize environment parameters
        self.max_sbj = max_sbj
        self.max_steps = max_steps
        self.curr_sbj_index = 0

        # ... (rest of your initialization code)
        # Francis (2023.10.13): File names are changed. We also have 'valence_test.csv' and 'arousal_test.csv'.
        self.max_sbj_index = max_sbj - 1
        self.valence_file = os.path.join(BASE_FOLDER, 'valence_train.csv')
        self.arousal_file = os.path.join(BASE_FOLDER, 'arousal_train.csv')

        self.max_steps = max_steps
        # self.grace_steps = 5  # we can reach the done condition for several frames without stopping, hope it can recover
        self.learning_steps = 20  # can not reach don condition
        self.steps_taken = 0
        self.done = False

    # Francis (2023.10.13): I made modification to csv files. [0:9] are regression coefficients, [9] is mean, and [10] is std of valence/arousal.
    def load_response(self, file, index):
        data = np.loadtxt(file, delimiter=',')
        return data[index, :9], data[index, 9], data[index, 10]

    def compute_human_response(self, curr_state, simulated=True):
        ''' compute human response from current state
        :param curr_state: {}
        :return: np.array(valence, arousal)
        '''
        if simulated:
            this_state = [0, 0, 0, 0, 0]
            this_state[:2] = curr_state['robot_continuous']
            this_state[2:] = curr_state['robot_discrete']
            currStateMat = np.array(
                [1, this_state[0] ** 2, this_state[0], this_state[1] ** 2, this_state[1], this_state[2],
                 this_state[3], this_state[4], 1])

            # Francis (2023.10.13): I made changes. Now valence and arousal are not standardized but their original distributions corresponding to each person.
            valence = np.matmul(currStateMat, self.val_coeffs) * self.val_std + self.val_mean
            arousal = np.matmul(currStateMat, self.aro_coeffs) * self.aro_std + self.aro_mean
            return np.array([valence, arousal])
        else:
            raise NotImplementedError

    def reset(self, subject='rand'):
        """
        Resets the environment.
        subject: rand, next, 1  (int)
        """
        init_state = {}
        init_action = {}  # place holder to reflush
        for key, value in self.observation_space.items():
            init_state[key] = value.sample()
        for key, value in self.action_space.items():
            init_action[key] = value.sample()
        if subject == 'rand':
            self.curr_sbj_index = np.random.randint(self.max_sbj_index + 1)  # select one random subject
        elif subject == 'next':
            self.curr_sbj_index += 1
        else:
            self.curr_sbj_index = int(subject)
        if verbose:
            print('random subject index: ', self.curr_sbj_index)

        # Francis (2023.10.13): I made changes corresponding to the modified 'load_response' function
        self.val_coeffs, self.val_mean, self.val_std = self.load_response(self.valence_file, self.curr_sbj_index)
        self.aro_coeffs, self.aro_mean, self.aro_std = self.load_response(self.arousal_file, self.curr_sbj_index)

        human_response = self.compute_human_response(init_state)
        self.state = {
            'human_response': human_response,
            'robot_continuous': init_state['robot_continuous'],
            'robot_discrete': init_state['robot_discrete']
        }
        self.steps_taken = 0
        self.done = False
        self.action = init_action
        self.total_time = 0
        self.total_brick = 0
        return self.state_dict_to_np(self.state)

    def compute_reward(self, state, steps):
        # from human response
        valence, arousal = state['human_response']
        aWeights = [0, 1, 0]
        option = 2
        if option == 1:  # big reward after reached goal and end
            if valence > self.val_mean and arousal > self.aro_mean:
                reward_from_human_response = 1000
                self.done = True
        else:  # done if we fail to maintain good human response after n step
            if valence < self.val_mean or arousal < self.aro_mean:
                if steps > self.learning_steps:  # eeg at 0.5 hz, 2 brick per minute. 10 (max 20) brick to learn
                    self.done = True
                # else:
                #   aWeights = [0.001, 0.0001, 0]
                #   # leyang changed to very small until after learning steps

        # Francis (2023.10.13): I changed this reward by subtracting the mean values.
        reward_from_human_response = (valence - self.val_mean) + (arousal - self.aro_mean)

        # reward from prod
        # t = s / v   (v = s / t)
        MoveDistance = 305  # cm
        ArmDistance = 2 * 60  # cm
        travelTime = 2 * (ArmDistance / state['robot_continuous'][1] + MoveDistance / state['robot_continuous'][0])
        if state['robot_discrete'][0] == 1:
            travelTime += 3
        if state['robot_discrete'][1] == 1:
            travelTime += 2
        if state['robot_discrete'][2] == 1:
            travelTime += 1
        # reward from steps
        reward_from_steps = 0.1 * steps

        # bricks per hour
        # reward_from_prod =  3600 / travelTime # reward_from_steps *
        # Leyang changed rewared to cum brick prod
        self.total_time += travelTime / 60
        self.total_brick += 1
        reward_from_prod = self.total_brick / self.total_time

        aReward = [reward_from_human_response, reward_from_prod,
                   reward_from_steps]  # leyang, increase weight of productivity as step increase
        weighted_reward = np.matmul(aReward, aWeights)

        if self.steps_taken % 10 == 1 and verbose:  # change to True to print output
            print(f"Step: {self.steps_taken}")
            print(f"State: {state}")
            print(
                f"Reward from Human Response: {reward_from_human_response:.2f}*{aWeights[0]} = {reward_from_human_response * aWeights[0]:.2f}")
            print(f"Reward from Prod: {reward_from_prod:.2f}*{aWeights[1]} = {reward_from_prod * aWeights[1]:.2f}")
            print(
                f"Reward from Steps: {reward_from_steps:.2f}*{aWeights[2]} = {reward_from_steps * aWeights[2]:.2f}")
            print(f"Weighted Reward: {weighted_reward:.2f}")
        return weighted_reward

    @staticmethod
    def action_to_dict_np(input_action):
        action_dict = {}
        action_dict['continuous'] = input_action[:2]
        action_dict['discrete'] = [0, 0, 0]
        for _ in range(3):
            if input_action[2 + _] > 0.5:  # input should be 0 or 1, but just to be safe
                action_dict['discrete'][_] = 1
            else:
                action_dict['discrete'][_] = 0
        return action_dict

    @staticmethod
    def action_to_dict_tensor(input_action):
        action_dict = {}
        action_dict['continuous'] = tf.constant(input_action[:2], dtype=tf.float32)
        action_dict['discrete'] = tf.constant([0, 0, 0], dtype=tf.int32)

        for i in range(3):
            if input_action[2 + i] > 0.5:  # input should be 0 or 1, but just to be safe
                action_dict['discrete'] = tf.tensor_scatter_nd_update(action_dict['discrete'], [[i]], [1])
            else:
                action_dict['discrete'] = tf.tensor_scatter_nd_update(action_dict['discrete'], [[i]], [0])

        return action_dict

    @staticmethod
    def state_dict_to_np(input_state):
        state_np = np.zeros(7)
        state_np[:2] = input_state['human_response']
        state_np[2:4] = input_state['robot_continuous']
        state_np[4:] = input_state['robot_discrete']
        return state_np

    @staticmethod
    def state_dict_to_tensor(input_state):
        state_tf = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)
        state_tf = tf.tensor_scatter_nd_update(state_tf, [[0], [1]], input_state['human_response'])
        state_tf = tf.tensor_scatter_nd_update(state_tf, [[2], [3]], input_state['robot_continuous'])
        state_tf = tf.tensor_scatter_nd_update(state_tf, [[4], [5], [6]], input_state['robot_discrete'])
        return state_tf

    def step(self, input_action):
        """Applies the action and returns new state, reward, done, and info."""
        if type(input_action) == np.ndarray:
            # print(f'Converting {type(input_action)} to dict')
            self.action = self.action_to_dict_np(input_action)
        elif type(input_action) == dict:
            self.action = input_action
        else:
            raise TypeError(f"Action type - {type(input_action)} is not supported, need to be dict or array")
        self.steps_taken += 1
        old_state = self.state
        self.state = {}
        rand_speed_factor = np.random.random() / 2 + 0.5
        self.state['robot_continuous'] = (self.action[
                                              'continuous'] - 0.5) * self.continous_change_speed * rand_speed_factor * 2 + \
                                         old_state['robot_continuous']
        # check if robot_continuous state is out of observation space bound
        for _ in range(2):
            if self.state['robot_continuous'][_] > self.observation_space['robot_continuous'].high[_]:
                self.state['robot_continuous'][_] = self.observation_space['robot_continuous'].high[_]
                # print("Warning: robot_continuous state is out of observation space up bound")
            elif self.state['robot_continuous'][_] < self.observation_space['robot_continuous'].low[_]:
                self.state['robot_continuous'][_] = self.observation_space['robot_continuous'].low[_]
                # print("Warning: robot_continuous state is out of observation space low bound")
        self.state['robot_discrete'] = self.action['discrete']
        self.state['human_response'] = self.compute_human_response(self.state)

        reward = self.compute_reward(self.state, self.steps_taken)
        return self.state_dict_to_np(self.state), reward, self.done

    def render(self, mode='human'):
        """Displays the current state of the environment."""
        if mode == 'human':
            for key, value in self.state.items():
                print(f"{key}: {value}")

# Create an instance of the custom environment
env = HumanResponseGymEnv()

# Reset the environment
initial_observation = env.reset()

# # Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()