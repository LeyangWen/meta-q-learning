import math
from rand_param_envs import gym
from rand_param_envs.gym import spaces
from rand_param_envs.gym.utils import seeding
import numpy as np
import os
from deprecated import deprecated


class KukaHumanResponse(gym.Env):
    def __init__(self, verbose=True, normalized=True):
        '''
        :param max_steps: maximum number of steps
        :param verbose: whether to print out information
        Currently relies on the Kuka..._Rand class to load the human response data
        '''
        super(KukaHumanResponse, self).__init__()
        # Define action and observation spaces
        # Action: 2 continuous values, 3 binary values, merge into one continuous box
        # first two elements (movement speed change; arm swing speed change), between -1 and 1
        # three random binary elements (proximity, level of autonomy, leader of collaboration)
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))
        # Observation: 4 continuous values, 2 binary values, merge into one continuous box
        # first two elements human response (valence, arousal), between -inf and inf
        # second two elements (movement speed; arm swing speed) between 27.8, 23.8 max speed and 143.8, 109.1
        # three random binary elements (proximity, level of autonomy, leader of collaboration)
        human_res_low_bnd, human_res_high_bnd = [-20, 20]
        move_spd_low_bnd, move_spd_high_bnd = [27.8, 143.8]
        arm_spd_low_bnd, arm_spd_high_bnd = [23.8, 109.1]
        low_binary, high_binary = [-1.0, 1.0]
        self.move_spd_low_bnd = move_spd_low_bnd
        self.move_spd_high_bnd = move_spd_high_bnd
        self.arm_spd_low_bnd = arm_spd_low_bnd
        self.arm_spd_high_bnd = arm_spd_high_bnd
        self.low_binary = low_binary
        self.high_binary = high_binary

        self.observation_space = spaces.Box(low=np.array([human_res_low_bnd, human_res_low_bnd, move_spd_low_bnd, arm_spd_low_bnd, low_binary, low_binary, low_binary]),
                                            high=np.array([human_res_high_bnd, human_res_high_bnd, move_spd_high_bnd, arm_spd_high_bnd, high_binary, high_binary, high_binary])
                                            )
        # todo: add to init input after testing
        self.verbose = verbose
        self.normalized = normalized

        self.continuous_change_speed = 15
        self.half_break_time = 2 # min #todo: maybe make this smaller 90% break time is very long
        self.max_steps = 1000  # not used
        self.max_brick_time = 4*60  # 4 hour
        self.learning_steps = 50  # can not reach done condition
        self.done_option = 5
        self.prod_reward_options = 3
        self._seed()
        self._reset()

    def _seed(self, seed=None):  # todo: not really integrated at the moment
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def calculate_traveltime(movement_speed, arm_swing_speed, proximity, level_of_autonomy, leader_of_collaboration):
        """
        :return: travel time in min
        """
        # t = s / v   (v = s / t)
        MoveDistance = 305  # cm
        ArmDistance = 2 * 60  # cm
        travelTime = 2 * (ArmDistance / arm_swing_speed + MoveDistance / movement_speed)
        if proximity > 0:  # +2s if positive
            travelTime += 2
        if level_of_autonomy > 0:  # +3s if positive
            travelTime += 3
        if leader_of_collaboration < 0:  # +1s if negative
            travelTime += 1
        travelTime += 5  # 5s for brick placement
        return travelTime / 60  # min

    @staticmethod
    def calculate_productivity(travelTime):
        """
        :return: bricks per hour
        """
        return 60 / travelTime

    def compute_human_response(self, curr_state, simulated=True, normalized="env"):
        """ compute human response from current state
        :param curr_state: [valence, arousal, movement_speed, arm_swing_speed, proximity, level_of_autonomy, leader_of_collaboration], optional first two elements
        :param simulated: whether to use simulated human response in csv file
        :return: np.array(valence, arousal)
        if self.normalized, valence and arousal zero mean and unit variance based on each subject
        else, valence and arousal are raw values
        """
        if simulated:
            if len(curr_state) == 7:
                robot_state = curr_state[2:].copy()  # discard first two elements, valence and arousal
            elif len(curr_state) == 5:
                robot_state = curr_state.copy()
            else:
                raise ValueError("curr_state should be length 5 or 7")

            if not np.all(np.abs(robot_state[2:]) == 1):   # check if last three is binary -1, 1
                raise ValueError("Last three elements of robot state should be binary -1 or 1 "
                                 f"got {robot_state} instead")
            robot_state[2:] = (robot_state[2:] + 1) / 2   # convert -1, 1 to 0, 1
            currStateMat = np.array(
                [1, robot_state[0] ** 2, robot_state[0], robot_state[1] ** 2, robot_state[1], robot_state[2],
                 robot_state[3], robot_state[4], 1])

            if normalized == "env":
                normalized = self.normalized
            else:
                assert normalized in [True, False]

            if normalized:
                valence = np.matmul(currStateMat, self.val_coeffs)
                arousal = np.matmul(currStateMat, self.aro_coeffs)
            else:
                valence = np.matmul(currStateMat, self.val_coeffs) * self.val_std + self.val_mean
                arousal = np.matmul(currStateMat, self.aro_coeffs) * self.aro_std + self.aro_mean
            return np.array([valence, arousal])
        else:
            raise NotImplementedError

    def _reset(self):
        self.state = self.observation_space.sample()
        # convert last three to binary -1, 1
        self.state[4:] = np.sign(self.state[4:])
        self.state[:2] = self.compute_human_response(self.state)

        self.done = False
        self.steps_taken = 0
        self.total_time = 0
        self.total_brick = 0
        self.total_break = 0
        self.total_break_time = 0  # min
        return np.array(self.state)

    @deprecated
    def compute_reward(self):
        state = self.state
        steps = self.steps_taken
        # from human response
        valence, arousal = state[:2]
        movement_speed, arm_swing_speed = state[2:4]
        proximity, level_of_autonomy, leader_of_collaboration = state[4:]

        aWeights = [0, 1, 0]
        done_option = self.done_option
        force_break_time = 0
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
        elif done_option == 3:  # force n min break if we fail to maintain good human response, done at max brick time
            # todo: this accepts normalized valence and arousal, need to change to raw valence and arousal
            if steps > self.learning_steps:
                sig_break = lambda x: -1 / (1 + np.exp(-x * 2)) + 1
                force_break_time = self.half_break_time * (sig_break(valence) + sig_break(arousal))
                self.total_time += force_break_time
                self.total_break_time += force_break_time
                if valence < 0 or arousal < 0:
                    self.total_break += 1
                    # with np.printoptions(precision=2, suppress=True):
                    #     print(f"@@@@@ Force Break #{self.total_break}:{force_break_time}, total time: {self.total_time:.2f}, brick: {self.total_brick}, sub_id: {self.sub_id}, state: {self.state}")
        elif done_option == 4:  # stop after the learning steps
            if steps > self.learning_steps:
                self.done = True
                valence_positive = valence > 0  # todo: this accepts normalized valence and arousal, need to change to raw valence and arousal
                arousal_positive = arousal > 0
                hr_factor_half = (int(valence_positive) + int(arousal_positive))/2
                hr_factor = arousal_positive and valence_positive
            else:
                hr_factor_half = 0
                hr_factor = 0
        elif done_option == 5:  # stop after the learning steps
            valence_positive = valence > 0  # todo: this accepts normalized valence and arousal, need to change to raw valence and arousal
            arousal_positive = arousal > 0
            hr_factor_half = (int(valence_positive) + int(arousal_positive)) / 2
            hr_factor = arousal_positive and valence_positive
            if steps > self.learning_steps:
                self.done = True
            else:
                hr_factor_half = hr_factor_half/self.learning_steps
                hr_factor = hr_factor/self.learning_steps

        reward_from_human_response = (valence - self.val_mean) + (arousal - self.aro_mean)  # check if this need normalization

        # reward from prod
        travelTime = self.calculate_traveltime(movement_speed, arm_swing_speed, proximity, level_of_autonomy, leader_of_collaboration)
        # bricks per hour
        #

        self.travelTime = travelTime
        self.total_time += travelTime
        self.total_brick += 1

        prod_reward_options = self.prod_reward_options
        if prod_reward_options == 0:  # reward by current brick productivity (speed) brick/hour
            reward_from_prod = 60 / (travelTime+force_break_time)
        elif prod_reward_options == 1:  # cumulative brick productivity
            reward_from_prod = self.total_brick / self.total_time
        elif prod_reward_options == 2:  # +1 per brick, end by max time
            reward_from_prod = 1
        elif prod_reward_options == 3:  # only reward based on the last step
            self.hr_factor = hr_factor
            reward_from_prod = self.hr_factor * 60 / (travelTime)
        elif prod_reward_options == 4:  # only reward based on the last step
            self.hr_factor = hr_factor_half
            reward_from_prod = self.hr_factor * 60 / (travelTime)

        # reward from steps
        reward_from_steps = 0.1 * steps

        aReward = [reward_from_human_response, reward_from_prod, reward_from_steps]
        weighted_reward = np.matmul(aReward, aWeights)

        if self.steps_taken % 10 == 1 and self.verbose and False:  # change to True to print output
            print(f"Step: {self.steps_taken}")
            print(f"State: {state}")
            print(
                f"Reward from Human Response: {reward_from_human_response:.2f}*{aWeights[0]} = {reward_from_human_response * aWeights[0]:.2f}")
            print(f"Reward from Prod: {reward_from_prod:.2f}*{aWeights[1]} = {reward_from_prod * aWeights[1]:.2f}")
            print(
                f"Reward from Steps: {reward_from_steps:.2f}*{aWeights[2]} = {reward_from_steps * aWeights[2]:.2f}")
            print(f"Weighted Reward: {weighted_reward:.2f}")
        return weighted_reward

    @deprecated
    def compute_prod_reward(self, curr_state):
        """ compute productivity reward from current state
        :param curr_state:
        :return: productivity reward
        """
        travelTime = self.calculate_traveltime(curr_state[2], curr_state[3], curr_state[4], curr_state[5], curr_state[6])
        productivity = self.calculate_productivity(travelTime)
        return productivity

    @deprecated
    def _step(self, action):
        self.action = action
        self.steps_taken += 1
        movement_speed, arm_swing_speed = self.state[2:4]

        # update movement speed and arm swing speed
        movement_speed += self.action[0] * self.continuous_change_speed
        arm_swing_speed += self.action[1] * self.continuous_change_speed
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
        # if self.total_time > self.max_brick_time:
        #     done = True
        #     if self.verbose:
        #         print(f"@@@@@@@@@@@@@@@@@@@ sub #{self.sub_id} reached max time, summary @@@@@@@@@@@@@@@@@@@")
        #         print(f"Total time: {self.total_time:.2f}, brick: {self.total_brick}, break: {self.total_break} - {self.total_break_time / self.total_time * 100:.1f}%")
        #         with np.printoptions(precision=2, suppress=True):
        #             print(f"State: human response: {self.state[:2]}, continuous: {self.state[2:4]}, binary: {self.state[4:]}")
        #             print(f"Action: continuous: {self.action[:2]}, binary: {self.action[2:]}")
        #         print()

        if done and self.verbose:
            print(f"@@@@@@@@@@@@@@@@@@@ sub #{self.sub_id} finished learning after {self.learning_steps}, summary @@@@@@@@@@@@@@@@@@@")
            prod = 60 / (self.travelTime)
            print(f"Productivity: {prod:.1f} brick/hour, HR factor: {self.hr_factor}, reward: {reward:.1f}")
            with np.printoptions(precision=2, suppress=True):
                print(f"State: human response: {self.state[:2]}, continuous: {self.state[2:4]}, binary: {self.state[4:]}")
                print(f"Action: continuous: {self.action[:2]}, binary: {self.action[2:]}")
            print()

        return self.state, reward, done, {}

    def _render(self):
        print(f"Step: {self.steps_taken}, State: {self.state}, "
              f"Action,{self.action}, Done: {self.done}, "
              f"Reward: {self.compute_reward()}")


class KukaHumanResponse_Rand(KukaHumanResponse):
    def __init__(self, task={}, n_tasks=18, randomize_tasks=False, verbose=True, normalized=True):
        '''
        :param task: task is a dictionary with key 'goal_position'
        :param n_tasks: number of tasks, 18 subjects in total, 13 for training, 5 for testing
        :param randomize_tasks: whether to randomize tasks
        '''
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self.load_from_task(self.tasks[0])
        # goal_position
        super(KukaHumanResponse_Rand, self).__init__(verbose=verbose, normalized=normalized)

    def load_response(self, file, index):
        data = np.loadtxt(file, delimiter=',')
        return data[index, :9], data[index, 9], data[index, 10]

    def step(self, action):
        return self._step(action)

    def reset(self):
        return self._reset()

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
        self._reset()

    def load_from_task(self, task):
        self._task = task
        self.sub_id = task['sub_id']
        self.val_coeffs = task['val_coeffs']
        self.val_mean = task['val_mean']
        self.val_std = task['val_std']
        self.aro_coeffs = task['aro_coeffs']
        self.aro_mean = task['aro_mean']
        self.aro_std = task['aro_std']



if __name__ == '__main__':
    env = KukaHumanResponse_Rand()
    env.reset()
    continuous_bin = np.arange(0, 1.01, 0.01)
    binary_bin = [0, 1]
    bin_map = [(0,0, a, b, x, y, z) for a in continuous_bin for b in continuous_bin for x in binary_bin for y in binary_bin for z in binary_bin]
    move_spd_low_bnd, move_spd_high_bnd = [27.8, 143.8]
    arm_spd_low_bnd, arm_spd_high_bnd = [23.8, 109.1]

    for subject_id in range(18):
        print('------------------', subject_id, '------------------')
        largest_productivity = 0
        largest_state = None
        largest_av = None
        env.reset_task(subject_id)
        for i in range(len(bin_map)):
            state = bin_map[i]
            state = np.array(state)
            state[0+2] = state[0+2]*(move_spd_high_bnd - move_spd_low_bnd) + move_spd_low_bnd
            state[1+2] = state[1+2]*(arm_spd_high_bnd - arm_spd_low_bnd) + arm_spd_low_bnd
            state[2+2] = state[2+2]
            state[3+2] = state[3+2]
            state[4+2] = state[4+2]
            traveltime = env.calculate_traveltime(state[2], state[3], state[4], state[5], state[6])
            productivity = env.calculate_productivity(traveltime)
            a, v = env.compute_human_response(state)
            if a >0 and v >0:
                if productivity > largest_productivity:
                    largest_productivity = productivity
                    largest_state = state
                    largest_av = (a, v)
        print(f"largest productivity: {largest_productivity}, state: {largest_state}, aro-var: {largest_av}")


