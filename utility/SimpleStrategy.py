import numpy as np


class SimpleStrategy:
    def __init__(self):
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

        self.best_state = None
        self.best_human_response = None

    def run(self):
        pass


class MaxProductivityStrategy(SimpleStrategy):
    """
    This strategy aims to maximize HRC team productivity, don't care about human response
    Best state: [move_spd_high_bnd, arm_spd_high_bnd, low_binary, low_binary, high_binary]
    """
    def __init__(self):
        super().__init__()
        self.best_state = np.array(self.move_spd_high_bnd, self.arm_spd_high_bnd, self.low_binary, self.low_binary, self.high_binary)

    def find_best_state(self, env, data_buffer):
        this_human_response = env.compute_human_response(self.best_state)
        # if args.normalized_human_response, env returns normalized human response, otherwise, return actual human response
        # data_buffer knows if it still needed to be normalized, so just pass it to data_buffer.normalize_human_response
        valance, arousal = data_buffer.normalize_human_response(this_human_response)
        self.best_human_response = np.array([valance, arousal])


class SearchDownStrategy(SimpleStrategy):
    """
    This strategy aims to search down the robot speed and find the first state that have acceptable human response
    For binary variables, the binary cutoff (%) determines when it is set to the not productive side
    """
    def __init__(self, num_combinations=100, binary_cutoff=0.5):
        super().__init__()
        self.num_combinations = num_combinations
        self.binary_cutoff = binary_cutoff
        self.all_combinations = np.array([])
        self.set_all_combinations()

    def set_all_combinations(self):
        move_spd = np.linspace(self.move_spd_high_bnd, self.move_spd_low_bnd, self.num_combinations)
        arm_spd = np.linspace(self.arm_spd_high_bnd, self.arm_spd_low_bnd, self.num_combinations)
        binary_cutoff_num = int(self.num_combinations * self.binary_cutoff)
        binary_array = np.ones(self.num_combinations)
        binary_array[:binary_cutoff_num] = -1
        binary_array_1 = binary_array.copy()  # low bnd good
        binary_array_2 = binary_array.copy()  # low bnd good
        binary_array_3 = binary_array.copy()*-1  # high bnd good
        self.all_combinations = np.array([move_spd, arm_spd, binary_array_1, binary_array_2, binary_array_3]).T

    def find_best_state(self, env, data_buffer):
        for state in self.all_combinations:
            this_human_response = env.compute_human_response(self.best_state)
            # if args.normalized_human_response, env returns normalized human response, otherwise, return actual human response
            # data_buffer knows if it still needed to be normalized, so just pass it to data_buffer.normalize_human_response
            valance, arousal = data_buffer.normalize_human_response(this_human_response)

            if arousal > 0 and valance > 0:
                break
        self.best_state = state
        self.best_human_response = np.array([valance, arousal])
        return state  # if no acceptable state found, return the last state