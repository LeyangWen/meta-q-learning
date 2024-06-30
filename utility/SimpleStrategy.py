import numpy as np
from utility.CriteriaChecker import CriteriaChecker
from utility.OptimalResult import OptimalResult


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

        # Optimal Result Object to stores best infos
        self.optimal_result = OptimalResult()

    def run(self):
        pass


class MaxProductivityStrategy(SimpleStrategy):
    """
    This strategy aims to maximize HRC team productivity, don't care about human response
    Best state: [move_spd_high_bnd, arm_spd_high_bnd, low_binary, low_binary, high_binary]
    """

    def __init__(self):
        super().__init__()
        self.strategy_name = 'MaxProductivity'
        self.optimal_result.best_robot_state = np.array(
            [self.move_spd_high_bnd, self.arm_spd_high_bnd, self.low_binary, self.low_binary, self.high_binary])

    def find_best_state(self, env, data_buffer):
        """
        In this case, already known the best state, so just return it
        """
        this_human_response = env.compute_human_response(
            self.optimal_result.best_robot_state)
        # if args.normalized_human_response, env returns normalized human response, otherwise, return actual human response
        # data_buffer knows if it still needed to be normalized, so just pass it to data_buffer.normalize_human_response

        # MODIFY: Just use human_response right now
        human_response = data_buffer.normalize_human_response(
            this_human_response)
        self.optimal_result.best_human_response = human_response

        if not env.normalized:
            centroid_loader = data_buffer
        else:
            centroid_loader = env

        self.optimal_result.best_satisfy_number, self.optimal_result.best_satisfy_type = CriteriaChecker.satisfy_check(human_response, normalized=env.normalized,
                                                                                                                       eng_centroids=centroid_loader.eng_centroids, vig_centroids=centroid_loader.vig_centroids,
                                                                                                                       eng_normalized_centroids=centroid_loader.eng_normalized_centroids, vig_normalized_centroids=centroid_loader.vig_normalized_centroids)

        self.optimal_result.best_travel_time = env.calculate_traveltime(
            *self.optimal_result.best_robot_state)
        self.optimal_result.best_productivity = env.calculate_productivity(
            self.optimal_result.best_travel_time)

        return self.optimal_result.best_robot_state


class SearchDownStrategy(SimpleStrategy):
    """
    This strategy aims to search down the robot speed and find the first state that have acceptable human response
    For binary variables, the binary cutoff (%) determines when it is set to the not productive side
    """

    def __init__(self, num_combinations=100, binary_cutoff=0.5):
        super().__init__()
        self.strategy_name = 'SearchDown'
        self.num_combinations = num_combinations
        self.binary_cutoff = binary_cutoff
        self.all_combinations = np.array([])
        self.set_all_combinations()

    def set_all_combinations(self):
        move_spd = np.linspace(self.move_spd_high_bnd,
                               self.move_spd_low_bnd, self.num_combinations)
        arm_spd = np.linspace(self.arm_spd_high_bnd,
                              self.arm_spd_low_bnd, self.num_combinations)
        binary_cutoff_num = int(self.num_combinations * self.binary_cutoff)
        binary_array = np.ones(self.num_combinations)
        binary_array[:binary_cutoff_num] = -1
        binary_array_1 = binary_array.copy()  # low bnd good
        binary_array_2 = binary_array.copy()  # low bnd good
        binary_array_3 = binary_array.copy()*-1  # high bnd good
        self.all_combinations = np.array(
            [move_spd, arm_spd, binary_array_1, binary_array_2, binary_array_3]).T

    def find_best_state(self, env, data_buffer):

        if not env.normalized:
            centroid_loader = data_buffer
        else:
            centroid_loader = env

        for robot_state in self.all_combinations:
            this_human_response = env.compute_human_response(robot_state)
            # if args.normalized_human_response, env returns normalized human response, otherwise, return actual human response
            # data_buffer knows if it still needed to be normalized, so just pass it to data_buffer.normalize_human_response

            # MODIFY: Just use human_response but same criteria checking so it doesn't run into errors
            human_response = data_buffer.normalize_human_response(
                this_human_response)

            self.optimal_result.check_and_update(human_response, robot_state, )

            current_satisfy_number, current_satisfy_type = CriteriaChecker.satisfy_check(human_response, normalized=env.normalized,
                                                                                         eng_centroids=centroid_loader.eng_centroids, vig_centroids=centroid_loader.vig_centroids,
                                                                                         eng_normalized_centroids=centroid_loader.eng_normalized_centroids, vig_normalized_centroids=centroid_loader.vig_normalized_centroids)
            # Check if current's satisfy type is larger than optimal result
            if current_satisfy_number > self.optimal_result.best_satisfy_number:
                self.optimal_result.best_satisfy_number = current_satisfy_number
                self.optimal_result.best_satisfy_type = current_satisfy_type
                self.optimal_result.best_robot_state = robot_state
                self.optimal_result.best_human_response = human_response
                self.optimal_result.best_travel_time = env.calculate_traveltime(
                    *self.optimal_result.best_robot_state)
                self.optimal_result.best_productivity = env.calculate_productivity(
                    self.optimal_result.best_travel_time)

                # If all responses all satisfied, we can break
                if current_satisfy_number == 4:
                    break
        return self.optimal_result.best_robot_state
