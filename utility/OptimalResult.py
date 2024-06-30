from utility.CriteriaChecker import CriteriaChecker


class OptimalResult:
    """
    Object instance stores the result of the best result/productivity
    Criteria:
        -- "Best" by amount of responses that are satisfied (e.g. 4 -> 3 -> 2 -> 1)

    """

    def __init__(self, normalized=True) -> None:
        self.best_satisfy_number = 0
        self.best_productivity = 0
        self.best_robot_state = []
        self.best_human_response = []
        self.have_result = None
        self.best_satisfy_type = ""
        self.normalized = normalized

        # Index Map for Human Response
        self.index_map = {
            0: "VAL",
            1: "ARO",
            2: "ENG",
            3: "VIG"
        }

        # Criteria Check Method Map for Human Response
        self.methods_map = {
            "VAL": CriteriaChecker.satisfy_valence,
            "ARO": CriteriaChecker.satisfy_arousal,
            "ENG": CriteriaChecker.satisfy_engagement,
            "VIG": CriteriaChecker.satisfy_vigilance
        }

    def check_for_update(self, human_response, robot_state, productivity,
                         eng_centroids, vig_centroids, eng_normalized_centroids, vig_normalized_centroids) -> None:
        """
        Method that checks how many responses are satisfied
        Then compare with the best result so far, and update if better
        """

        current_satisfy_number, current_satisfy_type = CriteriaChecker.satisfy_check(human_response, normalized=self.normalized,
                                                                                     eng_centroids=eng_centroids, eng_normalized_centroids=eng_normalized_centroids,
                                                                                     vig_centroids=vig_centroids, vig_normalized_centroids=vig_normalized_centroids)

        # Now compare with best so far
        # Condition: (1) If current satisfy number is more than best number or
        #            (2) If number is same, productivity is better than best
        if (current_satisfy_number > self.best_satisfy_number) or ((self.best_satisfy_number == current_satisfy_number) and
                                                                   (productivity > self.best_productivity)):
            self.best_productivity = productivity
            self.best_human_response = human_response
            self.best_robot_state = robot_state
            self.have_result = True
            self.best_satisfy_type = current_satisfy_type
