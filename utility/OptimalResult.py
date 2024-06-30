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
        Method that will check if each human response satisfies the critiera first
        Then check with the best result so far, and update best result if better
        """

        current_satisfy_type = ""
        current_satisfy_number = 0

        # Check for each response
        for i in range(len(human_response)):
            is_satisfy_type = False
            response_value = human_response[i]
            response_type = self.index_map[i]
            check_method = self.methods_map[response_type]

            # Check if it is engagement
            if response_type == "ENG":
                is_satisfy_type = check_method(response_value, normalized=self.normalized,
                                               eng_centroids=eng_centroids, eng_normalized_centroids=eng_normalized_centroids)
            elif response_type == "VIG":
                is_satisfy_type = check_method(response_value, normalized=self.normalized,
                                               vig_centroids=vig_centroids, vig_normalized_centroids=vig_normalized_centroids)
            else:
                is_satisfy_type = check_method(response_value)

            # Check if this response satisfies the criteria
            if is_satisfy_type:
                current_satisfy_number += 1
                current_satisfy_type += response_type
                # Use '-' to connect
                current_satisfy_type += "-" if i < len(
                    human_response) - 1 else ""

        # Now compare with current best
        # Condition: (1) If current satisfy number is more than best number or
        #            (2) If number is same, productivity is better than best
        if (current_satisfy_number > self.best_satisfy_number) or ((self.best_satisfy_number == current_satisfy_number) and
                                                                   (productivity > self.best_productivity)):
            self.best_productivity = productivity
            self.best_human_response = human_response
            self.best_robot_state = robot_state
            self.have_result = True
            self.best_satisfy_type = current_satisfy_type
