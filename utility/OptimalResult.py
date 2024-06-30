

class OptimalResult:
    """
    Object instance stores the result of the best result/productivity
    Criteria:
        -- "Best" by amount of responses that are satisfied (e.g. 4 -> 3 -> 2 -> 1)
        
    """
    def __init__(self) -> None:
        self.best_number_of_satisifed = 0
        self.best_productivity = 0
        self.best_robot_state = []
        self.best_human_response = []
        self.have_result = None
        self.best_satisfied_responses = ""