import numpy as np


class ResultLogger:
    def __init__(self, subject_id, args):
        self.subject_id = subject_id
        self.args = args
        self.episode_num = args.episode_num
        self.episode_linspace = np.linspace(0, args.episode_num-1, args.episode_num)
        self.is_exploit = np.zeros(args.episode_num)
        self.robot_state = []
        self.human_response = []
        self.good_human_response = np.zeros(args.episode_num)
        self.good_human_response_percentage = np.zeros(args.episode_num)
        self.arousal_mean_var = []
        self.valance_mean_var = []
        self.productivity = np.zeros(args.episode_num)
        self.productivity_max = None
        self.productivity_percentage = np.zeros(args.episode_num)
        self.checkpoint_file = None

        self.exploit_total_num = 0
        self.exploit_success_num = 0

    def add_exploit(self, i, robot_state, human_response, productivity):
        self.is_exploit[i] = 1.0
        self.exploit_total_num += 1
        self.robot_state.append(robot_state)
        self.human_response.append(human_response)
        self.productivity[i] = productivity

