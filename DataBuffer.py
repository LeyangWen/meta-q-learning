import numpy as np


class DataBuffer:
    """
    This is a data buffer to store the data point for training
    """
    def __init__(self):
        self.buffer = []
        self.human_response_buffer = []
        self.robot_state_buffer = []
        self.length = 0

    def add(self, robot_state, human_response):
        data_point = {'robot_state': robot_state, 'human_response': human_response}
        self.buffer.append(data_point)
        self.human_response_buffer.append(human_response)
        self.robot_state_buffer.append(robot_state)
        self.length += 1

    def sample(self, batch_size):
        # todo: currently random sample, some data might be sampled multiple times or missed
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        robot_state_buffer_np = np.array(self.robot_state_buffer)
        human_response_buffer_np = np.array(self.human_response_buffer)
        return human_response_buffer_np[idx], robot_state_buffer_np[idx]

    def __len__(self):
        return self.length
