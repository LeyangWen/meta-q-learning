import numpy as np


class DataBuffer:
    """
    This is a data buffer to store the data point for training
    """
    def __init__(self, args):
        self.normalized_human_response = args.normalized_human_response
        self.buffer = []
        self.human_response_buffer = []  # human normalized human response
        self.robot_state_buffer = []
        self.productivity_buffer = []
        self.good_human_response_buffer = []
        self.is_exploit_buffer = []
        self.length = 0

        self.val_std = 0
        self.val_mean = 0
        self.aro_std = 0
        self.aro_mean = 0

    def add(self, robot_state, human_response, productivity, good_human_response, is_exploit=True):
        """
        :param robot_state: 5D robot state, first 2 continuous, last 3 discrete
        :param human_response: 2D, valance and arousal, should be raw from env
        :param is_exploit: whether this data point is collected by exploit or explore
        """
        data_point = {'robot_state': robot_state, 'human_response': human_response}
        self.buffer.append(data_point)
        self.human_response_buffer.append(human_response)
        self.robot_state_buffer.append(robot_state)
        self.productivity_buffer.append(productivity)
        self.good_human_response_buffer.append(good_human_response)
        self.is_exploit_buffer.append(is_exploit)
        self.length += 1
        if not is_exploit:  # only update the normalization parameters for random sampled data points (i.e., explore)
            self.update_normalization_parameters()

    def sample(self, batch_size):
        # todo: currently random sample, some data might be sampled multiple times or missed
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        robot_state_buffer_np = np.array(self.robot_state_buffer)
        human_response_buffer_np = np.array(self.human_response_buffer)
        return human_response_buffer_np[idx], robot_state_buffer_np[idx]

    def update_normalization_parameters(self):
        """ Calculate the normalization parameters for the human response in the data buffer so far, but only when is_exploit is False
        :return: val_mean, val_std, aro_mean, aro_std
        """
        human_response_buffer_np = np.array(self.human_response_buffer)
        human_response_buffer_np_exploit = human_response_buffer_np[np.array(self.is_exploit_buffer) == False]
        self.val_mean = np.mean(human_response_buffer_np_exploit[:, 0])
        self.val_std = np.std(human_response_buffer_np_exploit[:, 0])
        self.aro_mean = np.mean(human_response_buffer_np_exploit[:, 1])
        self.aro_std = np.std(human_response_buffer_np_exploit[:, 1])

    def normalize_human_response(self, human_response):
        """ Normalize human response using the normalization parameters in the data buffer
        :param human_response: 2D, valance and arousal
        :return: normalized human response
        """
        if self.normalized_human_response:
            return human_response
        else:
            human_response = np.array(human_response)
            human_response[0] = (human_response[0] - self.val_mean) / self.val_std
            human_response[1] = (human_response[1] - self.aro_mean) / self.aro_std
            return human_response

    def normalize_human_response_batch(self, human_response_batch):
        """ Normalize human response using the normalization parameters in the data buffer
        :param human_response_batch: (batch_size, 2), valance and arousal
        :return: normalized human response
        """
        if self.normalized_human_response:
            return human_response_batch
        else:
            human_response_batch = np.array(human_response_batch)
            human_response_batch[:, 0] = (human_response_batch[:, 0] - self.val_mean) / self.val_std
            human_response_batch[:, 1] = (human_response_batch[:, 1] - self.aro_mean) / self.aro_std
            return human_response_batch

    def __len__(self):
        return self.length
