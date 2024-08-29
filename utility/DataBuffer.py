import numpy as np
from sklearn.cluster import KMeans


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

        # NOTE: Initialize a buffer of buffer stores response satisfy number
        # Stores from number 0 to args.num_responses + 1
        # EG. if there are 4 human respones, then will be [[satisfy 0], [satisfy 1], [satisfy 2], [satisfy 3], [satisfy 4]]
        self.response_satisfy_number_buffers = np.empty(
            (args.num_responses + 1, 0))
        self.response_satisfy_type_buffer = []

        self.is_exploit_buffer = []
        self.length = 0

        self.val_std = 0
        self.val_mean = 0
        self.aro_std = 0
        self.aro_mean = 0

        # Add Parameters for Engagement and Vigilance
        self.eng_std = 0
        self.eng_mean = 0
        self.eng_centroids = np.zeros(3)
        self.eng_normalized_centroids = np.zeros(3)

        self.vig_std = 0
        self.vig_mean = 0
        self.vig_centroids = np.zeros(3)
        self.vig_normalized_centroids = np.zeros(3)

        # This is for the case when we use the prefix eight values
        self.pause_update_normalization_param = False

    def add(self, robot_state, human_response, productivity, response_satify_number_array, response_satisfy_type, is_exploit=True):
        """
        :param robot_state: 5D robot state, first 2 continuous, last 3 discrete
        :param human_response: 2D, valance and arousal, should be raw from env
        :param is_exploit: whether this data point is collected by exploit or explore
        """
        data_point = {'robot_state': robot_state,
                      'human_response': human_response}
        self.buffer.append(data_point)
        self.human_response_buffer.append(human_response)
        self.robot_state_buffer.append(robot_state)
        self.productivity_buffer.append(productivity)

        # Append the column of response satisfy number array
        self.response_satisfy_number_buffers = np.insert(self.response_satisfy_number_buffers, len(
            self.response_satisfy_number_buffers[0]), response_satify_number_array, axis=1)
        
        # Append  the satisfy type
        self.response_satisfy_type_buffer.append(response_satisfy_type)

        self.is_exploit_buffer.append(is_exploit)
        self.length += 1
        # only update the normalization parameters for random sampled data points (i.e., explore)
        if not is_exploit:
            self.update_normalization_parameters()

    def calculate_prefix_eight(self, env):
        prefix_eight_robot_state = [[27.8, 73.2, -1, -1, 1],
                                    [79.8, 73.2, -1, -1, 1],
                                    [143.8, 73.2, -1, -1, 1],
                                    [79.8, 23.8, -1, -1, 1],
                                    [79.8, 109.1, -1, -1, 1],
                                    [79.8, 73.2, 1, -1, 1],
                                    [79.8, 73.2, -1, 1, 1],
                                    [79.8, 73.2, -1, -1, -1]]

        # Compute the human response and add to the data buffer for the prefix values
        for prefix_robot_state in prefix_eight_robot_state:
            prefix_human_response = env.compute_human_response(
                prefix_robot_state)
            self.add(prefix_robot_state, prefix_human_response,
                     np.nan, [np.nan] * (env.num_responses + 1), np.nan, is_exploit=False)

        # Calculate the centroids, mean, and std
        self.update_normalization_parameters()

        # Set the pause_update variable to True so there will be no longer update
        self.pause_update_normalization_param = True

    def sample(self, batch_size):
        # todo: currently random sample, some data might be sampled multiple times or missed
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        robot_state_buffer_np = np.array(self.robot_state_buffer)
        human_response_buffer_np = np.array(self.human_response_buffer)
        return human_response_buffer_np[idx], robot_state_buffer_np[idx]

    # Function that finds the n clusters for Engagement and Vigilance
    # Also find their normalized valeus
    def calculate_clusters(self, human_responses_buffer_np_exploit, n_clusters=3):
        """
        Params: 
        human response in the data buffer only when is_exploit = False

        Modify:
        values of different eng_centroid and vig_centroid
        """

        # Initialize the KMeans from Sklearn
        k_means_cluster = KMeans(n_clusters=n_clusters, random_state=482)

        # Find Centroid values for the engagement
        k_means_cluster.fit(
            (human_responses_buffer_np_exploit[:, 2]).reshape(-1, 1))
        engagement_clusters = k_means_cluster.cluster_centers_
        engagement_clusters = engagement_clusters.flatten()
        sorted_engagement_clusters = np.sort(engagement_clusters)

        # Find the Centroid values for the vigilance
        k_means_cluster.fit(
            (human_responses_buffer_np_exploit[:, 3]).reshape(-1, 1))
        vigilance_clusters = k_means_cluster.cluster_centers_
        vigilance_clusters = vigilance_clusters.flatten()
        sorted_vigilance_clusters = np.sort(vigilance_clusters)

        # Check if the data is already normal
        # If normal: normal centroids is the same as sorted result,
        # If not normal: normal centroid is normalizing the sorted result
        if not self.normalized_human_response:
            self.eng_centroids = sorted_engagement_clusters
            self.eng_normalized_centroids = (
                self.eng_centroids - self.eng_mean) / self.eng_std

            self.vig_centroids = sorted_vigilance_clusters
            self.vig_normalized_centroids = (
                self.vig_centroids - self.vig_mean) / self.vig_std
        else:
            self.eng_normalized_centroids = sorted_engagement_clusters
            self.eng_centroids = (
                self.eng_normalized_centroids * self.eng_std) + self.eng_mean

            self.vig_normalized_centroids = sorted_vigilance_clusters
            self.vig_centroids = (
                self.vig_normalized_centroids * self.vig_std) + self.vig_mean

    def update_normalization_parameters(self):
        """ Calculate the normalization parameters for the human response in the data buffer so far, but only when is_exploit is False
        :return: val_mean, val_std, aro_mean, aro_std
        """

        # Check if paused upstate
        # If not paused, then we update the parameters
        if not self.pause_update_normalization_param:
            human_response_buffer_np = np.array(self.human_response_buffer)
            human_response_buffer_np_exploit = human_response_buffer_np[np.array(
                self.is_exploit_buffer) == False]
            self.val_mean = np.mean(human_response_buffer_np_exploit[:, 0])
            self.val_std = np.std(human_response_buffer_np_exploit[:, 0])
            self.aro_mean = np.mean(human_response_buffer_np_exploit[:, 1])
            self.aro_std = np.std(human_response_buffer_np_exploit[:, 1])

            # Update Engagement and Vigilance Parameters as well
            self.eng_mean = np.mean(human_response_buffer_np_exploit[:, 2])
            self.eng_std = np.std(human_response_buffer_np_exploit[:, 2])
            self.vig_mean = np.mean(human_response_buffer_np_exploit[:, 3])
            self.vig_std = np.std(human_response_buffer_np_exploit[:, 3])

            # Find the three centroids as well
            if (len(human_response_buffer_np_exploit) >= 3):
                self.calculate_clusters(human_response_buffer_np_exploit)

    def get_normalization_parameters(self):
        param = [self.val_mean, self.val_std, self.aro_mean, self.aro_std,
                    self.eng_mean, self.eng_std, self.vig_mean, self.vig_std,
                    self.eng_centroids, self.eng_normalized_centroids,
                    self.vig_centroids, self.vig_normalized_centroids]
        return param

    def normalize_human_response(self, human_response):
        """ Normalize human response using the normalization parameters in the data buffer
        :param human_response: 2D, valance and arousal, engagement and vigilance
        :return: normalized human response
        """
        if self.normalized_human_response:
            return human_response
        else:
            human_response = np.array(human_response)
            human_response[0] = (human_response[0] -
                                 self.val_mean) / self.val_std
            human_response[1] = (human_response[1] -
                                 self.aro_mean) / self.aro_std

            # Update for Engagement and Vigilance
            human_response[2] = (human_response[2] -
                                 self.eng_mean) / self.eng_std
            human_response[3] = (human_response[3] -
                                 self.vig_mean) / self.vig_std
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
            human_response_batch[:, 0] = (
                human_response_batch[:, 0] - self.val_mean) / self.val_std
            human_response_batch[:, 1] = (
                human_response_batch[:, 1] - self.aro_mean) / self.aro_std

            # Update for Engagement and Vigilance
            human_response_batch[:, 2] = (
                human_response_batch[:, 2] - self.eng_mean) / self.eng_std
            human_response_batch[:, 3] = (
                human_response_batch[:, 3] - self.vig_mean) / self.vig_std

            return human_response_batch

    def __len__(self):
        return self.length
