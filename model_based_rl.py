import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from rand_param_envs.gym.envs.HRC.kuka_human_response import KukaHumanResponse, KukaHumanResponse_Rand
import argparse

class HumanResponseModel(torch.nn.Module):
    """
    This is a nural network model to predict human response (valance and arousal) based on the HRC robot state
    """
    def __init__(self, input_size=5, hidden_size=64, output_size=2):
        super(HumanResponseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the layers
        self.linear1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


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
        return robot_state_buffer_np[idx], human_response_buffer_np[idx]

    def __len__(self):
        return self.length


def grid_search(env, model):
    continuous_bin = np.arange(0, 1.01, 0.01)
    binary_bin = [0, 1]
    bin_map = [(0, 0, a, b, x, y, z) for a in continuous_bin for b in continuous_bin for x in binary_bin for y in
               binary_bin for z in binary_bin]
    move_spd_low_bnd, move_spd_high_bnd = [27.8, 143.8]
    arm_spd_low_bnd, arm_spd_high_bnd = [23.8, 109.1]
    best_reward = -1000
    best_state = False
    for i in range(len(bin_map)):
        state = bin_map[i]
        state = np.array(state)
        state[0 + 2] = state[0 + 2] * (move_spd_high_bnd - move_spd_low_bnd) + move_spd_low_bnd
        state[1 + 2] = state[1 + 2] * (arm_spd_high_bnd - arm_spd_low_bnd) + arm_spd_low_bnd
        state[2 + 2] = state[2 + 2] * 2 - 1
        state[3 + 2] = state[3 + 2] * 2 - 1
        state[4 + 2] = state[4 + 2] * 2 - 1
        traveltime = env.calculate_traveltime(state[2], state[3], state[4], state[5], state[6])
        productivity = env.calculate_productivity(traveltime)
        robot_state_tensor = torch.from_numpy(state[2:]).float()  # Convert numpy array to PyTorch tensor
        a, v = model(robot_state_tensor)
        if a > 0 and v > 0:
            if productivity > best_reward:
                best_reward = productivity
                best_state = state
    return best_state, best_reward

def generate_random_data_point():
    kuka = KukaHumanResponse_Rand()
    data_point = kuka.get_data_point()
    return data_point

def train_model(model, data_buffer, optimizer, loss_function, batch_size=64):
    # Sample data points from the buffer
    robot_states, human_responses = data_buffer.sample(batch_size)

    # Convert numpy arrays to PyTorch tensors and move to device
    robot_states = torch.from_numpy(robot_states).float().to(device)
    human_responses = torch.from_numpy(human_responses).float().to(device)

    # Forward pass
    outputs = model(robot_states)

    # Compute loss
    loss = loss_function(outputs, human_responses)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def parse_args():
    parser = argparse.ArgumentParser(description='Human Response Model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    env = KukaHumanResponse_Rand()  # Create the environment
    env.reset()
    for subject_id in range(2):  #18
        print('------------------ Subject', subject_id, '------------------')
        env.reset_task(subject_id)

        # Create the model
        model = HumanResponseModel().to(device)

        # Create the data buffer
        data_buffer = DataBuffer()
        # Define the optimizer and the loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_function = torch.nn.MSELoss()

        exploration_rate = 1.0
        exploration_decay_rate = 0.99
        batch_size = 64


        for _ in range(batch_size*10):  # fill the buffer with random data points
            data_point = env._reset()
            human_response = data_point[:2]
            robot_state = data_point[2:]
            data_buffer.add(robot_state, human_response)

        for i in range(1000):  # run 1000 episodes of HRC interaction
            if np.random.random() > exploration_rate:
                robot_state, reward = grid_search(env, model)
                if robot_state:
                    human_response = env.compute_human_response(robot_state)
                    with np.printoptions(precision=2):
                        print(f"{i}th episode, productivity: {reward:.2f}, human response: {human_response}, robot state: {robot_state}")
                else:  # random point since grid search got no results
                    data_point = env._reset()
                    human_response = data_point[:2]
                    robot_state = data_point[2:]
                    print(f"{i}th episode, random explore because no grid result")
            else:  # random explore
                data_point = env._reset()
                human_response = data_point[:2]
                robot_state = data_point[2:]

                print(f"{i}th episode, random explore")


            # store in buffer
            data_buffer.add(robot_state, human_response)

            for training_step in range(2):
                print(f"Training step {training_step}")
                train_model(model, data_buffer, optimizer, loss_function, batch_size)

            # update epsilon
            exploration_rate = exploration_rate * exploration_decay_rate



