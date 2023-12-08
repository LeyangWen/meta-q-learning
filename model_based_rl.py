import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from rand_param_envs.gym.envs.HRC.kuka_human_response import KukaHumanResponse, KukaHumanResponse_Rand
import argparse
import wandb


class HumanResponseModel(torch.nn.Module):
    """
    This is a nural network model to predict human response (valance and arousal) based on the HRC robot state
    """
    def __init__(self, input_size=5, hidden_size=32, output_size=2, dropout_rate=0):  # dropout too aggressive for online learning
        super(HumanResponseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the layers
        self.linear1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.linear3 = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = torch.nn.functional.tanh(x) * 10
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
        return human_response_buffer_np[idx], robot_state_buffer_np[idx]

    def __len__(self):
        return self.length


def grid_search(args, env, model, GT=False):
    step = 1/args.grid_search_num
    continuous_bin = np.arange(0, 1+step, step)
    binary_bin = [-1, 1]
    bin_map = [(0, 0, a, b, x, y, z) for a in continuous_bin for b in continuous_bin for x in binary_bin for y in
               binary_bin for z in binary_bin]
    move_spd_low_bnd, move_spd_high_bnd = [27.8, 143.8]  # todo: get from env
    arm_spd_low_bnd, arm_spd_high_bnd = [23.8, 109.1]

    full_states = np.array(bin_map)  # full state needed only for env.compute_human_response
    full_states[:, 2] = full_states[:, 2] * (move_spd_high_bnd - move_spd_low_bnd) + move_spd_low_bnd
    full_states[:, 3] = full_states[:, 3] * (arm_spd_high_bnd - arm_spd_low_bnd) + arm_spd_low_bnd

    if not GT:
        robot_states = torch.from_numpy(full_states[:, 2:]).float().to(args.device)  # Convert numpy array to PyTorch tensor
        arousals, valances = model(robot_states).detach().cpu().numpy().T

    best_reward = -1000
    best_robot_state = []
    have_result = False
    for i in range(len(bin_map)):
        this_state = full_states[i]
        travelTime = env.calculate_traveltime(this_state[2], this_state[3], this_state[4], this_state[5], this_state[6])
        productivity = env.calculate_productivity(travelTime)
        if GT:
            arousal, valance = env.compute_human_response(this_state)
        else:
            arousal = arousals[i]
            valance = valances[i]
        if arousal > 0 and valance > 0:
            if productivity > best_reward:
                best_reward = productivity
                best_robot_state = this_state[2:]
                have_result = True
    return best_robot_state, best_reward, have_result


def train_step(args, model, data_buffer, optimizer, loss_function, batch_size):
    # Sample data points from the buffer
    human_responses, robot_states = data_buffer.sample(batch_size)
    # todo: regularization here

    # Convert numpy arrays to PyTorch tensors and move to args.device
    robot_states = torch.from_numpy(robot_states).float().to(args.device)
    human_responses = torch.from_numpy(human_responses).float().to(args.device)

    # Forward pass
    outputs = model(robot_states)

    # Compute loss
    loss = loss_function(outputs, human_responses)
    # wandb.log({f"loss_sub{args.sub_id}": loss.item()})

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def choose_device(args):
    if args.device == 'cuda':
        print('Trying GPU...')
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cpu':
        print('Using CPU...')
        args.device = torch.device("cpu")
    else:
        print('Unknown device, using CPU...')
        args.device = torch.device("cpu")
    print('Using args.device:', args.device)
    return args


def random_explore(args, env):
    data_point = env.reset()
    human_response = data_point[:2]
    robot_state = data_point[2:]
    return human_response, robot_state

def parse_args():
    parser = argparse.ArgumentParser(description='Human Response Model')
    parser.add_argument('--device', default='cuda', help='device, cpu or cuda')
    parser.add_argument('--grid_search_num', default=200, type=int, help='number of grid search, positive integer')
    parser.add_argument('--random_explore_num', default=32, type=int, help='number of random explore, positive integer')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size for training, positive integer')
    parser.add_argument('--train_step_per_episode', default=10, type=int, help='number of training steps per episode, positive integer')
    parser.add_argument('--episode_num', default=64, type=int, help='batch size for training, positive integer')
    parser.add_argument('--exploration_rate', default=0.5, type=float, help='exploration rate, float between 0 and 1')
    parser.add_argument('--exploration_decay_rate', default=0.99, type=float, help='exploration decay rate, float between 0 and 1')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate, float between 0 and 1')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay, float between 0 and 1')
    # parser.add_argument('--pretrained_model', default=None, help='path to pretrained model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args = choose_device(args)

    wandb.init(project="HRC_model_based_rl", name="Test2-32rand-512total-morePlot", config=vars(args))   # Initialize a new run
    wandb.define_metric("train/episode")  # define our custom x axis metric
    wandb.define_metric("*", step_metric="train/episode")  # set all other train/ metrics to use this step

    env = KukaHumanResponse_Rand()  # Create the environment
    env.reset()
    for subject_id in range(18):  #18
        print('\n\n------------------------------------ Subject', subject_id, '------------------------------------')
        env.reset_task(subject_id)
        args.sub_id = subject_id
        GT_robot_state, GT_best_reward, GT_have_result = grid_search(args, env, None, GT=True)
        human_response = env.compute_human_response(GT_robot_state)
        if GT_have_result:
            print(f"productivity: {GT_best_reward:.2f}, human response: {human_response}, robot state: {GT_robot_state}")
        else:
            print("No GT result")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print()
        # Create the model
        model = HumanResponseModel().to(args.device)

        # Create the data buffer
        data_buffer = DataBuffer()
        # Define the optimizer and the loss function
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_function = torch.nn.MSELoss()

        exploration_rate = args.exploration_rate

        for _ in range(args.random_explore_num):  # fill the buffer with random data points
            print(f"Fill the buffer with random data points {_}/{args.random_explore_num}...", end="\r")
            data_point = env.reset()
            human_response = data_point[:2]
            robot_state = data_point[2:]
            data_buffer.add(robot_state, human_response)
        print(f"Buffer filled with {args.random_explore_num} random data points")

        exploit_total_num = 0
        exploit_success_num = 0
        for i in range(args.episode_num):  # run 1000 episodes of HRC interaction
            log_dict = {}
            log_dict["train/episode"] = i  # our custom x axis metric
            if np.random.random() > exploration_rate:
                robot_state, reward, have_result = grid_search(args, env, model)
                if have_result:
                    exploit_total_num += 1
                    human_response = env.compute_human_response(robot_state)
                    good_human_response = True if (human_response[0] > 0 and human_response[1] > 0) else False
                    with np.printoptions(precision=2):
                        log_dict[f"train/Subject({env.sub_id})/Good human response"] = float(good_human_response*1.0)
                        if good_human_response:
                            exploit_success_num += 1
                            print(f"{i}, good HR: {good_human_response}, productivity: {reward:.2f}, HR: {human_response}, robot state: {robot_state}")
                            log_dict[f"train/Subject({env.sub_id})/Productivity_max({GT_best_reward:.0f}bk\hr)"] = reward
                            log_dict[
                                f"train/Subject({env.sub_id})/Productivity_max({GT_best_reward:.0f}bk\hr)_all"] = reward
                        else:
                            log_dict[f"train/Subject({env.sub_id})/Productivity_max({GT_best_reward:.0f}bk\hr)_all"] = 0
                        log_dict[f"train/Subject({env.sub_id})/Good human response %)"] = float(
                            exploit_success_num / exploit_total_num)
                else:  # random point since grid search got no results with positive valance and arousal
                    human_response, robot_state = random_explore(args, env)
                    # print(f"{i}th episode, random explore because no grid result")
            else:  # random explore
                human_response, robot_state = random_explore(args, env)
                # print(f"{i}th episode, random explore, exploration rate: {exploration_rate*100:.1f}%")
            wandb.log(log_dict)

            # store in buffer
            data_buffer.add(robot_state, human_response)
            model.train()
            for training_step in range(args.train_step_per_episode):
                print(f"Training step {training_step}...", end="\r")
                train_step(args, model, data_buffer, optimizer, loss_function, args.train_batch_size)

            # update epsilon
            exploration_rate = exploration_rate * args.exploration_decay_rate



