import torch
import torch.optim as optim
import numpy as np
import argparse
import wandb
import time
import pickle
import os

from rand_param_envs.gym.envs.HRC.kuka_human_response import KukaHumanResponse_Rand
from HumanResponseModel import HumanResponseModel
from utility.DataBuffer import DataBuffer
from utility.utility import *


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

    best_reward = 0
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


def random_explore(args, env):
    data_point = env.reset()
    human_response = data_point[:2]
    robot_state = data_point[2:]
    # travelTime = env.calculate_traveltime(data_point[2], data_point[3], data_point[4], data_point[5], data_point[6])
    # productivity = env.calculate_productivity(travelTime)
    return human_response, robot_state


def parse_args():
    parser = argparse.ArgumentParser(description='Human Response Model')
    parser.add_argument('--device', default='cuda', help='device, cpu or cuda')
    parser.add_argument('--grid_search_num', default=100, type=int, help='number of grid search, positive integer')
    parser.add_argument('--random_explore_num', default=32, type=int, help='number of random explore, positive integer')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size for training, positive integer')
    parser.add_argument('--train_step_per_episode', default=1000, type=int, help='number of training steps per episode, positive integer')
    parser.add_argument('--episode_num', default=100, type=int, help='batch size for training, positive integer')
    parser.add_argument('--exploration_rate', default=0.5, type=float, help='exploration rate, float between 0 and 1')
    parser.add_argument('--exploration_decay_rate', default=0.99, type=float, help='exploration decay rate, float between 0 and 1')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate, float between 0 and 1')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay, float between 0 and 1')
    # parser.add_argument('--pretrained_model', default=None, help='path to pretrained model')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--wandb_project', default='HRC_model_based_rl', help='wandb project name')
    parser.add_argument('--wandb_name', default='Debug-32rand-512total-morePlot', help='wandb run name')
    parser.add_argument('--result_look_back_episode', default=20, type=int, help='number of episodes to look back for best result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args = choose_device(args)

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))  # Initialize a new run
    wandb.define_metric("train/episode")  # define our custom x axis metric
    wandb.define_metric("train/*", step_metric="train/episode")  # set all other train/ metrics to use this step

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

        model = HumanResponseModel().to(args.device)   # Create the model
        data_buffer = DataBuffer()  # Create the data buffer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # Define the optimizer and the loss function
        loss_function = torch.nn.MSELoss()
        exploration_rate = args.exploration_rate

        for _ in range(args.random_explore_num):  # fill the buffer with random data points
            print(f"Fill the buffer with random data points {_}/{args.random_explore_num}...", end="\r")
            data_point = env.reset()
            human_response = data_point[:2]
            robot_state = data_point[2:]
            data_buffer.add(robot_state, human_response, np.nan,  is_exploit=False)
        print(f"Buffer filled with {args.random_explore_num} random data points")

        exploit_success_num = 0
        exploit_total_num = 0
        reward = np.nan
        good_human_response = np.nan
        for i in range(args.episode_num):  # run n episodes of HRC interaction
            is_exploit = np.random.random() > exploration_rate
            if is_exploit:  # exploit
                exploit_total_num += 1
                robot_state, reward, have_result = grid_search(args, env, model)
                if have_result:
                    human_response = env.compute_human_response(robot_state)
                    good_human_response = True if (human_response[0] > 0 and human_response[1] > 0) else False
                    with np.printoptions(precision=2):
                        if good_human_response:
                            exploit_success_num += 1
                            print(f"{i}, good HR: {good_human_response}, productivity: {reward:.2f}, HR: {human_response}, robot state: {robot_state}")
                else:  # random point since grid search got no results with positive valance and arousal
                    human_response, robot_state = random_explore(args, env)
            else:  # random explore
                human_response, robot_state = random_explore(args, env)

            #### log ####
            log_dict = {}
            log_dict["train/episode"] = i  # our custom x axis metric
            log_dict[f"train/Subject({env.sub_id})/Productivity)"] = reward  # can not have "." in name or wandb plot have wrong x axis
            log_dict[f"train/Subject({env.sub_id})/Good human response %)"] = exploit_success_num / (exploit_total_num+1e-6)
            log_dict[f"train/Subject({env.sub_id})/Productivity %"] = reward / GT_best_reward
            log_dict[f"train/Subject({env.sub_id})/Robot movement speed"] = robot_state[0]
            log_dict[f"train/Subject({env.sub_id})/Robot movement speed %"] = robot_state[0] / GT_robot_state[0]
            log_dict[f"train/Subject({env.sub_id})/Robot arm speed"] = robot_state[1]
            log_dict[f"train/Subject({env.sub_id})/Is exploit"] = float(is_exploit*1.0)
            log_dict[f"train/Subject({env.sub_id})/Good human response"] = float(good_human_response*1.0)

            wandb.log(log_dict)

            # store in buffer
            data_buffer.add(robot_state, human_response, reward, is_exploit)
            model.train()
            for training_step in range(args.train_step_per_episode):
                print(f"Training step {training_step}...", end="\r")
                train_step(args, model, data_buffer, optimizer, loss_function, args.train_batch_size)

            # update epsilon
            exploration_rate = exploration_rate * args.exploration_decay_rate

        # look back few episodes to find best result
        converge_result = {"robot_state": [], "human_response": [], "productivity": []}
        best_productivity = 0
        for look_back in range(args.result_look_back_episode):
            if data_buffer.is_exploit_buffer[-look_back]:
                if data_buffer.productivity_buffer[-look_back] > best_productivity:
                    best_productivity = data_buffer.productivity_buffer[-look_back]
                    converge_result["robot_state"] = data_buffer.robot_state_buffer[-look_back]
                    converge_result["human_response"] = data_buffer.human_response_buffer[-look_back]
                    converge_result["productivity"] = data_buffer.productivity_buffer[-look_back]
        if best_productivity == 0:
            raise Exception(f"No best result found in the looking back {args.result_look_back_episode} episode in buffer")
        #### log ####
        wandb_GT_table = wandb.Table(
            columns=[" ", "Productivity", "Valance", "Arousal", "Robot Movement Speed", "Arm Swing Speed",
                     "Proximity", "Autonomy", "Collab"])
        wandb_GT_table.add_data("GT", GT_best_reward, *human_response, *GT_robot_state)
        wandb_GT_table.add_data("Results", converge_result["productivity"], *converge_result["human_response"], *converge_result["robot_state"])
        wandb.log({f"Train/Subject({subject_id})/Results": wandb_GT_table})

        # Save the model and result
        checkpoint_file = f"{args.checkpoint_dir}/{args.wandb_name}/subject_{args.sub_id}.pt"
        if not os.path.exists(os.path.dirname(checkpoint_file)):
            os.makedirs(os.path.dirname(checkpoint_file))
        torch.save(model.state_dict(), checkpoint_file)
