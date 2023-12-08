import torch
import torch.optim as optim
import numpy as np
import argparse
import wandb

from rand_param_envs.gym.envs.HRC.kuka_human_response import KukaHumanResponse, KukaHumanResponse_Rand
from HumanResponseModel import HumanResponseModel
from DataBuffer import DataBuffer
from utility import *

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



