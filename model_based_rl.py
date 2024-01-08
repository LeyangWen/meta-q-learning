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
from utility.SimpleStrategy import MaxProductivityStrategy, SearchDownStrategy

#todo: meta learning - change to one subject only, write code to determine convergence speed, write meta-eval loops
def train_step(args, model, data_buffer, optimizer, loss_function, batch_size):
    # Sample data points from the buffer
    human_responses, robot_states = data_buffer.sample(batch_size)
    if not args.normalized_human_response:  # env returns actual human response not normalized, normalize here using cumulative mean and std from explore data points
        human_responses = data_buffer.normalize_human_response_batch(human_responses)

    # Convert numpy arrays to PyTorch tensors and move to args.device
    robot_states = torch.from_numpy(robot_states).float().to(args.device)
    human_responses = torch.from_numpy(human_responses).float().to(args.device)

    # Forward pass
    outputs = model(robot_states)

    # Compute loss
    loss = loss_function(outputs, human_responses)
    wandb.log({f"train/human_response_loss": loss.item()})
    # wandb.log({f"epoch_human_response_loss": loss.item()})

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def grid_search(args, env, model=None, data_buffer=None, GT=False):
    """ Grid search for the best robot state with max productivity and positive valance and arousal
    :param args: args
    :param env: env
    :param model: model
    :param data_buffer: data buffer
    :param GT: whether to use ground truth model
    :return: best_robot_state, best_reward, have_result
    """
    # todo finish data_buffer part
    if GT:
        search_num = args.gt_grid_search_num
    else:
        if args.add_noise_during_grid_search:
            noise = np.random.randint(-args.add_noise_during_grid_search, args.add_noise_during_grid_search)
        else:
            noise = 0
        search_num = args.grid_search_num + noise
    continuous_bin = np.linspace(0, 1, search_num)
    binary_bin = [env.low_binary, env.high_binary]  # [-1, 1]
    bin_map = [(0, 0, a, b, x, y, z) for a in continuous_bin for b in continuous_bin for x in binary_bin for y in
               binary_bin for z in binary_bin]
    move_spd_low_bnd, move_spd_high_bnd = [env.move_spd_low_bnd, env.move_spd_high_bnd]  # [27.8, 143.8]
    arm_spd_low_bnd, arm_spd_high_bnd = [env.arm_spd_low_bnd, env.arm_spd_high_bnd]  # [23.8, 109.1]

    full_states = np.array(bin_map)  # full state needed only for env.compute_human_response
    full_states[:, 2] = full_states[:, 2] * (move_spd_high_bnd - move_spd_low_bnd) + move_spd_low_bnd
    full_states[:, 3] = full_states[:, 3] * (arm_spd_high_bnd - arm_spd_low_bnd) + arm_spd_low_bnd

    if not GT:
        robot_states = torch.from_numpy(full_states[:, 2:]).float().to(args.device)  # Convert numpy array to PyTorch tensor
        valances, arousals = model(robot_states).detach().cpu().numpy().T  # output of the model is already normalized

    best_reward = 0
    best_robot_state = []
    best_human_response = []
    have_result = False
    for i in range(len(bin_map)):
        this_state = full_states[i]
        travelTime = env.calculate_traveltime(this_state[2], this_state[3], this_state[4], this_state[5], this_state[6])
        productivity = env.calculate_productivity(travelTime)
        if GT:
            this_human_response = env.compute_human_response(this_state)
            # if args.normalized_human_response, env returns normalized human response, otherwise, return actual human response
            # data_buffer knows if it still needed to be normalized, so just pass it to data_buffer.normalize_human_response
            valance, arousal = data_buffer.normalize_human_response(this_human_response)

        else:
            valance = valances[i]
            arousal = arousals[i]
        if arousal > 0 and valance > 0:
            if productivity > best_reward:
                best_reward = productivity
                best_robot_state = this_state[2:]
                best_human_response = [valance, arousal]
                have_result = True
    return best_robot_state, best_reward, best_human_response, have_result


def random_explore(args, env):
    data_point = env.reset()
    human_response = data_point[:2]
    robot_state = data_point[2:]
    # travelTime = env.calculate_traveltime(data_point[2], data_point[3], data_point[4], data_point[5], data_point[6])
    # productivity = env.calculate_productivity(travelTime)
    return human_response, robot_state


def look_back_in_buffer(data_buffer, look_back_episode):
    """ Look back in the data buffer a few episodes to find the best result
    :param data_buffer: data buffer
    :param look_back_episode: number of episodes to look back
    :return: best_productivity, converge_result, found_result
    """
    converge_result = {"robot_state": data_buffer.robot_state_buffer[-1],
                       "human_response": data_buffer.human_response_buffer[-1],
                       "human_response_normalized": data_buffer.normalize_human_response(data_buffer.human_response_buffer[-1]),
                       "productivity": data_buffer.productivity_buffer[-1]}
    best_productivity = 0
    found_result = False
    for look_back in range(look_back_episode):
        if data_buffer.is_exploit_buffer[-look_back]:
            if data_buffer.good_human_response_buffer[-look_back]:
                if data_buffer.productivity_buffer[-look_back] > best_productivity:
                    best_productivity = data_buffer.productivity_buffer[-look_back]
                    converge_result["robot_state"] = data_buffer.robot_state_buffer[-look_back]
                    converge_result["human_response"] = data_buffer.human_response_buffer[-look_back]
                    converge_result["human_response_normalized"] = data_buffer.normalize_human_response(
                        data_buffer.human_response_buffer[-look_back])
                    converge_result["productivity"] = data_buffer.productivity_buffer[-look_back]
                    found_result = True
    return converge_result, found_result


def parse_args():
    parser = argparse.ArgumentParser(description='Human Response Model')
    parser.add_argument('--device', default='cuda', help='device, cpu or cuda')
    # Training parameters
    parser.add_argument('--grid_search_num', default=100, type=int, help='number of grid search, positive integer')
    parser.add_argument('--gt_grid_search_num', default=500, type=int, help='number of grid search for GT, positive integer')
    parser.add_argument('--random_explore_num', default=32, type=int, help='number of random explore, positive integer')
    parser.add_argument('--train_batch_size', default=32, type=int, help='batch size for training, positive integer')
    parser.add_argument('--train_step_per_episode', default=1024, type=int, help='number of training steps per episode, positive integer')
    parser.add_argument('--episode_num', default=512, type=int, help='batch size for training, positive integer')
    parser.add_argument('--exploration_rate', default=0.5, type=float, help='exploration rate, float between 0 and 1')
    parser.add_argument('--exploration_decay_rate', default=0.99, type=float, help='exploration decay rate, float between 0 and 1')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate, float between 0 and 1')
    parser.add_argument('--weight_decay', default=0.001, type=float, help='weight decay, float between 0 and 1')
    # parser.add_argument('--pretrained_model', default=None, help='path to pretrained model')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', help='path to save checkpoints')

    # Wandb settings
    parser.add_argument('--wandb_project', default='HRC_old_1', help='wandb project name')
    # parser.add_argument('--wandb_name', default='Test2-32rand-512after_fixedNorm_0.001decay', help='wandb run name')
    parser.add_argument('--wandb_mode', default='online', type=str, help='choose from on, offline, disabled')
    parser.add_argument('--wandb_api_key', default='x'*40, help='wandb key')

    # Other settings
    parser.add_argument('--result_look_back_episode', default=[10, 20, 50, 100], type=list, help='number of episodes to look back for best result')
    parser.add_argument('--normalized_human_response', default=True, type=bool, help='if True, assume env returns normalized human response')
    parser.add_argument('--add_noise_during_grid_search', default=20, type=int, help='whether to add noise during grid search, set to 0 or false to deactivate')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for smaller cycles')  # default: False if store_true
    parser.add_argument('--slurm_id', default=0, type=int, help='slurm id, used to mark runs')
    parser.add_argument('--arg_notes', default="new code, but old test with pre-normalized env", type=str, help='notes for this run, will be stored in wandb')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    args = choose_device(args)
    start_time = time.time()

    if not args.wandb_api_key == 'x'*40:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_MODE"] = args.wandb_mode
    if args.debug_mode:  # small test
        print(f"In debug mode {'!'*10}")
        args.wandb_project = 'HRC_debug_1'
        args.episode_num = 100
        args.train_step_per_episode = 10
        args.train_batch_size = 10
        args.gt_grid_search_num = 100

    env = KukaHumanResponse_Rand(normalized=args.normalized_human_response)  # Create the environment
    env.reset()
    for subject_id in range(18):  # 18 subjects
        print('\n\n------------------------------------ Subject', subject_id, '------------------------------------')
        args.sub_id = subject_id
        this_run = wandb.init(project=args.wandb_project, name=f"Subject_{args.sub_id}", config=vars(args))  # Initialize a new run
        this_run.define_metric("train/episode")  # define our custom x axis metric
        this_run.define_metric("train/*", step_metric="train/episode")  # set all other train/ metrics to use this step

        env.reset_task(subject_id)


        model = HumanResponseModel().to(args.device)   # Create the model
        data_buffer = DataBuffer(args)  # Create the data buffer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # Define the optimizer and the loss function
        loss_function = torch.nn.MSELoss()
        exploration_rate = args.exploration_rate

        # Step 1: fill the buffer with random data points
        for _ in range(args.random_explore_num):
            if args.slurm_id == 0:
                print(f"Fill the buffer with random data points {_}/{args.random_explore_num}...", end="\r")
            data_point = env.reset()
            human_response = data_point[:2]
            robot_state = data_point[2:]
            data_buffer.add(robot_state, human_response, np.nan, np.nan, is_exploit=False)
        current_time = time.time()
        print(f"[{(current_time - start_time)/60:.2f} min] Buffer filled with {args.random_explore_num} random data points")

        # Step 2: run n episodes of HRC interaction, generate data points and train the model in each episode
        exploit_success_num = 0
        exploit_total_num = 0
        reward = np.nan
        good_human_response = np.nan
        log_dicts = []
        for i in range(args.episode_num):
            is_exploit = np.random.random() > exploration_rate
            if is_exploit:  # exploit
                exploit_total_num += 1
                robot_state, reward, est_human_response, have_result = grid_search(args, env, model=model)
                if have_result:
                    human_response = data_buffer.normalize_human_response(env.compute_human_response(robot_state))
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
            log_dict[f"train/time (s)"] = time.time() - current_time
            log_dict[f"train/Good human response %"] = exploit_success_num / (exploit_total_num + 1e-6)
            log_dict[f"train/values/Productivity (br_per_hr)"] = reward  # can not have "." in name or wandb plot have wrong x axis
            log_dict[f"train/values/Robot movement speed (m_per_s)"] = robot_state[0]
            log_dict[f"train/values/Robot arm speed (m_per_s)"] = robot_state[1]


            log_dict[f"train/bool/Is exploit"] = float(is_exploit*1.0)
            log_dict[f"train/bool/Good human response"] = float(good_human_response*1.0)


            this_run.log(log_dict)
            log_dicts.append(log_dict)
            #### log ####

            # store in buffer
            data_buffer.add(robot_state, human_response, reward, good_human_response, is_exploit=is_exploit)
            model.train()
            for training_step in range(args.train_step_per_episode):
                if args.slurm_id == 0:
                    print(f"Training step {training_step}...", end="\r")
                train_step(args, model, data_buffer, optimizer, loss_function, args.train_batch_size)

            # update epsilon
            exploration_rate = exploration_rate * args.exploration_decay_rate

        # Grid search must be done after training if using normalization parameters from random search in data buffer
        GT_robot_state, GT_best_reward, GT_human_response, GT_have_result = grid_search(args, env, data_buffer=data_buffer, GT=True)
        if GT_have_result:
            print(f"productivity: {GT_best_reward:.2f}, human response: {GT_human_response}, robot state: {GT_robot_state}")
        else:
            print("No GT result")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print()

        for i in range(args.episode_num):  # % calculation needs to be put after the GT calculation
            re_log_dict = {}
            re_log_dict["train/episode"] = i  # our custom x axis metric
            reward = log_dicts[i][f"train/values/Productivity (br_per_hr)"]
            re_log_dict[f"train/Productivity %"] = reward / GT_best_reward
            this_run.log(re_log_dict)


        # step 3: look back few episodes to find best result for this subject
        #### find converge and log ####
        print(f"Best result in the looking back {args.result_look_back_episode} episode in buffer")
        print("Logging table in wandb...")
        # a) table header here (one row)
        wandb_GT_table = wandb.Table(
            columns=["Subject", "Category", "Look Back Num", "Good Human Response",
                     "Productivity", "Productivity %",
                     # "Observed Valance", "Observed Arousal",
                     "Observed Normalized Valance", "Observed Normalized Arousal",
                     "Robot Movement Speed", "Arm Swing Speed",
                     "Proximity", "Autonomy", "Collab"])

        # b) GT result (one row)
        GT_human_response_normalized = data_buffer.normalize_human_response(GT_human_response)
        wandb_GT_table.add_data(args.sub_id, "GT", None, None,
                                GT_best_reward, None,
                                # *GT_human_response,
                                *GT_human_response_normalized,
                                *GT_robot_state)

        # c) simple strategy results (multiple rows)
        strategies = [MaxProductivityStrategy(), SearchDownStrategy()]
        for strategy in strategies:
            strategy.find_best_state(env, data_buffer)
            wandb_GT_table.add_data(args.sub_id, f"{strategy.strategy_name}", None, strategy.good_human_response,
                                    strategy.best_productivity, strategy.best_productivity / GT_best_reward,
                                    # *strategy.best_human_response,
                                    *strategy.best_human_response,  # already normalized
                                    *strategy.best_robot_state)

        # d) look back converge results (multiple rows)
        for look_back_episode in args.result_look_back_episode:  # [5,10,20,50,100]
            converge_result, found_result = look_back_in_buffer(data_buffer, look_back_episode)
            wandb_GT_table.add_data(args.sub_id, "Results", look_back_episode, found_result,
                                    converge_result["productivity"], converge_result["productivity"] / GT_best_reward,
                                    # *converge_result["human_response"],
                                    *converge_result["human_response_normalized"],
                                    *converge_result["robot_state"])

        this_run.log({f"Train/Results": wandb_GT_table})

        # Save the model and result
        checkpoint_file = f"{args.checkpoint_dir}/{args.wandb_project}/subject_{args.sub_id}.pt"
        if not os.path.exists(os.path.dirname(checkpoint_file)):
            os.makedirs(os.path.dirname(checkpoint_file))
        torch.save(model.state_dict(), checkpoint_file)

        elapsed_time = time.time() - start_time
        subject_time = time.time() - current_time
        current_time = time.time()
        print(f"[{(elapsed_time)/60:.2f} min] Subject {subject_id} finished, subject time: {(subject_time)/60:.2f} min")
        this_run.finish()
