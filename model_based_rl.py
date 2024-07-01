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
from utility.CriteriaChecker import CriteriaChecker
from utility.OptimalResult import OptimalResult
# todo: meta learning - change to one subject only, write code to determine convergence speed, write meta-eval loops


def train_step(args, model, data_buffer, optimizer, loss_function, batch_size):
    # Sample data points from the buffer
    human_responses, robot_states = data_buffer.sample(batch_size)
    if not args.normalized_human_response:  # env returns actual human response not normalized, normalize here using cumulative mean and std from explore data points
        human_responses = data_buffer.normalize_human_response_batch(
            human_responses)

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
            noise = np.random.randint(-args.add_noise_during_grid_search,
                                      args.add_noise_during_grid_search)
        else:
            noise = 0
        search_num = args.grid_search_num + noise
    continuous_bin = np.linspace(0, 1, search_num)
    binary_bin = [env.low_binary, env.high_binary]  # [-1, 1]
    bin_map = [(0,) * env.num_responses + (a, b, x, y, z) for a in continuous_bin for b in continuous_bin for x in binary_bin for y in
               binary_bin for z in binary_bin]
    move_spd_low_bnd, move_spd_high_bnd = [
        env.move_spd_low_bnd, env.move_spd_high_bnd]  # [27.8, 143.8]
    arm_spd_low_bnd, arm_spd_high_bnd = [
        env.arm_spd_low_bnd, env.arm_spd_high_bnd]  # [23.8, 109.1]

    # full state needed only for env.compute_human_response
    full_states = np.array(bin_map)
    full_states[:, env.num_responses] = full_states[:, env.num_responses] * \
        (move_spd_high_bnd - move_spd_low_bnd) + move_spd_low_bnd
    full_states[:, env.num_responses + 1] = full_states[:, env.num_responses + 1] * \
        (arm_spd_high_bnd - arm_spd_low_bnd) + arm_spd_low_bnd

    if not GT:
        robot_states = torch.from_numpy(full_states[:, env.num_responses:]).float().to(
            args.device)  # Convert numpy array to PyTorch tensor
        valances, arousals, engagements, vigilances = model(robot_states).detach(
        ).cpu().numpy().T  # output of the model is already normalized

    optimal_result = OptimalResult()

    for i in range(len(bin_map)):
        this_state = full_states[i]
        travelTime = env.calculate_traveltime(this_state[env.num_responses], this_state[env.num_responses+1],
                                              this_state[env.num_responses+2], this_state[env.num_responses+3], this_state[env.num_responses+4])
        productivity = env.calculate_productivity(travelTime)
        if GT:
            this_human_response = env.compute_human_response(this_state)
            # if args.normalized_human_response, env returns normalized human response, otherwise, return actual human response
            # data_buffer knows if it still needed to be normalized, so just pass it to data_buffer.normalize_human_response
            human_response = data_buffer.normalize_human_response(
                this_human_response)

        else:
            valance = valances[i]
            arousal = arousals[i]
            engagement = engagements[i]
            vigilance = vigilances[i]
            human_response = [valance, arousal, engagement, vigilance]

        # Determine 4 -> 2 - Function, return 2 bool
        # Check if is normalized, if so using 'env', else using 'data_buffer'
        if not args.normalized_human_response:
            centroid_loader = data_buffer
        else:
            centroid_loader = env

        robot_state = this_state[env.num_responses:]

        # Run OptimalResult's check_and_update to check and update best info
        optimal_result.check_and_update(human_response, robot_state, productivity, normalized=args.normalized_human_response,
                                        eng_centroids=centroid_loader.eng_centroids, vig_centroids=centroid_loader.vig_centroids,
                                        eng_normalized_centroids=centroid_loader.eng_normalized_centroids, vig_normalized_centroids=centroid_loader.vig_normalized_centroids)

    return optimal_result


def random_explore(args, env):
    data_point = env.reset()
    raw_human_response = data_point[:env.num_responses]
    robot_state = data_point[env.num_responses:]
    # travelTime = env.calculate_traveltime(data_point[2], data_point[3], data_point[4], data_point[5], data_point[6])
    # productivity = env.calculate_productivity(travelTime)
    return raw_human_response, robot_state


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

    # Extract the is_exploit_buffer array
    is_exploit_buffer = data_buffer.is_exploit_buffer[-look_back_episode:]

    # Extract the response_satisfy_number_arrays buffer
    # All columns from -look_back_episode and to the right
    response_satisfy_number_buffers = data_buffer.response_satisfy_number_buffers[:, -
                                                                look_back_episode:]

    # Multiply the two arrays to get a mask_array that is both exploit and satisfy the numbers
    # E.G, is_exploit = [False, True], and response_satisfy_number_buffers=[[True, False], [True, True]]
    # Result: [[False, False], [False, True]]
    is_exploit_response_satisfy_number_buffers = [
        is_exploit_buffer * response_satisfy_number_buffer for response_satisfy_number_buffer in response_satisfy_number_buffers]

    # Multiply the is_exploit_response_satisfy_number_buffers with our productivity buffer
    # EG, is_exploit_response_satisfy_numbers_array = [[True, False], [False, True]], productivity = [21, 32]
    # Result: [[21, 0], [0, 32]]
    response_satisfy_number_productivity = [data_buffer.productivity *
                                   is_exploit_response_satisfy_number_buffer for is_exploit_response_satisfy_number_buffer in is_exploit_response_satisfy_number_buffers]

    # Start with the last index(satisfy number)
    # E.G, if there are 4 human responses, the response_satisfy_number_productivity will be np.array of [[index 0], [index 1], [index 2], [index 3], [index 4]]
    # where each index indicates the number of human responses satisfy the critier
    # We want to check from the back (4 -> 3 -> 2 -> 1), we don't consider the case when no human response satisfies
    for response_satisfy_number in range(len(response_satisfy_number_productivity) - 1, 0, -1):
        # Calculate the maximum productivity of all cases where "response_satisfy_number" of human_responses are satisfied
        this_response_satisfy_number_max_productivity = np.max(
            response_satisfy_number_productivity[response_satisfy_number])

        # If the max value is larger than 0, this means at least some episodes reach the response_satisfy_number
        if this_response_satisfy_number_max_productivity > 0:
            # Find the index of the max_productivity
            index = np.argmax(response_satisfy_number_productivity[response_satisfy_number])

            # Then, the Actual Index of that episode in the data_buffer will be -look_back_episode + index
            # EG. if the original productivity is [12, 18, 23, 20, 11]
            # If look_back_episode is 4, we have -> [18, 23, 20, 11]
            # Then, the argmax index will be 1 corresponding to 23
            # Means -look_back_episode + index = -4 + 1 = -3, which is 23's index in the original productivity

            best_productivity = this_response_satisfy_number_max_productivity
            converge_result["robot_state"] = data_buffer.robot_state_buffer[-look_back_episode + index]
            converge_result["human_response"] = data_buffer.human_response_buffer[-look_back_episode + index]
            converge_result["human_response_normalized"] = data_buffer.normalize_human_response(
                data_buffer.human_response_buffer[-look_back_episode + index])
            converge_result["productivity"] = data_buffer.productivity_buffer[-look_back_episode + index]
            found_result = True

            # We can break since the order is guaranteened to be descending
            break

    return converge_result, found_result


def parse_args():
    parser = argparse.ArgumentParser(description='Human Response Model')
    parser.add_argument('--device', default='cuda', help='device, cpu or cuda')
    # Training parameters
    parser.add_argument('--grid_search_num', default=100,
                        type=int, help='number of grid search, positive integer')
    parser.add_argument('--gt_grid_search_num', default=500, type=int,
                        help='number of grid search for GT, positive integer')
    parser.add_argument('--random_explore_num', default=128,
                        type=int, help='number of random explore, positive integer')
    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='batch size for training, positive integer')
    parser.add_argument('--train_step_per_episode', default=1024, type=int,
                        help='number of training steps per episode, positive integer')
    parser.add_argument('--episode_num', default=512, type=int,
                        help='batch size for training, positive integer')
    parser.add_argument('--exploration_rate', default=0.5,
                        type=float, help='exploration rate, float between 0 and 1')
    parser.add_argument('--exploration_decay_rate', default=0.99,
                        type=float, help='exploration decay rate, float between 0 and 1')
    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help='learning rate, float between 0 and 1')
    parser.add_argument('--weight_decay', default=0.001,
                        type=float, help='weight decay, float between 0 and 1')
    # parser.add_argument('--pretrained_model', default=None, help='path to pretrained model')
    parser.add_argument(
        '--checkpoint_dir', default='./checkpoints', help='path to save checkpoints')

    # Wandb settings
    parser.add_argument(
        '--wandb_project', default='HRC_4HR_all_2', help='wandb project name')
    # parser.add_argument('--wandb_name', default='Test2-32rand-512after_fixedNorm_0.001decay', help='wandb run name')
    parser.add_argument('--wandb_mode', default='disabled',
                        type=str, help='choose from on, offline, disabled')
    parser.add_argument('--wandb_api_key', default='x'*40, help='wandb key')

    # Other settings
    parser.add_argument("--num_responses", default=4, type=int, help="number of human responses")
    parser.add_argument('--result_look_back_episode', default=[
                        10, 20, 50, 100], type=list, help='number of episodes to look back for best result')
    parser.add_argument('--normalized_human_response', default=True, type=bool,
                        help='if True, assume env returns normalized human response')
    parser.add_argument('--add_noise_during_grid_search', default=20, type=int,
                        help='whether to add noise during grid search, set to 0 or false to deactivate')

    parser.add_argument('--debug_mode', default=False,
                        help='Enable debug mode for smaller cycles')
    parser.add_argument('--slurm_id', default=0, type=int,
                        help='slurm id, used to mark runs')
    parser.add_argument('--arg_notes', default="increased number of random explore upfront to help with estimating mean",
                        type=str, help='notes for this run, will be stored in wandb')
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

    env = KukaHumanResponse_Rand(
        normalized=args.normalized_human_response, num_responses=args.num_responses)  # Create the environment
    env.reset()
    for subject_id in range(18):  # 18 subjects
        print('\n\n------------------------------------ Subject',
              subject_id, '------------------------------------')
        args.sub_id = subject_id
        this_run = wandb.init(project=args.wandb_project, name=f"Subject_{args.sub_id}", config=vars(
            args))  # Initialize a new run
        # define our custom x axis metric
        this_run.define_metric("train/episode")
        # set all other train/ metrics to use this step
        this_run.define_metric("train/*", step_metric="train/episode")

        env.reset_task(subject_id)

        model = HumanResponseModel().to(args.device)   # Create the model
        data_buffer = DataBuffer(args)  # Create the data buffer
        # Define the optimizer and the loss function
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loss_function = torch.nn.MSELoss()
        exploration_rate = args.exploration_rate

        # # Step 1: fill the buffer with random data points
        # for _ in range(args.random_explore_num):
        #     if args.slurm_id == 0:
        #         print(
        #             f"Fill the buffer with random data points {_}/{args.random_explore_num}...", end="\r")
        #     data_point = env.reset()
        #     raw_human_response = data_point[:env.num_responses]
        #     robot_state = data_point[env.num_responses:]
        #     data_buffer.add(robot_state, raw_human_response,
        #                     np.nan, np.nan, np.nan, np.nan, is_exploit=False)
        # current_time = time.time()
        # print(f"[{(current_time - start_time)/60:.2f} min] Buffer filled with {args.random_explore_num} random data points")
        #
        # # Step 2: run n episodes of HRC interaction, generate data points and train the model in each episode
        # exploit_success_num = 0
        # exploit_total_num = 0
        # reward = np.nan
        # good_human_response_all = np.nan
        # good_human_response_val_aro = np.nan
        # good_human_response_eng_vig = np.nan
        # log_dicts = []
        # for i in range(args.episode_num):
        #     is_exploit = np.random.random() > exploration_rate
        #     if is_exploit:  # exploit
        #         robot_state, reward, est_human_response, have_result, est_satisfy_type = grid_search(
        #             args, env, model=model, data_buffer=data_buffer)
        #         if have_result:
        #             exploit_total_num += 1
        #             raw_human_response = env.compute_human_response(
        #                 robot_state)
        #             human_response = data_buffer.normalize_human_response(
        #                 raw_human_response)
        #
        #             is_satisfy_val_aro, is_satisfy_eng_vig = CriteriaChecker.satisfy_all_requirements(human_response, normalized=args.normalized_human_response,
        #                                                                                               eng_centroids=env.eng_centroids, vig_centroids=env.vig_centroids,
        #                                                                                               eng_normalized_centroids=env.eng_normalized_centroids, vig_normalized_centroids=env.vig_normalized_centroids)
        #             good_human_response_all = is_satisfy_eng_vig and is_satisfy_val_aro
        #             good_human_response_val_aro = is_satisfy_val_aro
        #             good_human_response_eng_vig = is_satisfy_eng_vig
        #
        #             with np.printoptions(precision=2):
        #                 if good_human_response_val_aro:
        #                     exploit_success_num += 1
        #                     print(
        #                         f"{i}, good_HR_ALL:{good_human_response_all}, good_VAL_ARO:{good_human_response_val_aro}, good_ENG_VIG:{good_human_response_eng_vig}, prod:{reward:.2f}, HR:{human_response}, robot state:{robot_state}", end="\r")
        #         else:  # random point since grid search got no results with positive valance and arousal
        #             is_exploit = False
        #             raw_human_response, robot_state = random_explore(args, env)
        #     else:  # random explore
        #         raw_human_response, robot_state = random_explore(args, env)
        #
        #     #### log ####
        #     log_dict = {}
        #     log_dict["train/episode"] = i  # our custom x axis metric
        #     log_dict[f"train/time (s)"] = time.time() - current_time
        #     log_dict[f"train/Good human response %"] = exploit_success_num / \
        #         (exploit_total_num + 1e-6)
        #     # can not have "." in name or wandb plot have wrong x axis
        #     log_dict[f"train/values/Productivity (br_per_hr)"] = reward
        #     log_dict[f"train/values/Robot movement speed (m_per_s)"] = robot_state[0]
        #     log_dict[f"train/values/Robot arm speed (m_per_s)"] = robot_state[1]
        #
        #     log_dict[f"train/bool/Is exploit"] = float(is_exploit*1.0)
        #     log_dict[f"train/bool/Good human response all"] = float(
        #         good_human_response_all*1.0)
        #     log_dict[f"train/bool/Good human response val-aro"] = float(
        #         good_human_response_val_aro*1.0)
        #     log_dict[f"train/bool/Good human response eng-vig"] = float(
        #         good_human_response_eng_vig*1.0)
        #
        #     this_run.log(log_dict)
        #     log_dicts.append(log_dict)
        #     #### log ####
        #
        #     # store in buffer
        #     # is_exploit)  # good human response might change as norm params change
        #     data_buffer.add(robot_state, raw_human_response,
        #                     reward, good_human_response_all, good_human_response_val_aro, good_human_response_eng_vig, is_exploit=True)
        #     model.train()
        #     for training_step in range(args.train_step_per_episode):
        #         if args.slurm_id == 0:
        #             print(f"Training step {training_step}...", end="\r")
        #         train_step(args, model, data_buffer, optimizer,
        #                    loss_function, args.train_batch_size)
        #
        #     # update epsilon
        #     exploration_rate = exploration_rate * args.exploration_decay_rate

        # Grid search must be done after training if using normalization parameters from random search in data buffer
        GT_robot_state, GT_best_reward, GT_human_response, GT_have_result, GT_satisfy_type = grid_search(
            args, env, data_buffer=data_buffer, GT=True)
        print()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if GT_have_result:
            print(
                f"productivity: {GT_best_reward:.2f}, human response: {GT_human_response}, robot state: {GT_robot_state}, satisfy type: {GT_satisfy_type}")
        else:
            print("No GT result")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ GT @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print()

        # # % calculation needs to be put after the GT calculation
        # for i in range(args.episode_num):
        #     re_log_dict = {}
        #     re_log_dict["train/episode"] = i  # our custom x axis metric
        #     reward = log_dicts[i][f"train/values/Productivity (br_per_hr)"]
        #     re_log_dict[f"train/Productivity %"] = np.nan if np.isnan(
        #         reward) else (reward / GT_best_reward)
        #     this_run.log(re_log_dict)
        #
        # # step 3: look back few episodes to find best result for this subject
        # #### find converge and log ####
        # print(
        #     f"Best result in the looking back {args.result_look_back_episode} episode in buffer")
        # print("Logging table in wandb...")
        # # a) table header here (one row)
        # # MODIFY: Add the 2 columns for engagement and vigilance
        # wandb_GT_table = wandb.Table(
        #     columns=["Subject", "Category", "Look Back Num", "Good Human Response",
        #              "Productivity", "Productivity %",
        #              # "Observed Valance", "Observed Arousal",
        #              "Observed Normalized Valance", "Observed Normalized Arousal", "Observed Normalized Engagement", "Observed Normalized Vigilance",
        #              "Robot Movement Speed", "Arm Swing Speed",
        #              "Proximity", "Autonomy", "Collab"])
        #
        # # b) GT result (one row)
        # wandb_GT_table.add_data(args.sub_id, "GT", None, None,
        #                         GT_best_reward, None,
        #                         # *GT_human_response,
        #                         *GT_human_response,
        #                         *GT_robot_state)
        #
        # # c) Simple strategy results (multiple rows)
        # strategies = [MaxProductivityStrategy(), SearchDownStrategy()]
        # for strategy in strategies:
        #     strategy.find_best_state(env, data_buffer)
        #     wandb_GT_table.add_data(args.sub_id, f"{strategy.strategy_name}", None, strategy.good_human_response,
        #                             strategy.best_productivity, strategy.best_productivity / GT_best_reward,
        #                             # *strategy.best_human_response,
        #                             *strategy.best_human_response,  # already normalized
        #                             *strategy.best_robot_state)
        #
        # # d) look back converge results (multiple rows)
        # # [5,10,20,50,100]
        # for look_back_episode in args.result_look_back_episode:
        #     converge_result, found_result = look_back_in_buffer(
        #         data_buffer, look_back_episode)
        #     wandb_GT_table.add_data(args.sub_id, "Results", look_back_episode, found_result,
        #                             converge_result["productivity"], converge_result["productivity"] /
        #                             GT_best_reward,
        #                             # *converge_result["human_response"],
        #                             *converge_result["human_response_normalized"],
        #                             *converge_result["robot_state"])
        #
        # this_run.log({f"Train/Table/Results": wandb_GT_table})
        #
        # # Save the model and result
        # checkpoint_file = f"{args.checkpoint_dir}/{args.wandb_project}/subject_{args.sub_id}.pt"
        # if not os.path.exists(os.path.dirname(checkpoint_file)):
        #     os.makedirs(os.path.dirname(checkpoint_file))
        # torch.save(model.state_dict(), checkpoint_file)
        #
        # elapsed_time = time.time() - start_time
        # subject_time = time.time() - current_time
        # current_time = time.time()
        # print(f"[{(elapsed_time)/60:.2f} min] Subject {subject_id} finished, subject time: {(subject_time)/60:.2f} min")
        this_run.finish()
