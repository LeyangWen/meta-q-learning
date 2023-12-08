import torch
import numpy as np

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




def random_explore(args, env):
    data_point = env.reset()
    human_response = data_point[:2]
    robot_state = data_point[2:]
    return human_response, robot_state