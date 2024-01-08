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