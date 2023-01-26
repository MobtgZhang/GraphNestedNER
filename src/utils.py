import random
import numpy as np
import torch

def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)
