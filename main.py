import torch
import numpy as np
import random
from tcgu import TCGU
from parameter_parser import parameter_parser
import warnings
warnings.filterwarnings("ignore")

def _set_random_seed(seed=2024):
    
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")

if __name__=='__main__':
    _set_random_seed(seed=2024)
    args = parameter_parser()
    TCGU(args)