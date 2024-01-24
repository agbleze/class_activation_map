

import torch
import torchvision
import matplotlib.pyplot as plt

import time
import os
import numpy as np
import random
from distutils.version import LooseVersion as version
from itertools import product

def seed_setting(sd):
    os.environ["PL_GLOBAL_SEED"] = str(sd)
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)

