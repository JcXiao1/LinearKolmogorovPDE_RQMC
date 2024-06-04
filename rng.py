import torch
import math
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     random.seed(seed)

## generates minibatch_size lower discrepancy tensors on [lower,upper]^dimension 
def Generate_sobol(size,dimension,lower,upper,scramble = False):
    sampler = torch.quasirandom.SobolEngine(dimension,scramble)
    m = int(math.log(size, 2))
    x = lower+sampler.draw_base2(m)*(upper-lower)
    return x

def Normdf_inv(sample):
    v = 0.5 + (1 - torch.finfo(sample.dtype).eps) * (sample - 0.5)
    norm_sample = torch.erfinv(2 * v - 1) * math.sqrt(2)
    return norm_sample
