import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from pde import PDES
from model import MLPNet, init_weights
from rng import Generate_sobol,Normdf_inv,setup_seed



def train_model(config,
                batch_size,
                seed,
                SampleMethod,
                device = torch.device('cuda:0')):
    setup_seed(seed)
    dim = config['dim']
    lr = config['lr']
    decay_step = config['decay_step']
    decay_ratio = config['decay_ratio']
    model = MLPNet(dim,config['width'],config['depth'])
    model.apply(init_weights)
    model.to(device)
    pde_target = PDES[config['PDE']](dimension = dim, device=device)
    N = pde_target.params["N"]
    loss_target = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),lr)
    lr_reduce = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size = decay_step,
                                          gamma = decay_ratio)
    x_test=np.loadtxt(config['x_val_path'])
    y_val=np.loadtxt(config['y_val_path'])
    y_val = y_val.reshape(-1,1)
    x_test=torch.Tensor(x_test).to(device)
    y_val=torch.Tensor(y_val).to(device)
    loss_records = np.zeros(config['num_iter']+1)
    L2_records = np.zeros(config['num_iter']+1)
    model.eval()
    y_test = model(x_test)
    L2_records[0] = torch.norm(y_test-y_val,2).item()/torch.norm(y_val,2).item()
    store_path = os.path.join(os.getcwd(),
                              f"{config['PDE']}_{dim}dim_data/batchsize_{batch_size}/lr_{lr}_decay_{decay_step}_ratio_{decay_ratio}")
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    for i in tqdm(range(config['num_iter'])):
        #train
        model.train()
        optimizer.zero_grad()
        if SampleMethod == 'MC':
            x = torch.rand(batch_size,dim,device=device)
            w = torch.randn(batch_size, dim, device=device)
        if SampleMethod == 'RQMC':
            xw = Generate_sobol(batch_size,(N+1)*dim,0,1,scramble = True)
            xw = xw.to(device)
            x = xw[:,0:dim]
            w = Normdf_inv(xw[:,dim:(N+1)*dim]) 
        x = pde_target.region_uniform(x)
        y = pde_target.sde(x,w)
        output = model(x)
        losses = loss_target(output, y)
        losses.backward()
        optimizer.step()
        lr_reduce.step()
        
        #compute L_2 error
        loss_records[i+1] = losses.item()
        model.eval()
        y_test = model(x_test)
        L2_records[i+1] = torch.norm(y_test-y_val,2).item()/torch.norm(y_val,2).item()
    np.savetxt(os.path.join(store_path,f"loss_{SampleMethod}_{seed}seed.txt")
            , loss_records)
    np.savetxt(os.path.join(store_path,f"l2error_{SampleMethod}_{seed}seed.txt")
            , L2_records)
    torch.save(model,
               os.path.join(store_path,f"{config['PDE']}_{SampleMethod}_{seed}seed.pth"))
        
       