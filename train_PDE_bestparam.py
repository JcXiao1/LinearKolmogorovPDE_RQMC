from train import train_model
import random 
import torch
import argparse
import pickle
                
if __name__ == '__main__':    
    with open('bestparams.pkl','rb') as file:
        tasks = pickle.load(file)
    parser = argparse.ArgumentParser(description='Train a model on a specified GPU')
    parser.add_argument('--pde', type=str, required=True, choices=['BSmodel','Heat_Paraboloid'], help='pde')
    parser.add_argument('--dim', type=int, required=True, choices=[5,50], help='dim')
    parser.add_argument('--log2_batchsize', type=int, required=True, help='log2(batchsize)')
    parser.add_argument('--sample', type=str, required=True, choices=['MC','RQMC'], help='sample method')
    parser.add_argument('--gpu', type=int, required=True, help='GPU device ID')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    for config in tasks:
        if (config['batch_size'] == 2**(args.log2_batchsize))&(config['SampleMethod'] == args.sample)&\
            (config['PDE'] == args.pde)&(config['dim'] == args.dim):
            PDE,batch_size,dim = config['PDE'],config['batch_size'],config['dim'] 
            config['store_path'] = f'{PDE}_{dim}dim/batchsize_{batch_size}'
            train_model(config,
                        batch_size,
                        config['seed'],
                        config['SampleMethod'],
                        device)