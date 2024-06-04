from train import train_model
import random 
import torch
import argparse
import pickle
                
if __name__ == '__main__':    
    with open('hyperparams.pkl', 'rb') as file:
        tasks = pickle.load(file)
    parser = argparse.ArgumentParser(description='Train a model on a specified GPU')
    parser.add_argument('--task_id', type=int, required=True, help='task_id')
    parser.add_argument('--gpu', type=int, required=True, help='GPU device ID')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')
    config = tasks[args.task_id]
    batch_size,seed,SampleMethod = config['batch_size'],config['seed'],config['SampleMethod']
    train_model(config,batch_size,seed,SampleMethod,device)