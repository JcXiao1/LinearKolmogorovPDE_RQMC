# Deep learing based on the RQMC sampling method for solving linear Kolmogorov PDEs

## File Descriptions

- `bestparams.pkl`: A pickle file containing the best parameters.
- `hyperparam.py`: Script for handling hyperparameters.
- `hyperparam.sh`: Shell script for setting up or running hyperparameter-related tasks.
- `hyperparams.pkl`: A pickle file for storing hyperparameters.
- `model.py`: Script defining the model architecture and related functions.
- `pde.py`: Script for handling partial differential equations (PDEs) related operations.
- `rng.py`: Script for RQMC-type random number generation.
- `train.py`: Script containing the `train_model` function used for training the model.
- `train_PDE_bestparam.py`: Main script to run for training the model using the best parameters.


## Usage

To run the `train_PDE_bestparam.py` script, you need to provide several command-line arguments. Below are the instructions for running the script.

### Running the Training Script

```sh
python train_PDE_bestparam.py --pde <PDE> --dim <DIM> --log2_batchsize <LOG2_BATCHSIZE> --sample <SAMPLE_METHOD> --gpu <GPU_ID>
