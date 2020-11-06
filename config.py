import os

# File location settings
datadir = 'data/'
root = 'results/' #savedir

if not os.path.exists(root):
    os.mkdir(root)

# General settings
seed = 1 #random state

# Data split settings
val_size = .25
test_size = .25 #this is the percentage of the validation set assigned to the test set

# Model settings
model_names = ('Hospital-mean', 'HGLM', 'Elastic net', 'Fully non-linear', 'Proposed model')

# which models to train
train_models    = { 'hospital_only' : False,
                    'hglm'          : False,
                    'lasso'         : False,
                    'black_box'     : False,
                    'nn'            : False}
# which models to optimize
optimize_models = { 'hospital_only' : False,
                    'hglm'          : False,
                    'lasso'         : False,
                    'black_box'     : False, 
                    'nn'            : False} 

for m in train_models.keys():
    if not os.path.exists(root + m):
        os.mkdir(root + m)

# Hyperparameters used if nn is not optimized
hyper_params = dict(layers_diag = 0, 
                    layers_patient = 2, 
                    nodes_diag = 512, 
                    nodes_patient = 256, 
                    dropout = .25)

# Hyperparametergrid if model is optimized
param_grid = {'nn': dict(layers_diag = [0,1,2],
                         layers_patient = [0,1,2],
                         nodes_diag = [512, 1024],
                         nodes_patient = [256, 512],
                         dropout = [.25, .5]),
			  'hglm': dict(lambda_1 = [1e-4],
                           lambda_2 = [1e-3, 1e-4, 1e-5, 1e-6]),
			  'lasso': dict(lambda_1 = [1e-3, 1e-4, 1e-5, 1e-6],
                            lambda_2 = [1e-3, 1e-4, 1e-5, 1e-6])}

# training parameters
epochs = 100
patience = 5 
batch_size = 16384