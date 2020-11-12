"""
Set config.py before running this file

Use this file to read NRD (after running data_pipeline.py), and to train and optimize models

Note that the HGLM and LASSO are trained according to CCS disease categories (as in the Yale-New Haven report), 
and are thus trained differently than the NN's.

The data (NRD) is quite big, so make sure you have the right hardware.
Because the data is so big, a batch_generator has to be used. 
Additionally, because the hosp array is huge and therefore sparse, we cannot calculate the error metrics in a normal way.
For this we have to use the metrics_callback.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.pser_process_gpu_memory_fraction = 0.5
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import pandas as pd
import numpy as np
import scipy as sp

from models import NN, REGR
from plot import PlotResults
from config import *

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from keras.callbacks import ModelCheckpoint
from clr_callback import *
from metrics_callback import *

import gc
import time
t_begin = time.time()

# This step takes long, so you can turn it off if you're not training any models with this parameter
read_ccs = True

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.sparse.csr_matrix((loader['arr_0'], loader['indices'], loader['indptr']), shape=loader['shape'])
 
def batch_generator(y, diag, patient, hosp, cohort=None):
    """
    Generate batches because the hospital sparse array cannot fit into RAM if it is dense
    Source: https://stackoverflow.com/questions/37609892/keras-sparse-matrix-issue?rq=1
    """
    samples_per_epoch = y.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y)[0])
    while 1:       
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        y_batch = y[index_batch]
        patient_batch = patient[index_batch,:]
        hosp_batch = hosp[index_batch,:].todense()
        diag_batch = diag[index_batch,:]
        if cohort is not None:
            cohort_batch = cohort[index_batch,:].todense()
            diag_batch = diag_batch.todense()
        counter += 1
        x_batch = [np.array(diag_batch), np.array(patient_batch), np.array(hosp_batch)]
        if cohort is not None:
            x_batch.append(np.array(cohort_batch))
        yield x_batch, y_batch
        if (counter >= number_of_batches): #after using all data of the training set
            counter=0
            np.random.shuffle(index) #shuffle so next epoch uses different batch order

def train(model, model_type, savedir):
    """
    Train neural network
    """
    filepath=savedir+'weights-'+model_type+'.hdf5'
    n = int(1e5)

    # data to evaluate after each epoch
    if model_type == 'hglm' or model_type == 'lasso':
        x_train = X_diag_train, patient_train, hosp_train, X_cohort_train
        x_val = X_diag_val, patient_val, hosp_val, X_cohort_val
        eval_x_train = [np.array(X_diag_train[:n].todense()), np.array(patient_train[:n]), np.array(hosp_train[:n].todense()), np.array(X_cohort_train[:n].todense())]
        eval_x_val = [np.array(X_diag_val[:n].todense()), np.array(patient_val[:n]), np.array(hosp_val[:n].todense()), np.array(X_cohort_val[:n].todense())]
    else:
        x_train = diag_train, patient_train, hosp_train
        x_val = diag_val, patient_val, hosp_val
        eval_x_train = [np.array(diag_train[:n]), np.array(patient_train[:n]), np.array(hosp_train[:n].todense())]
        eval_x_val = [np.array(diag_val[:n]), np.array(patient_val[:n]), np.array(hosp_val[:n].todense())]

    # compute class weights
    class_weights = compute_class_weight('balanced', np.unique(y_train.squeeze()), y_train.squeeze())

    # callbacks
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min', save_best_only=True) #model checkpoint
    clr_triangular = CyclicLR(mode='triangular', step_size=1000.) #cyclic learning rate
    metrics = Metrics(training_data=(eval_x_train, y_train[:n]), validation_data=(eval_x_val, y_val[:n])) 
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)
    callbacks_list = [checkpoint, metrics, clr_triangular, earlystopping]

    # fit model (use generator for unsparsing hospital array
    model.fit_generator(generator = batch_generator(y_train, *x_train),
                        steps_per_epoch = np.ceil(y_train.shape[0] / batch_size),
                        epochs = epochs,
                        verbose = 1,
                        callbacks = callbacks_list,
                        class_weight=class_weights,
                        validation_data = batch_generator(y_val, *x_val),
                        validation_steps = np.ceil(y_val.shape[0] / batch_size))

    # load weights of best model 
    model.load_weights(filepath) 
    model.metrics = metrics

    # save training and validation metrics
    results = pd.DataFrame.from_dict(model.history.history).join(pd.DataFrame.from_dict(model.metrics.metrics))
    results.to_csv(savedir+'training_metrics_mean.csv')
        
    # plot training and validation metrics
    PlotResults(savedir).plot_training_metrics(model)
    
    return model

def evaluate(y_true, y_pred, cohort, savedir, plotting=False):
    """
    Calculate precision, recall, f1, bce, roc-auc and pr-auc given the true values (y_true) and prediction probabilities (y_pred)
    t = the probability threshold for binary classification labels.
    """
    # determine classification threshold
    fpr, tpr, thres = roc_curve(y_true, y_pred)
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=np.arange(len(tpr)) ), 'threshold' : pd.Series(thres, index=np.arange(len(tpr)))})
    t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['threshold'].values[0]
    y_label = (y_pred > t).astype(np.int)
    
    # scores
    precision = precision_score(y_true, y_label)
    recall = recall_score(y_true, y_label)
    f1 = f1_score(y_true, y_label)
    bce = log_loss(y_true, y_pred)        
    roc_auc = roc_auc_score(y_true, y_pred)
    
    # plot roc and precision-recall curves
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recalls, precisions)
    
    if plotting:
        PlotResults(savedir).plot_roc_curve(fpr, tpr, roc_auc, cohort)
        PlotResults(savedir).plot_pr_curve(precisions, recalls, pr_auc, cohort, balance)
    
    return [precision, recall, f1, bce, roc_auc, pr_auc], [fpr, tpr], [precisions, recalls]

def test(model, model_type, savedir):
    """
    Test performance on test set and on individual cohorts
    Input
        model       - (trained) keras model to be evaluated
    """
    cohorts = ['cr', 'cv', 'me', 'ne', 'su']
    metrics = ['precision', 'recall', 'f1', 'bce', 'roc-auc', 'pr-auc']
    score = dict()
    
    # evaluate performance per cohort
    for c in cohorts:
        idx = (cohort_test == c).squeeze()
        if model_type == 'hglm' or model_type == 'lasso':
            x_test = [X_diag_test[idx,:], patient_test[idx,:], hosp_test[idx,:], X_cohort_test[idx,:]]
        else:
            x_test = [diag_test[idx,:], patient_test[idx,:], hosp_test[idx,:]]
                    
        y_pred = model.predict(x_test)
        score[c], _, _ = evaluate(y_test[idx,:], y_pred, c, savedir)

    # evaluate performanec on total test set
    if model_type == 'hglm' or model_type == 'lasso':
        x_test = [X_diag_test, patient_test, hosp_test, X_cohort_test]
    else:
        x_test = [diag_test, patient_test, hosp_test]
    
    y_pred = model.predict(x_test)
    score['tot'], roc, pr = evaluate(y_test, y_pred, 'tot', savedir)
    
    score = pd.DataFrame.from_dict(score, orient = 'index', columns = metrics)
    score.to_csv(savedir+'test_metrics.csv', index=False)

    score.to_latex(savedir+'cohort_performance_'+model_type+'.tex')

    return score, roc, pr, y_pred

def gridsearch(savedir, param_grid, name='nn'):
    grid = list(ParameterGrid(param_grid[name]))
    np.save(savedir+'param_grid.npy', np.array(grid))
    
    metrics = ['val_loss', 'loss', 'precision', 'recall', 'f1', 'bce', 'roc-auc', 'pr-auc',  'val_precision', 'val_recall', 'val_f1', 'val_bce', 'val_roc-auc', 'val_pr-auc']
    simple_keys = {'dropout':'do', 'layers_diag':'ld', 'layers_patient':'lp', 'nodes_diag':'nd', 'nodes_patient':'np', 'lambda_1':'l1', 'lambda_2':'l2'}

    results = np.empty((len(grid)), dtype=object)
    optim_results = pd.DataFrame(columns=metrics, index=np.arange(len(grid))) #note that the optimal metric for each hyperparameter combination is evaluated for all different metrics, (so not that the minimal loss is taken and the rest of the columns contain metrics corresponding to that epoch)
    sort_params = pd.DataFrame(columns=metrics, index=np.arange(len(grid)))
    
    for i, g in enumerate(grid):
        dir_gs = '_'.join([simple_keys[k]+str(v) for k,v in g.items()])
        if not os.path.exists(savedir+dir_gs):
            os.mkdir(savedir+dir_gs)
        
        if name == 'nn':
            mod = NN(name, diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], num_hosp, np.mean(y_train), **g)
        elif name == 'hglm':
            mod = REGR(name, X_diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], X_cohort_train.shape[1], num_hosp, np.mean(y_train), **g)
        elif name == 'lasso':
            mod = REGR(name, X_diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], X_cohort_train.shape[1], num_hosp, np.mean(y_train), **g)

        mod.model = train(mod.model, name, savedir+dir_gs+'/')
        
        results[i] = pd.DataFrame.from_dict(mod.model.history.history).join(pd.DataFrame.from_dict(mod.model.metrics.metrics))
        results[i].to_csv(savedir+dir_gs+'/training_metrics_mean.csv')
        
        np.save(savedir+'training_metrics_gridsearch.npy', results)

        for m in metrics:
            if 'loss' in m or 'bce' in m: 
                optim_results.loc[i, m] = np.amin(results[i][m])
            else:
                optim_results.loc[i,m] = np.amax(results[i][m])
        optim_results.to_csv(savedir+'optim_training_metrics_gridsearch.csv') 
        
        PlotResults(savedir+dir_gs+'/').plot_training_metrics(mod.model) 
        test(mod.model, name, savedir+dir_gs+'/')
    
        del mod ; gc.collect()
    
    for m in metrics:
        if 'loss' in m or 'bce' in m:
            sort_params[m] = np.array(grid)[np.argsort(optim_results[m]).values]
        else:
            sort_params[m] = np.array(grid)[np.argsort(optim_results[m]).values[::-1]]
        print("metric: ", m, "best params: ", sort_params[m][0])
    
    sort_params.to_csv(savedir+'sorted_params_gridsearch.csv') 

###----------------------------------- Read data -----------------------------------###
hosp = load_sparse_csr(datadir+'hosp.npz')
diag = pd.read_csv(datadir+'diag.csv', index_col = 0, header=None).values
patient = pd.read_csv(datadir+'patient.csv', index_col = 0)
y = pd.read_csv(datadir+'labels.csv', index_col = 0, header=None).values

#standardize patient age and gender
patient['age'] = StandardScaler().fit_transform(patient['age'].values.reshape(-1,1)) 
patient['gender'] = patient['gender'].replace(0,-1)

balance = np.mean(y)
charges = patient['charges'].values
hosp_id = patient['hosp_key'].values

y_train, y_val, patient_train, patient_val, hosp_train, hosp_val, diag_train, diag_val, hosp_id_train, hosp_id_val, cohort_train, cohort_val, \
idx_train, idx_val = train_test_split(y, patient[['gender','age']].values, hosp, diag,
                                            hosp_id, patient['cohort'].values, np.arange(len(y)),
                                            stratify=hosp_id, test_size=val_size, random_state=seed)

y_val, y_test, patient_val, patient_test, hosp_val, hosp_test, diag_val, diag_test, hosp_id_val, hosp_id_test, cohort_val, cohort_test, \
idx_val, idx_test = train_test_split(y_val, patient_val, hosp_val, diag_val,
                                             hosp_id_val, cohort_val, idx_val,
                                             stratify=hosp_id_val, test_size=test_size, random_state=seed)

# free some memory
patient.drop(columns=['gender','age', 'charges'])  
del hosp, diag
gc.collect()

if train_models['hglm'] or train_models['lasso'] or read_ccs:
    ccs = pd.read_csv(datadir+'ccs.csv', index_col = 0, header = None, dtype = 'Int64').values 
    ccs_train = ccs[idx_train]
    ccs_val = ccs[idx_val]
    ccs_test = ccs[idx_test]

    #elements for one-hot encoding
    enc1 = MultiLabelBinarizer(sparse_output=True)
    enc1.fit(ccs) ; enc1.classes_ = enc1.classes_[~np.isnan(enc1.classes_.astype(float))]
    X_diag_train = enc1.transform(ccs_train)
    X_diag_val = enc1.transform(ccs_val)
    X_diag_test = enc1.transform(ccs_test)
    
    enc2 = OneHotEncoder(sparse=True)
    X_cohort_test = enc2.fit_transform(cohort_test.astype(str).reshape(-1,1))
    X_cohort_val = enc2.fit_transform(cohort_val.astype(str).reshape(-1,1))
    X_cohort_train = enc2.fit_transform(cohort_train.astype(str).reshape(-1,1))
    
del ccs

hosp_test = hosp_test.todense()
num_hosp = len(np.unique(hosp_id_test))

t_end_data = time.time()
print("All data read, total time elapsed: ", t_end_data - t_begin, "(s)")

###----------------------------------- Models -----------------------------------###
M = { 'hospital_only': NN('hospital_only', diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], num_hosp, np.mean(y_train)),
      'hglm': REGR('hglm', X_diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], X_cohort_train.shape[1], num_hosp, np.mean(y_train), lambda_2=1e-6),
      'lasso': REGR('lasso', X_diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], X_cohort_train.shape[1], num_hosp, np.mean(y_train), lambda_1=1e-6, lambda_2=1e-6),
      'black_box': NN('black_box', diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], num_hosp, np.mean(y_train), **hyper_params),
      'nn':NN('nn', diag_train.shape[1], patient_train.shape[1], hosp_train.shape[1], num_hosp, np.mean(y_train), **hyper_params)}
    
results = {i:None for i in M.keys()}
roc = {i:None for i in M.keys()}
pr = {i:None for i in M.keys()}
y_pred = {i:None for i in M.keys()}

t_begin_model = time.time()
for name in M.keys():
    print(name)
    savedir = root + name + '/'
    # optimize
    if optimize_models[name]:
        if not os.path.exists(savedir+'gridsearch/'):
            os.mkdir(savedir+'gridsearch/')
        gridsearch(savedir+'gridsearch/', param_grid, name=name)
    # train
    if train_models[name]:
        M[name].model = train(M[name].model, name, savedir)
    # only test
    else:
        M[name].model.load_weights(savedir+'weights-'+name+'.hdf5')
    
    # hospital performance by cohort
    results[name], roc[name], pr[name], y_pred[name] = test(M[name].model, name, savedir)

    t_end_model = time.time()
    print("Model fit and evaluated, time elapsed since begin fit: ", t_end_model - t_begin_model, "(s)")
    t_begin_model = t_end_model

pd.concat(results.values(), keys=results.keys()).to_csv(root+'results.csv')

###----------------------------------- Compare models -----------------------------------###
fprs = [r[0] for r in roc.values()]
tprs = [r[1] for r in roc.values()]
precisions = [r[0] for r in pr.values()]
recalls = [r[1] for r in pr.values()]
rocauc = [r['roc-auc']['tot'] for r in results.values()]
prauc = [r['pr-auc']['tot'] for r in results.values()]

np.save(root+'fprs', fprs)
np.save(root+'tprs', tprs)
np.save(root+'precisions', precisions)
np.save(root+'recalls', recalls)
np.save(root+'rocauc', rocauc)
np.save(root+'prauc', prauc)
np.save(root+'balance', balance)

PlotResults(root).plot_roc_curves(fprs, tprs, rocauc, model_names)
PlotResults(root).plot_pr_curves(precisions, recalls, prauc, balance, model_names)

###----------------------------------- Infer hospital-specific performance -----------------------------------###
M_perf = ('hospital_only', 'hglm', 'nn')
omega = {i:M[i].model.get_layer("output").get_weights()[0][0:num_hosp,:].T[0] for i in M_perf}#weights of the output layer connecting to hospitals (ignore bias weight)
beta = {i:M[i].model.get_layer('output').get_weights()[0][-1][0] for i in M_perf} #weights of the output layer connecting to diagnosis + patient effect
bias = {i:M[i].model.get_layer('output').get_weights()[1][0] for i in M_perf} #bias weight
alpha = {i: omega[i] + bias[i] for i in M_perf}

pd.DataFrame.from_dict(omega).to_csv(root+'omegas.csv')
pd.DataFrame.from_dict(alpha).to_csv(root+'alphas.csv')
pd.DataFrame(beta, index=[0]).to_csv(root+'beta.csv')
pd.DataFrame(bias, index=[0]).to_csv(root+'bias.csv')

omega = pd.read_csv(root+'omegas.csv', index_col=0).to_dict('list')
alpha = pd.read_csv(root+'alphas.csv', index_col=0).to_dict('list')
beta = pd.read_csv(root+'beta.csv', index_col=0).to_dict('records')[0]
bias = pd.read_csv(root+'bias.csv', index_col=0).to_dict('records')[0]

limit_plot = max(abs(np.amin(omega['nn'])), np.amax(omega['nn']))
PlotResults(root).plot_dist_performance(omega['nn'], 'omega', bias['nn'], beta['nn'], xlim=(-limit_plot, limit_plot))
PlotResults(root).plot_dist_performance(alpha['nn'], 'alpha', bias['nn'], beta['nn'], xlim=(-limit_plot+bias['nn'], limit_plot+bias['nn']))

# Comparison hospital-specific performance
PlotResults(root).plot_dist_difference(omega, 'hglm', 'nn', 'omega', 'HGLM', 'Proposed model')