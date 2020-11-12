import numpy as np
import pandas as pd

from keras.callbacks import Callback

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve, precision_recall_curve, auc

def calc_metrics(x, y, x_val, y_val, model, metrics, epoch):
    def threshold(Y, Y_pred):
        # determine classification threshold
        fpr, tpr, thres = roc_curve(Y, Y_pred)
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=np.arange(len(tpr)) ), 'threshold' : pd.Series(thres, index=np.arange(len(tpr)))})
        t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['threshold'].values[0]
        return t
    
    # read dataset iteratively (size n)
    y_pred = model.predict(x)
    y_pred_val = model.predict(x_val)
    
    t = threshold(y, y_pred) 
    
    y_label = (y_pred > t).astype(np.int)
    y_label_val = (y_pred_val > t).astype(np.int)

    metrics.loc[epoch,'precision'] = precision_score(y, y_label, labels=[0,1])
    metrics.loc[epoch,'val_precision'] = precision_score(y_val, y_label_val, labels=[0,1])
    
    metrics.loc[epoch,'recall'] = recall_score(y, y_label, labels=[0,1])
    metrics.loc[epoch,'val_recall'] = recall_score(y_val, y_label_val, labels=[0,1])
    
    metrics.loc[epoch,'f1'] = f1_score(y, y_label, labels=[0,1])
    metrics.loc[epoch,'val_f1'] = f1_score(y_val, y_label_val, labels=[0,1])
    
    metrics.loc[epoch,'bce'] = log_loss(y, y_pred)
    metrics.loc[epoch,'val_bce'] = log_loss(y_val, y_pred_val)
    
    metrics.loc[epoch,'roc-auc'] = roc_auc_score(y, y_pred)
    metrics.loc[epoch,'val_roc-auc'] = roc_auc_score(y_val, y_pred_val)
    
    precision, recall, _ = precision_recall_curve(y, y_pred)
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_pred_val)
    metrics.loc[epoch,'pr-auc'] = auc(recall, precision)
    metrics.loc[epoch,'val_pr-auc'] = auc(recall_val, precision_val)
    
    print("bce: {:7.4f} - precision: {:7.4f} - recall: {:7.4f} - f1: {:7.4f} - roc-auc: {:7.4f} - pr-auc: {:7.4f} ".format(metrics['bce'][epoch], metrics['precision'][epoch], metrics['recall'][epoch], metrics['f1'][epoch], metrics['roc-auc'][epoch], metrics['pr-auc'][epoch]))
    print("val_bce: {:7.4f} - val_precision: {:7.4f} - val_recall: {:7.4f} - val_f1: {:7.4f} - val_roc-auc: {:7.4f} - val_pr-auc: {:7.4f} ".format(metrics['val_bce'][epoch], metrics['val_precision'][epoch], metrics['val_recall'][epoch], metrics['val_f1'][epoch], metrics['val_roc-auc'][epoch], metrics['val_pr-auc'][epoch]))
    return metrics

class Metrics(Callback):
    """
    Calculate classification performance metrics at the end of each epoch, using the entire validation and training data sets
    (Note that keras does not do this by default, it only calculates metrics on batches)
    Examples: https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
    https://github.com/keras-team/keras/issues/3230#issuecomment-319208366
    """
    
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
    def on_train_begin(self, logs={}):
        self.metrics = pd.DataFrame(columns=['precision','recall','f1','bce','roc-auc','pr-auc','val_precision','val_recall','val_f1','val_bce','val_roc-auc','val_pr-auc'])
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.metrics = calc_metrics(self.x, self.y, self.x_val, self.y_val, self.model, self.metrics, epoch)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return