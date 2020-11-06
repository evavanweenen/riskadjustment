import numpy as np
import pandas as pd
import os
import gc
from plot import PlotData
from config import datadir, root

savedir = root + 'descriptives/'
if not os.path.exists(savedir):
    os.mkdir(savedir)

#read data
y = pd.read_csv(datadir+'labels.csv', index_col=0, header=None)
y.reset_index(inplace=True, drop=True)

patient = pd.read_csv(datadir+'patient.csv', index_col=0)
patient.reset_index(inplace=True, drop = True)
patient['readmit'] = y ; del y ; gc.collect()  

diag_raw = pd.read_csv(datadir+'diag_raw.csv', index_col=0, header=None)
patient['diagnoses'] = diag_raw.notnull().sum(axis=1)

#---------------------------------------- patient summary --------------------------------------------
def collapse(x):
    d = {}
    d['patients'] = x['hosp_key'].count()
    d['gender'] = x['gender'].mean()
    d['age_mean'] = x['age'].mean()
    d['age_std'] = x['age'].std()
    d['diagnoses_mean'] = x['diagnoses'].mean()
    d['diagnoses_std'] = x['diagnoses'].std()
    d['readmission'] = x['readmit'].mean()*100
    return pd.Series(d, index = ['patients', 'gender', 'age_mean', 'age_std', 'diagnoses_mean', 'diagnoses_std', 'readmission'])#'charges', 'readmission'])

patient_summary = patient.groupby(['cohort']).apply(collapse)
total = pd.DataFrame([[np.sum(patient_summary.patients), 
                       np.mean(patient.gender),
                       np.mean(patient.age),
                       np.std(patient.age),
                       np.mean(patient.diagnoses),
                       np.std(patient.diagnoses),
                       np.mean(patient.readmit)*100]], 
                       columns = ['patients', 'gender', 'age_mean', 'age_std', 'diagnoses_mean', 'diagnoses_std', 'readmission'])#'charges','readmission'])

patient_summary = patient_summary.append(total).round({'patients':0, 'gender':3, 'age_mean':1, 'age_std':1, 'diagnoses_mean':0, 'diagnoses_std':0, 'readmission':1})#'charges':0,'readmission':1})

patient_summary.to_latex(savedir+'patient_summary_full.tex') #TODO

sns.boxplot(x='readmit', y='age', data=patient)

#---------------------------------------- diagnoses --------------------------------------------
# plot cdf of patient diagnoses
PlotData(savedir).plot_cdf(patient['diagnoses'], 'Number of diagnoses', 'diagnoses')

#---------------------------------------- most frequent conditions --------------------------------------------
diags = diag_raw.values.flatten()
diags = pd.DataFrame(diags)
diags.dropna(inplace=True)
diags = diags.astype('category')
diags[0].value_counts().head(10).to_latex(savedir+'diagnoses_mostfrequent.tex') #TODO