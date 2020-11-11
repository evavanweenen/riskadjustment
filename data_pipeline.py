# -*- coding: utf-8 -*-
"""
Preprocessing of
Nationwide Readmissions Database 2016: 
https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddbdocumentation.jsp

Info columns: https://www.hcup-us.ahrq.gov/db/nation/nrd/stats/FileSpecifications_NRD_2016_Core.TXT
and: https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddde.jsp
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict
import gc
from config import datadir

n_diag = 36 ; n_diag += 1

def icd10_to_icd9(icd10):
    """
    Map ICD-10 to ICD-9 codes using conversion of:
    https://github.com/AtlasCUMC/ICD10-ICD9-codes-conversion
    """
    print("Converting ICD-10 to ICD-9 codes..")
    dictionary = pd.read_csv(datadir+'icd_conversion/ICD_9_10_d_v1.1.csv', delimiter = '|')
    
    dictionary['TargetI9'] = dictionary['TargetI9'].str.replace('.', '')
    dictionary['Flags'] = dictionary['Flags'].str.replace('.', '')
    
    for col in icd10.columns:
        icd10[col] = icd10[col].map(dictionary.set_index('TargetI9')['Flags'].to_dict())

    return icd10

def icd2ccs(data, primary_only=False):
    """
    Map ICD10 codes to CCS codes
    
    ICD10 to CCS code mappings for diagnoses (cm) and procedures (pcs)
    Diagnosis (ICD10-CM): https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp
    Procedure (ICD10-PCS): https://www.hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp
    """
    print("Converting ICD-10 to CCS codes..")
    # read icd10 to ccs mapping for diagnoses (cm) and procedures (pcs or pr)
    icd2ccs_cm = pd.read_csv(datadir+'ccs/ccs_dx_icd10cm_2018_1.csv', delimiter=',', usecols=(0,1), header=0, names=('icd10', 'ccs_cm')) 
    icd2ccs_pr = pd.read_csv(datadir+'ccs/ccs_pr_icd10pcs_2018_1.csv', delimiter=',', usecols=(0,1), header=0, names=('icd10', 'ccs_pr'))

    # preprocess icd10 to ccs mapping
    for i in range(len(icd2ccs_cm.columns)):
        icd2ccs_cm.iloc[:,i] = icd2ccs_cm.iloc[:,i].str.replace("'", "")
        icd2ccs_cm.iloc[:,i] = icd2ccs_cm.iloc[:,i].str.replace(" ", "")
    for i in range(len(icd2ccs_pr.columns)):
        icd2ccs_pr.iloc[:,i] = icd2ccs_pr.iloc[:,i].str.replace("'", "")
        icd2ccs_pr.iloc[:,i] = icd2ccs_pr.iloc[:,i].str.replace(" ", "")
    
    # convert icd10 to ccs mapping tables to dictionaries
    icd2ccs_cm = icd2ccs_cm.set_index('icd10')['ccs_cm'].to_dict() 
    icd2ccs_pr = icd2ccs_pr.set_index('icd10')['ccs_pr'].to_dict()
    
    if primary_only:
        diag_range = 2
        proc_range = 2
    else:
        diag_range = len(diagcols) + 1
        proc_range = len(proccols) + 1
    
    # map ICD10 diagnosis codes to CCS codes
    for i in range(1,diag_range):
        data['ccs_cm%s'%i] = data['diag%s'%i].map(icd2ccs_cm)
    
    # map ICD10 procedures codes to CCS codes
    for i in range(1,proc_range):
        data['ccs_pr%s'%i] = data['proc%s'%i].map(icd2ccs_pr)
    return data
 
def exclusion(data):
    """
    Perform this step both for the index admission and readmission
    """
    print("Exclude CCS codes for psychiatry, cancer and rehabilitation..")
    coh_excl_ccs_cm = pd.read_csv(datadir+'ccs/cohort_exclusions_ccs_cm.csv', delimiter=',', dtype=int) #CCS diagnosis codes for psychiatry, cancer or rehabilitation 2018

    # remove patients with psychiatry, cancer and rehabilitation CCS diagnoses
    data = data[~data['ccs_cm1'].isin(coh_excl_ccs_cm.astype(str))]

    return data

def exclusion_indexadmission(data):
    """
    Perform this step only for the INDEX ADMISSION
    """
    print("Remove patients who died or with invalid diagnoses..")
    # remove patients who died during index admission
    data = data[data['died'] == 0] 
    
    # remove patients with no diagnosis
    data = data[data['diag1'].notnull()]
    data = data[data['diag1'] != 'incn']
    data = data[data['diag1'] != 'invl']
      
    return data
    
def planned_procedures(data):
    """
    A visit is not considered a readmission if it is somehow planned
    -> Remove all planned from 'data'
    (following HWR 2019 report: Figure PR.1 - Planned Readmission Algorithm Flowchart for 2018 CCS codes)
    1. Exclude planned procedures (Table PR.1)
    2. Exclude planned diagnoses (Table PR.2)
    3. Exclude potentially planned procedures (Table PR.3) UNLESS principal discharge diagnosis of readmission is acute or complication of care occurred (Table PR.4)
    
    Note that the 2019 HWR report is used in which they use 2018 CCS codes, conform with the CCS code version used in this code
    """
    print("Remove planned procedures..")
    
    ccs_pr_plan = np.genfromtxt(datadir+'ccs/pr1_plan_ccs_pcs.csv', delimiter=',', dtype=int) #CCS planned procedures 2018
    ccs_cm_plan = np.genfromtxt(datadir+'ccs/pr2_plan_ccs_cm.csv', delimiter=',', dtype=int) #CCS planned diagnoses (maintenance chemotherapy or rehabilitation care) 2018
    
    ccs_pr_potplan = np.genfromtxt(datadir+'ccs/pr3_potplan_ccs_pcs.csv', delimiter=',', dtype=int) #CCS potentially planned procedures 2018 
    icd_pr_potplan = np.genfromtxt(datadir+'ccs/pr3_potplan_icd10_pcs.csv', delimiter=',', dtype='U8') #ICD10-PCS potentially planned procedures 2018
    
    ccs_cm_acute = np.genfromtxt(datadir+'ccs/pr4_acute_ccs_cm.csv', delimiter=',', dtype=int) #CCS acute diagnoses & complications of care 2018
    icd_cm_acute = np.genfromtxt(datadir+'ccs/pr4_acute_icd10_cm.csv', delimiter=',', dtype='U8') #ICD10-CM acute diagnoses & complications of care 2018
    icd_cm_acute = np.array([i.replace('.','') for i in icd_cm_acute]) #remove dots from icd strings

    # remove planned procedures
    data = data[~data['ccs_pr1'].isin(ccs_pr_plan.astype(str))]
    
    # remove always planned diagnoses
    data = data[~data['ccs_cm1'].isin(ccs_cm_plan.astype(str))]
    
    # remove potentially planned procedures if they are not acute
    potplan_mask = (data['ccs_pr1'].isin(ccs_pr_potplan.astype(str)) | data['proc1'].isin(icd_pr_potplan.astype(str)))
    acute_mask = (data['ccs_cm1'].isin(ccs_cm_acute.astype(str)) | data['diag1'].isin(icd_cm_acute.astype(str)))   
    data = data[~(potplan_mask & ~acute_mask)] 
    
    return data

def to_cohort(data):
    """
    Map CCS and ICD codes to cohorts according to 2019 Yale New-Haven report
    
    Note: Patients with surgery are moved to surgery cohort, regardless of their diagnosis
    """
    print("Divide patients into cohorts..")
    
    # CSS cohorts 2018 (from 2019 HWR report)
    coh_cr_ccs_cm = {'%s'%i:'cr' for i in np.genfromtxt(datadir+'ccs/cohort_cardiorespiratory_ccs_cm.csv', delimiter=',', dtype=int)} #cardiorespiratory cohort ccs
    coh_cv_ccs_cm = {'%s'%i:'cv' for i in np.genfromtxt(datadir+'ccs/cohort_cardiovascular_ccs_cm.csv', delimiter=',', dtype=int)} #cardiovascular cohort ccs
    coh_me_ccs_cm = {'%s'%i:'me' for i in np.genfromtxt(datadir+'ccs/cohort_medicine_ccs_cm.csv', delimiter=',', dtype=int)} #medicine cohort ccs
    coh_ne_ccs_cm = {'%s'%i:'ne' for i in np.genfromtxt(datadir+'ccs/cohort_neurology_ccs_cm.csv', delimiter=',', dtype=int)} #neurology cohort ccs
   
    coh_su_icd_pr = {'%s'%i:'su' for i in np.genfromtxt(datadir+'ccs/cohort_surgery_icd10_pcs.csv', delimiter=',', dtype='U8')} #surgery/gynaecology cohort icd10
    coh_su_ccs_pr = {'%s'%i:'su' for i in np.genfromtxt(datadir+'ccs/cohort_surgery_ccs_pcs.csv', delimiter=',', dtype=int)} #surgery/gynaecology cohort ccs

    # CCS and ICD to cohort mapping
    ccs_cm2cohort = {**coh_cr_ccs_cm, **coh_cv_ccs_cm, **coh_me_ccs_cm, **coh_ne_ccs_cm}
    ccs_pr2cohort = coh_su_ccs_pr
    icd_pr2cohort = coh_su_icd_pr
    
    # map to surgery category!!
    data['cohort'] = data['ccs_pr1'].map(ccs_pr2cohort)
    data['cohort'][data['cohort'].isna()] = data['proc1'][data['cohort'].isna()].map(icd_pr2cohort)

    # map to all other cohort categories (cardiorespiratory, cariovascular, medicine and neurology)
    data['cohort'][data['cohort'].isna()] = data['ccs_cm1'][data['cohort'].isna()].map(ccs_cm2cohort)

    # remove patients not belonging to any cohort
    data = data[data['cohort'].values.astype(str) != 'nan']
    
    return data    

def preprocess_embeddings():
    with open(datadir+'embeddings/IDX_IPR_C_N_L_month_ALL_MEMBERS_fold1_s300_w20_ss5_hs_thr12.txt', 'r') as fd:
        icds = [x.split()[0][4:] for x in fd.readlines() if 'IDX' in x]
    with open(datadir+'embeddings/IDX_IPR_C_N_L_month_ALL_MEMBERS_fold1_s300_w20_ss5_hs_thr12.txt', 'r') as fd:   
        vecs = [x.split()[1:] for x in fd.readlines() if 'IDX' in x]
        
    vecs_f = []
    for v in vecs:
        vecs_f.append([float(x) for x in v])
        
    df = pd.DataFrame(np.hstack([np.array(icds).reshape(-1,1), np.array(vecs_f).reshape(-1,300)]))
    df.iloc[:,0] = df.iloc[:,0].str.replace('.','')

    df = df.set_index(0)
    df = df.append(pd.DataFrame(np.random.uniform(low = -1, high = 1, size = (1, len(df.columns))),
                                index = ['UNK'],
                                columns = df.columns))

    df.to_csv(datadir+'embeddings/icd2emb.csv', header=None)


def icd2embedding_idx(data):
    """
    Map ICD-9 codes to the indices of medical embeddings of Choi et al. 2016
    """
    print("Convert ICD-9 codes to embeddings..")
    # read icd2embeddings files and create dictionary with icd codes linking to indices of this file
    preprocess_embeddings()
    icd2emb = pd.read_csv(datadir+'embeddings/icd2emb.csv', header=None)
    word2idx = dict(zip(icd2emb[0].values, icd2emb.index))

    # index icd9s for embeddings
    data[icd_cols] = icd10_to_icd9(data[icd_cols])

    for col in icd_cols:       
        data[col] = [word2idx.get(key) for key in data[col]]

    data[icd_cols] = data[icd_cols].fillna(word2idx.get('UNK'))
    
    return data

# Define columns
cols = {"age":0,                                            # age at admission (years)
        "died":2,                                           # died during hospitalization
        "key":53,                                           # unique record identifier
        "los":54,                                           # length of stay
        "visitlink":62,                                     # patient linkage variable in the NRD
        "payer":64,                                         #
        "daystoevent":60,                                   # days from "start date" to admission / admission "date" (days)
        "hosp_key":52,                                      # unique hospital identifier
        "gender":50,                                        # gender of patient
        "proc1":65, "proc2":66, "proc3":67,                 # procedures codes
        "charges":98}                                       #
diag_dict = {'diag%s'%i:i+9 for i in np.arange(1,36)}       # patient diagnoses
cols.update(diag_dict)
        
cols = OrderedDict(sorted(cols.items(), key=lambda kv: kv[1]))  #sort cols

diagcols = list(diag_dict.keys())
proccols = ['proc1', 'proc2', 'proc3']
patientcols = ['gender', 'age', 'charges', 'hosp_key', 'payer', 'cohort']
icd_cols = diagcols + proccols
ccs_cm_cols = ['ccs_cm%s'%i for i in range(1,len(diagcols)+1)]

# Read NRD
print("Read full NRD file..") 
nrd = pd.read_csv(datadir+'NRD/NRD_2016/NRD_2016_Core.CSV', header = None, usecols = cols.values(), names=cols.keys())
print("Full NRD file loaded")

# Create variables
nrd["eos"] = nrd["daystoevent"] + nrd["los"] #end of stay
nrd['readmit'] = 0 #fill readmit column with zeros

# Perform inclusion and exclusion criteria + map ICD-10 to CCS
nrd = exclusion_indexadmission(nrd)

nrd = icd2ccs(nrd)
nrd = exclusion(nrd) 

# Identify readmissions by chunks
# note that the following only adds the boolean information about readmission to the NRD file
chunksize = int(1e6)
counter = 0
print("Reading NRD file iteratively..")
for chunk in pd.read_csv(datadir+'NRD/NRD_2016/NRD_2016_Core.CSV', chunksize=chunksize, header = None, usecols = cols.values(), names=cols.keys()):
    print(counter)
    chunk = icd2ccs(chunk, primary_only=True)
    chunk = exclusion(chunk)
    chunk = planned_procedures(chunk)

    #identify readmissions by visitlink, put corresponding key and daystoevent in separate columns 
    chunk = pd.merge(nrd[['key', 'visitlink', 'daystoevent', 'eos', 'hosp_key']], chunk[["key", "visitlink", "daystoevent", 'hosp_key']], on = "visitlink") 
    chunk = chunk[chunk.key_x != chunk.key_y] #select only readmissions (not same admissions)
    chunk = chunk[chunk.daystoevent_y >=  chunk.eos] #select right order of visit (and remove claims with overlapping dates)
    chunk = chunk[chunk.daystoevent_y <=  chunk.eos+30] #select only readmissions within 30 days
    
    readmission_keys = chunk["key_x"].unique() #select unique admission keys from chunk 
   
    #set nrd.readmit to 1 if identified as readmission       
    readmit_indices = nrd[nrd['key'].isin(readmission_keys)].index #list row indices of readmissions
    nrd.loc[readmit_indices,'readmit'] = 1
    counter += 1

del chunk
gc.collect()
print("NRD file read iteratively and readmissions identified")

# Remove small hospitals to save memory
hosp_counts = nrd.groupby(by='hosp_key').count().iloc[:,0]
nrd = nrd[nrd['hosp_key'].isin(hosp_counts[hosp_counts>500].index.values)]

# Remove hospitals with no readmissions (NOTE: this does nothing)
hosp_readmit = nrd[['hosp_key', 'readmit']].groupby('hosp_key').mean().iloc[:,0]    
nrd = nrd[nrd['hosp_key'].isin(hosp_readmit[hosp_readmit>0].index.values)]

# Divide visits into cohorts
nrd = to_cohort(nrd)

# Reindex
nrd.index = list(range(len(nrd))) 

print("Save raw diagnosis file..")
nrd[diagcols].to_csv(datadir+'diag_raw.csv', header=False)

# Map ICD-9 to embedding indices
nrd = icd2embedding_idx(nrd)

# One-hot encoding hospital   
hosp = OneHotEncoder().fit_transform(nrd[["hosp_key"]].values)

print("Save hospital file..")
np.savez(datadir+'hosp', hosp.data, indices=hosp.indices, indptr=hosp.indptr, shape=hosp.shape)

# Save data into separate csv files
print("Save..")
nrd['readmit'].to_csv(datadir+'labels.csv', header=False)
nrd[diagcols].to_csv(datadir+'diag.csv', header=False)
nrd[ccs_cm_cols].to_csv(datadir+'ccs.csv', header=False)
nrd[patientcols].to_csv(datadir+'patient.csv') 

print("Save NRD backup")
nrd.to_csv(datadir+'nrd_2.csv')
print("Done")