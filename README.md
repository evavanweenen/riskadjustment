# Estimating Risk-Adjusted Hospital Performance
This is the python code supplementary to the article **'Estimating Risk-Adjusted Hospital Performance'** by Eva van Weenen and Stefan Feuerriegel, accepted at the *2020 IEEE International Conference on Big Data*. It proposes a new method for measuring hospital performance, while adjusting for patient risk. The paper can be found at TODO ArXiv. 

The following code is used to preprocess the NRD, and to run the experiments.

## Overview of files in this repository
Quick overview of the python files.
* `config.py` Settings (such as filepaths, and hyperparameters)
* `data_pipeline.py` Data preprocessing (and `run_data_pipeline.sh` to run it)
* `descriptives.py` Descriptives of the data
* `main.py` Train and optimize models
* `metrics_callback.py` Keras callback to calculate metrics on large dataset
* `models.py` Keras models
* `plot.py` Plot functions

## Requirements and setup
This project was executed on a server with Intel(R) Xeon(R) Gold 6242 CPU @ 2.80GHz, 20 GB RAM and a TITAN V 12.6 GB GPU. 

The code for this project was executed in Python 3.6.9 and the following python packages were used:
* scipy 1.3.2
* numpy 1.17.4
* pandas 0.25.3
* tensorflow 1.12.0
* keras 2.2.4
* scikit-learn 0.21.3 
* matplotlib 3.3.2
* seaborn 0.9.0

Additionally, CUDA 9.2 was installed.

## Data
To run the preprocessing of the Nationwide Readmissions Database, run the script `data_pipeline.py`. 

Please make sure before doing so all of the files listed underneath are downloaded. 
* The files under **Cohort inclusion/exclusion tables of [1]** and **Conversion of ICD-10 codes to CCS codes** will be available on this github page in  `data/ccs/`. However, if you wish to download a newer version, please follow the instructions underneath. 
* The files under **Conversion of ICD-10 to ICD-9 codes** and **Medical embeddings of [2]** have to be cloned from the respective github pages, but this is automated in the script `run_data_pipeline.py`. 
* The Nationwide readmissions database has to be downloaded yourself. It is easiest if you put this in `data/NRD/`.

The preprocessing of the NRD can be done using `data_pipeline.py`. Note that the NRD is quite big, so might likely not fit into your RAM. After preprocessing, the following files should be saved:
* `labels.csv` (for each patient whether a patient was readmitted or not)
* `diag_raw.csv` (for each patient a maximium of 35 diagnoses, where each diagnosis is an ICD-10 code)
* `diag.csv` (for each patient a maximum of 35 diagnoses, where each diagnosis is the index of the embedding of [2])
* `ccs.csv` (for each patient a maximum of 35 diagnoses, where each diagnoses is a CCS code)
* `patient.csv` (for each patient, data specific to that patient (age, gender, etc.))
* `hosp.npy` (for each patient, a one-hot encoding of the hospital they were admitted to)

To create descriptives of the dataset, run `descriptives.py`.  

### Nationwide Readmissions Database (NRD)
The data used for this project is the Nationwide Readmissions Database. 

More information about this database can be found at https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddbdocumentation.jsp

Information about variables: https://www.hcup-us.ahrq.gov/db/nation/nrd/stats/FileSpecifications_NRD_2016_Core.TXT and: https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddde.jsp

Put the downloaded file in `data/NRD/`.

### Conversion of ICD-10 to ICD-9 codes
Download the file ICD_9_10_d_v1.1.csv from https://github.com/AtlasCUMC/ICD10-ICD9-codes-conversion

Put the downloaded file in `data/icd_conversion/`

### Conversion of ICD-10 codes to CCS codes
Download 
* Diagnosis (ICD10-CM): https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp
* Procedure (ICD10-PCS): https://www.hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp

This will give you two files: `ccs_dx_icd10cm_2018_1.csv` and `ccs_pr_icd10pcs_2018_1.csv`. I copied these files into `data/ccs/`.

Note that [1] uses 2018 CCS codes.

### Cohort inclusion/exclusion tables of [1]
Download zip file from https://www.qualitynet.org/files/5eaadcffe8ffc8001f999225?filename=2019_Readmission_Meas_Updates_Specs.zip *(qualitynet.org -> Hospitals - Inpatient -> Measures -> Readmission Measures -> Resources -> Archived Measure Methodology -> 2019 Readmission Measures Updates and Specifications Reports (05/01/20))*. From the zip-file extract the file `2019HWRSuppFile.xlsx`, in which you will find cohort inclusion and exclusion tables as described in [1]. From this xls file, I manually copied the tables into csv files. For this I retrieved the following cohort inclusion/exclusion tables with CCS 2018 codes:
* `pr1_plan_ccs_pcs.csv` (CCS planned procedures 2018)
* `pr2_plan_ccs_cm.csv` (CCS planned diagnoses (maintenance chemotherapy or rehabilitation care) 2018)
* `pr3_potplan_ccs_pcs.csv` (CCS potentially planned procedures 2018)
* `pr3_potplan_icd10_pcs.csv` (ICD10-PCS potentially planned procedures 2018)
* `pr4_acute_ccs_cm.csv` (CCS acute diagnoses & complications of care 2018)
* `pr4_acute_icd10_cm.csv` (ICD10-CM acute diagnoses & complications of care 2018)

As well as the CCS 2018 codes for speciality cohorts:
* `cohort_cardiorespiratory_ccs_cm.csv` (cardiorespiratory cohort)
* `cohort_cardiovascular_ccs_cm.csv` (cardiovascular cohort)
* `cohort_medicine_ccs_cm.csv` (medicine cohort)
* `cohort_neurology_ccs_cm.csv` (neurology cohort)
* `cohort_surgery_icd10_pcs.csv` (surgery/gynaecology cohort)
* `cohort_surgery_ccs_pcs.csv` (surgery/gynaecology cohort)

All these files can be found in the `data/ccs/` folder.

### Medical embeddings of [2]
The medical embeddings of [2] for ICD-9 codes can be downloaded from https://github.com/clinicalml/embeddings. Download file `claims_codes_hs_300.txt.gz`, extract the file and put it in `data/embeddings/`. By running `data_pipeline.py` a file `icd2emb.csv` will be created, which is necessary when running the models later. 

## Model
Load the data and train the model using `main.py`. This code calls all other python files (`config.py`, `metrics_callback.py`, `clr_callback.py`, `models.py`, `plot.py`). 

Make sure you've downloaded the CLR callback as described underneath. 

Additionally, you can change hyperparameter settings and filepaths in `config.py`.

### Cyclical Learning Rates callback
Download `clr_callback.py` from https://github.com/bckenstler/CLR

## References
[1] Yale New Haven Health Services Corporation - Center for Outcomes Research & Evaluation (NHHSC/CORE), “2019 Hospital-Wide Read-mission Measure Updates and Specifications - Version 8.0,” Tech. Rep. March, 2019.

[2] Y. Choi, C. Y.-i. Chiu, and D. Sontag, “Learning Low-Dimensional Representations of Medical Concepts,” in AMIA Joint Summits on Translational Science proceedings, 2016, pp. 41–50

## Questions?
If you have any questions, please send an email to evanweenen@ethz.ch