# Estimating Risk-Adjusted Hospital Performance
This is the python code supplementary to the article **'Estimating Risk-Adjusted Hospital Performance'** by Eva van Weenen and Stefan Feuerriegel, accepted at the *2020 IEEE International Conference on Big Data*. It proposes a new method for measuring hospital performance, while adjusting for patient risk. The paper can be found on [ArXiv](https://arxiv.org/abs/2011.05149). 

The following code is used to preprocess the NRD, and to run the experiments.

## Overview of files in this repository
Quick overview of the python files.
* [config.py](../blob/main/config.py) Settings (such as filepaths, and hyperparameters)
* [data_pipeline.py](../blob/main/data_pipeline.py) Data preprocessing (and [run_data_pipeline.sh](../blob/main/run_data_pipeline.sh) to run it)
* [descriptives.py](../blob/main/descriptives.py) Descriptives of the data
* [main.py](../blob/main/main.py) Train and optimize models
* [metrics_callback.py](../blob/main/metrics_callback.py) Keras callback to calculate metrics on large dataset
* [models.py](../blob/main/models.py) Keras models
* [plot.py](../blob/main/plot.py) Plot functions

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

For the figures, latex needs to be installed. Else, comment this out in [plot.py](../blob/main/plot.py).

## Data
To run the preprocessing of the Nationwide Readmissions Database, run the script [data_pipeline.py](../blob/main/data_pipeline.py). 

Please make sure before doing so all of the files listed here are downloaded:
* The Nationwide readmissions database is not open source and has to be downloaded yourself. It is easiest if you put this in `data/NRD/`.
* The files under **Cohort inclusion/exclusion tables of [2]** and **Conversion of ICD-10 codes to CCS codes** are available on this github page in  `data/ccs/`. However, if you wish to download a newer version, please follow the instructions underneath. 
* The files under **Conversion of ICD-10 to ICD-9 codes** and **Medical embeddings of [3]** have to be cloned from the respective github pages, but this is automated in the script [run_data_pipeline.sh](../blob/main/run_data_pipeline.sh). Also, see instructions underneath.

The preprocessing of the NRD can be done using [data_pipeline.py](../blob/main/data_pipeline.py). Note that the NRD is quite big, so might likely not fit into your RAM. After preprocessing, the following files should be saved:
* `labels.csv` (for each patient whether a patient was readmitted or not)
* `diag_raw.csv` (for each patient a maximium of 35 diagnoses, where each diagnosis is an ICD-10 code)
* `diag.csv` (for each patient a maximum of 35 diagnoses, where each diagnosis is the index of the embedding of [3])
* `ccs.csv` (for each patient a maximum of 35 diagnoses, where each diagnoses is a CCS code)
* `patient.csv` (for each patient, data specific to that patient (age, gender, etc.))
* `hosp.npy` (for each patient, a one-hot encoding of the hospital they were admitted to)

To create descriptives of the dataset, run [descriptives.py](../blob/main/descriptives.py).  

### Nationwide Readmissions Database (NRD)
The data used for this project is the Nationwide Readmissions Database. This dataset is not open source, and has to be bought.

More information about this database can be found at https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddbdocumentation.jsp

Information about variables: https://www.hcup-us.ahrq.gov/db/nation/nrd/stats/FileSpecifications_NRD_2016_Core.TXT and: https://www.hcup-us.ahrq.gov/db/nation/nrd/nrddde.jsp

Put the downloaded file in `data/NRD/`.

### Conversion of ICD-10 codes to CCS codes
*[These files are already provided in this github repository.]*

Download 
* Diagnosis (ICD10-CM): https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp
* Procedure (ICD10-PCS): https://www.hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp

This will give you two files: `ccs_dx_icd10cm_2018_1.csv` and `ccs_pr_icd10pcs_2018_1.csv`. I copied these files into `data/ccs/`.

Note that [2] uses 2018 CCS codes.

### Cohort inclusion/exclusion tables of [2]
*[These files are already provided in this github repository.]*

Download zip file from https://www.qualitynet.org/files/5eaadcffe8ffc8001f999225?filename=2019_Readmission_Meas_Updates_Specs.zip *(qualitynet.org -> Hospitals - Inpatient -> Measures -> Readmission Measures -> Resources -> Archived Measure Methodology -> 2019 Readmission Measures Updates and Specifications Reports (05/01/20))*. From the zip-file extract the file [2019HWRSuppFile.xlsx](../blob/main/data/ccs/2019HWRSuppFile.xlsx), in which you will find cohort inclusion and exclusion tables as described in [2]. From this xls file, I manually copied the tables into csv files. For this I retrieved the following cohort inclusion/exclusion tables with CCS 2018 codes:
* [pr1_plan_ccs_pcs.csv](../blob/main/data/ccs/pr1_plan_ccs_pcs.csv) (CCS planned procedures 2018)
* [pr2_plan_ccs_cm.csv](../blob/main/data/ccs/pr2_plan_ccs_cm.csv) (CCS planned diagnoses (maintenance chemotherapy or rehabilitation care) 2018)
* [pr3_potplan_ccs_pcs.csv](../blob/main/data/ccs/pr3_potplan_ccs_pcs.csv) (CCS potentially planned procedures 2018)
* [pr3_potplan_icd10_pcs.csv](../blob/main/data/ccs/pr3_potplan_icd10_pcs.csv) (ICD10-PCS potentially planned procedures 2018)
* [pr4_acute_ccs_cm.csv](../blob/main/data/ccs/pr4_acute_ccs_cm.csv]) (CCS acute diagnoses & complications of care 2018)
* [pr4_acute_icd10_cm.csv](../blob/main/data/ccs/pr4_acute_icd10_cm.csv) (ICD10-CM acute diagnoses & complications of care 2018)

As well as the CCS 2018 codes for speciality cohorts:
* [cohort_cardiorespiratory_ccs_cm.csv](../blob/main/data/ccs/cohort_cardiorespiratory_ccs_cm.csv) (cardiorespiratory cohort)
* [cohort_cardiovascular_ccs_cm.csv](../blob/main/data/ccs/cohort_cardiovascular_ccs_cm.csv) (cardiovascular cohort)
* [cohort_medicine_ccs_cm.csv](../blob/main/data/ccs/cohort_medicine_ccs_cm.csv) (medicine cohort)
* [cohort_neurology_ccs_cm.csv](../blob/main/data/ccs/cohort_neurology_ccs_cm.csv) (neurology cohort)
* [cohort_surgery_icd10_pcs.csv](../blob/main/data/ccs/cohort_surgery_icd10_pcs.csv) (surgery/gynaecology cohort)
* [cohort_surgery_ccs_pcs.csv](../blob/main/data/ccs/cohort_surgery_ccs_pcs.csv) (surgery/gynaecology cohort)

All these files can be found in the `data/ccs/` folder.

### Conversion of ICD-10 to ICD-9 codes
Download the file ICD_9_10_d_v1.1.csv from https://github.com/AtlasCUMC/ICD10-ICD9-codes-conversion

```
wget https://github.com/AtlasCUMC/ICD10-ICD9-codes-conversion/raw/master/ICD_9_10_d_v1.1.csv
```

Put the downloaded file in `data/icd_conversion/`

### Medical embeddings of [3]
The medical embeddings of [3] for ICD-9 codes can be downloaded from https://github.com/clinicalml/embeddings. Download file `claims_codes_hs_300.txt.gz`, extract the file and put it in `data/embeddings/`. 

```
wget https://github.com/clinicalml/embeddings/raw/master/claims_codes_hs_300.txt.gz
gzip -d claims_codes_hs_300.txt.gz
mv claims_codes_hs_300.txt IDX_IPR_C_N_L_month_ALL_MEMBERS_fold1_s300_w20_ss5_hs_thr12.txt
```

By running [data_pipeline.py](../blob/main/data_pipeline.py) a file `icd2emb.csv` will be created, which is necessary when running the models later. 

## Model
Load the data and train the models using [main.py](../blob/main/main.py). This code calls all other python files ([config.py](../blob/main/config.py), [metrics_callback.py](../blob/main/metrics_callback.py), [clr_callback.py](../blob/main/clr_callback.py), [models.py](../blob/main/models.py), [plot.py](../blob/main/plot.py)). 

### Before running the main code
Make sure you've downloaded the Cyclical Learning Rate (CLR) callback [clr_callback.py](https://github.com/bckenstler/CLR/blob/master/clr_callback.py) from https://github.com/bckenstler/CLR. 

```
wget https://github.com/bckenstler/CLR/raw/master/clr_callback.py
```

Additionally, before running the code, you can change settings and filepaths in [config.py](../blob/main/config.py). The following settings can be changed there:
* The filepaths to the data and results
* Model names
* If you want to optimize the hyperparameters of a model, just train the model, or only test a pre-trained model.
* The hyperparameters for the neural networks, and the hyperparameter-grid to optimize. If you've set `optimize_models[_model_]` to `False`, the hyperparameter-grid will be ignored.
* Unoptimized hyperparameters related to the training process, such as epochs, batch size, etc.
* The ratio with which you want to split your dataset into a training, validation and test set.

### Running the main code
In the main code, the proposed model and four baselines are trained, see [1] Subsection IV.B. These names of the models in the paper correspond to the following names in the code:

Paper | Code
--- | ---
Hospital-mean | `hospital_only`
HGLM | `hglm`
Elastic-net | `lasso`
Fully non-linear | `black_box`
Proposed model | `nn`

Note that the `lasso` model in the code is called *Elastic-net* in the paper, due to the extra *L_2* regularization factor that all models have. This regularization was set to mimic the Bayesian prior that was used in [2].

After running the baselines, results will be saved in a structured way in the folder assigned to the variable `root`, as set in [config.py](../blob/main/config.py). More figures are saved than are published, so please have a look in this folder, if you want to do model diagnostics.

Note that the NRD is quite big. Specifically the file `hosp.npz` is quite troublesome, but necessary in this format for the last layer of the neural network. This required us to create some workarounds for the way data is given to the neural network, during training (e.g. `batch_generator`), but also during testing (e.g. [metrics_callback.py](../blob/main/metrics_callback.py)). 

Note that if you are not interested in training and testing the models `hglm` and `lasso`, please set `read_ccs` to false. This will save you a lot of time reading in the data.


## References
[1] E.G. van Weenen and S. Feuerriegel, "Estimating Risk-Adjusted Hospital Performance," in Proceedings of the 2020 IEEE International Conference on Big Data, 2020.

[2] Yale New Haven Health Services Corporation - Center for Outcomes Research & Evaluation (NHHSC/CORE), “2019 Hospital-Wide Read-mission Measure Updates and Specifications - Version 8.0,” Tech. Rep. March, 2019.

[3] Y. Choi, C. Y.-i. Chiu, and D. Sontag, “Learning Low-Dimensional Representations of Medical Concepts,” in AMIA Joint Summits on Translational Science proceedings, 2016, pp. 41–50

## Questions?
If you have any questions, please send an email to evanweenen@ethz.ch
