#!/bin/bash
mkdir data/icd_conversion
mkdir data/embeddings

cd data/icd_conversion
wget https://github.com/AtlasCUMC/ICD10-ICD9-codes-conversion/blob/master/ICD_9_10_d_v1.1.csv
cd ../embeddings
wget https://github.com/clinicalml/embeddings/raw/master/claims_codes_hs_300.txt.gz
gzip -d claims_codes_hs_300.txt.gz
mv claims_codes_hs_300.txt IDX_IPR_C_N_L_month_ALL_MEMBERS_fold1_s300_w20_ss5_hs_thr12.txt
cd ../..

echo "Make sure you have downloaded the NRD 2016, icd-10 to ccs code conversion files and cohort exclusion tables, before running the data_pipeline file."
python data_pipeline.py