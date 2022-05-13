#!/bin/bash



python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_dev_uw'     device=1 mode='eval' source_name='sdoh_challenge' eval_subset='dev'  source_subset='uw'    model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"
python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_dev_mimic'  device=1 mode='eval' source_name='sdoh_challenge' eval_subset='dev'  source_subset='mimic' model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"
python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_test_uw'    device=1 mode='eval' source_name='sdoh_challenge' eval_subset='test' source_subset='uw'    model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"
python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_test_mimic' device=1 mode='eval' source_name='sdoh_challenge' eval_subset='test' source_subset='mimic' model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"


python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_predict' device=1 mode='predict' source_dir='/home/lybarger/data/social_determinants_challenge_text/' subset=None model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"
