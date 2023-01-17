#!/bin/bash



python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_e10_d02'  device=1 mode='eval' subdir='subtask_a' source_name='sdoh_challenge' eval_source='mimic' eval_subset='test'      save_brat=True   model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"

python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_e10_d02'  device=1 mode='eval' subdir='subtask_b' source_name='sdoh_challenge' eval_source='uw'    eval_subset='train_dev' save_brat=True   model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"

python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_e14_d02'  device=1 mode='eval' subdir='subtask_c' source_name='sdoh_challenge' eval_source='uw'    eval_subset='test'      save_brat=True   model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_c/sdoh_challenge_e14_d02/save/"



python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_e14_d02_VENV_CHECK'  device=1 mode='eval' subdir='subtask_c' source_name='sdoh_challenge' eval_source='uw'    eval_subset='test'      save_brat=True   model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_c/sdoh_challenge_e14_d02/save/"





python3 runs/step112_multi_spert_infer.py with fast_run=False description='Train_mimic_train_Eval_uw_train'  device=0 mode='eval' subdir='check' source_name='sdoh_challenge' eval_source='uw' eval_subset='train' save_brat=False model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"

python3 runs/step112_multi_spert_infer.py with fast_run=False description='Train_mimic_train_Eval_uw_dev'  device=0 mode='eval' subdir='check' source_name='sdoh_challenge' eval_source='uw' eval_subset='dev' save_brat=False model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"

python3 runs/step112_multi_spert_infer.py with fast_run=False description='Train_mimic_train_Eval_uw_train_dev'  device=0 mode='eval' subdir='check' source_name='sdoh_challenge' eval_source='uw' eval_subset='train_dev' save_brat=False model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"



python3 runs/step112_multi_spert_infer.py with fast_run=False description='Train_mimic_train_Eval_uw_test'  device=0 mode='eval' subdir='check' source_name='sdoh_challenge' eval_source='uw' eval_subset='test' save_brat=False model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"


python3 runs/step112_multi_spert_infer.py with fast_run=False description='Train_mimic_train_Eval_mimic_test'  device=0 mode='eval' subdir='check' source_name='sdoh_challenge' eval_source='mimic' eval_subset='test' save_brat=False model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"

python3 runs/step112_multi_spert_infer.py with fast_run=False description='Train_mimic_train_Eval_mimic_dev'  device=0 mode='eval' subdir='check' source_name='sdoh_challenge' eval_source='mimic' eval_subset='dev' save_brat=False model_path="/home/lybarger/sdoh_challenge/analyses/step111_multi_spert_train/subtask_a/sdoh_challenge_e16_d02/save/"