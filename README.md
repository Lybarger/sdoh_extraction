# Social Determinants of Health

## BRAT Import
The annotated corpus in BRAT format can be import using the `step010_brat_import.py` script, for example:
```
python3 runs/step010_brat_import.py with source_name='sdoh_challenge' source_dir='/path_to_data/sdoh_corpus_challenge'
```


Quality checks can be performed on the imported BRAT corpus using the `step012_data_checks.py` script, for example:
```
python3 runs/step012_data_checks.py with source_name='sdoh_challenge'
```

## Extraction Model

### Training
Extraction models based on the mSpERT architecture can be trained using the `step111_multi_spert_train.py` script. There are many configurable parameters; however, below is some example usage:
```
python3 runs/step111_multi_spert_train.py with fast_run=False description='sdoh_challenge_e10_d02' source_name="sdoh_challenge" epochs=10  prop_drop=0.2 device=1
```
The trained model and relevant configuration files are saved in ""/path../analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save".


### Inference
The extraction models trained using the `step111_multi_spert_train.py` (see above) can be used for inference using `step112_multi_spert_infer.py`. There are two inference modes: *eval* and *predict*. *eval* is intended for evaluating the performance of the trained extractor on BRAT annotated data. *predict* is intended for generating predictions for unlabeled text.

#### Evaluation
Below is example usage for applying a trained extractor with data with supervised labels (BRAT):
```
python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_dev_uw' device=1 mode='eval' source_name='sdoh_challenge' eval_subset='dev'  source_subset='uw'    model_path="/path../analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"
```


#### Prediction
Below is example usage for applying a trained extractor to a directory of text (\*.txt) files. NOTE that the modell has only been trained and evaluated on social history section text. The directory of text files should be limited to social history section text, to avoid false positives.
```
python3 runs/step112_multi_spert_infer.py with fast_run=False description='sdoh_challenge_predict' device=1 mode='predict' source_dir='/path../dir_with_text_files/' subset=None model_path="/path../analyses/step111_multi_spert_train/train/sdoh_challenge_e10_d02/save"
```
