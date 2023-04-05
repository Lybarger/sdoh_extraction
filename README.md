# Social Determinants of Health Event Extraction
This repository is associated with our work exploring the extraction of social determinants of health (SDOH) from clinical text [1-3]. It utilizes the Social History Annotated Corpus (SHAC), which consists of clinical text with detailed event-based annotations for SDOH events such as alcohol, drug, tobacco, employment, and living situation. Each SDOH event is characterized through attributes related to status, extent, and temporality [1, 2].


This repository contains the Pipeline code for training and evaluating the mSpERT model. mSpERT is an augmented version of the original [Span-based Entity and Relation Transformer](https://ebooks.iospress.nl/volumearticle/55116) (SpERT) model developed by Eberts and Ulges [1]. SpERT jointly extracts entities and relations using BERT with output layers that classify spans and predict span relations. SpERT's span-based architecture allows overlapping span predictions but only allows a single label to be assigned to each span; however, annoataion schemes may assign multiple labels to a single span. We added additional classification layers to SpERT to accommodate multi-label span predictions. Figure 1 presents the mSpERT framework, which includes three classification layers: 1) Entity Type, 2) Entity Subtype, and 3) Relation.  The Entity Type and Relation layers are identical to the original SpERT, and the Entity Subtype layer is incorporated to generate multi-label span predictions. The example in Figure 1 is from out Social Determinant of Health (SDOH) event extraction work [2]. mSpERT was developed by cloning the [original SpERT GitHub repository](https://github.com/lavis-nlp/spert).

![Figure 1](figures/spert_multilabel.drawio.png)



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


## References
1. K. Lybarger, M. Ostendorf, and M. Yetisgen. Annotating social determinants of health using active learning, and characterizing determinants using neural event extraction. Journal Biomedical Informatics, 113:103631, 2021. doi: [10.1016/j.jbi.2020.103631](https://doi.org/10.1016/j.jbi.2020.103631).
2. K. Lybarger, M. Yetisgen, and Ö. Uzuner. The 2022 n2c2/UW shared task on extracting social determinants of health. Journal American Medical Informatics Association, 2023. doi: [10.1093/jamia/ocad012](https://doi.org/10.1093/jamia/ocad012).
3. K. Lybarger*, N. J. Dobbins*, R. Long, A. Singh, P.Wedgeworth, O. Ozuner, and M. Yetisgen. Leveraging natural language processing to augment structured social determinants of health data in the electronic health record. under review, 2023. doi: [10.48550/arXiv.2212.07538]().


*Authors contributed equally.
