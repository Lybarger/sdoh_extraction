# Social Determinants of Health Event Extraction

## Background
This repository trains and evaluates a deep learning information extraction model for extracting event-based representations of social determinants of health (SDOH) from clinical text. In previous work, we introduced the Social History Annotated Corpus (SHAC), which consists of clinical text with detailed event-based annotations for SDOH events such as alcohol, drug, tobacco, employment, and living situation [1]. SHAC was the benchmark gold standard dataset from the 2022 National NLP Clinical Challenges SDOH extraction task (n2c2/UW SDOH Challenge) [2]. In this repository, we introduce the Multi-label Span-based Entity and Relation Transformer (mSpERT), which we train and evaluate on SHAC [3]. This repository contains a pipeline for importing SHAC into a Python data structure, training mSpERT on SHAC, and evaluating mSpERT on SHAC.

mSpERT is an augmented version of the original [Span-based Entity and Relation Transformer](https://ebooks.iospress.nl/volumearticle/55116) (SpERT) model developed by Eberts and Ulges [4]. SpERT jointly extracts entities and relations using BERT with output layers that classify spans and predict span relations. SpERT's span-based architecture allows overlapping span predictions but only allows a single label to be assigned to each span; however, the SHAC annotations frequently assign multiple labels to a single span. To adapt SpERT to SHAC, we developed mSpERT. We added additional classification layers to SpERT to accommodate multi-label span label. Figure 1 presents the mSpERT framework, which includes three classification layers: 1) Entity Type, 2) Entity Subtype, and 3) Relation.  The Entity Type and Relation layers are identical to the original SpERT, and the Entity Subtype layer is incorporated to generate multi-label span predictions.  mSpERT was developed by cloning the [original SpERT GitHub repository](https://github.com/lavis-nlp/spert).

<img src="figures/spert_multilabel.drawio.png" width=75% height=75%>

__Figure 1: Multi-label Span-based Entity and Relation Transformer (mSpERT)__

## Data
SHAC consists of de-identified clinial text from MIMIC-III and the University of Washington (UW) that is annotated for SDOH. SHAC was used as gold standard data in the 2022 n2c2/UW SDOH Challenge. SHAC is currently only available to individuals that participated in the challenge; however, it will be made more broadly available starting in November 2023. Access to SHAC requires PhysioNet credentialing for the MIMIC-III data and a data use agreement for the UW data.
To access SHAC, please email both Kevin Lybarger ([klybarge@gmu.edu](mailto:klybarge@gmu.edu)) and Meliha Yetisgen ([melihay@uw.edu](mailto:melihay@uw.edu)).

## Trained Models
Trained mSpERT SDOH extraction models are available. To access the trained models, please email both Kevin Lybarger ([klybarge@gmu.edu](mailto:klybarge@gmu.edu)) and Meliha Yetisgen ([melihay@uw.edu](mailto:melihay@uw.edu)).

## Requirements
1. _mSpERT_ - This repository includes the code needed to load SHAC, process it into mSpERT format, and train and evaluate mSpERT. The mSpERT code is housed in a separate repository and should downloaded or cloned from: [https://github.com/Lybarger/mspert](https://github.com/Lybarger/mspert). 
2. _Evaluation_ - This repository utilizes the scoring routine from the n2c2/UW SDOH Challenge. Instructions for pip installing the scoring routine are available at:  [https://github.com/Lybarger/brat_scoring](https://github.com/Lybarger/brat_scoring).
3. _Requirements doc_ - A requirements document, [requirements.txt](requirements.txt), is included for the remaining dependancies.


## Pipeline

### BRAT Import
SHAC is annotated in BRAT format. The SHAC corpus can be imported into a Python-based data structure as follow:
```
python import_corpus.py --source /path/to/challenge/data/directory/ --output_file /path/to/corpus.pkl
```

### Extraction Model

#### Training
mSpERT can be trained on SHAC using the `train_mspert.py` script. Below is example useage:
`
python train_mspert.py --source_file /path/to/corpus.pkl  --destination /path/to/output/directory/ --mspert_path /path/to/mspert/directory/ --model_path "emilyalsentzer/Bio_ClinicalBERT" --tokenizer_path "emilyalsentzer/Bio_ClinicalBERT" --epochs 1 
`
The trained model and relevant configuration files are saved in "/path/to/output/directory/save/".


#### Inference
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
4. Eberts M, Ulges A. Span-Based Joint Entity and Relation Extraction with Transformer Pre-Training. In: European Conference on Artificial Intelligence; 2020. p. 2006-13. Available from: [https://ebooks.iospress.nl/volumearticle/55116](https://ebooks.iospress.nl/volumearticle/55116).

*Authors contributed equally.
