
from collections import OrderedDict, Counter
import numpy as np
import re

SPAN_ONLY = "Span_only"


NO_SUBTYPE = "no_subtype"
NO_CONCAT = "no_concat"
CONCAT_LOGITS = "concat_logits"
CONCAT_PROBS = "concat_probs"
LABEL_BIAS = "label_bias"


CORPUS_FILE = "corpus.pkl"
SCORES_FILE = 'scores.csv'
MODEL_FILE = "model.pkl"

TEXT_FILE_EXT = 'txt'
ANN_FILE_EXT = 'ann'
EVENT = "event"
RELATION = "relation"
TEXTBOUND = "textbound"
ATTRIBUTE = "attribute"

ENCODING = 'utf-8'

# BRAT files
ANNOTATION_CONF = 'annotation.conf'
VISUAL_CONF = 'visual.conf'
ATTRIBUTE_PATTERN = '{}_'


ENTITIES = 'entities'
ENTITY = "entity"
ENTITIES_SUMMARY = 'entities_summary'
RELATIONS = 'relations'
RELATIONS_SUMMARY = 'relations_summary'
DOC_LABELS = 'doc_labels'
DOC_LABELS_SUMMARY = "doc_labels_summary"
SENT_LABELS = "sent_labels"
SENT_LABELS_SUMMARY = "sent_labels_summary"
EVENTS = "events"
COUNT = "count"
LABELED = "Labeled"
EVENT_TYPE = "event_type"
ARGUMENTS = "arguments"
START = "start"
END = "end"
TOKENS = "token"
HEAD = "head"
TAIL = "tail"

EXACT = "exact"
PARTIAL = "partial"
OVERLAP = "overlap"
LABEL = "label"

NONE = 'none'
ID = "id"
SUBSET = "subset"

TRIGGER = 'Trigger'

TYPE = "type"
SUBTYPE = "subtype"
ARG_1 = 'argument 1'
ARG_2 = 'argument 2'
ROLE = 'role'
SUBTYPE_A = "subtype_a"
SUBTYPE_B = "subtype_b"
ROLE = "role"
TRIGGER_SUBTYPE = "trigger_subtype"
ENTITY_SUBTYPE = "entity_subtype"


TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
QC = 'qc'
CV = 'cv'
CV_PREDICT = 'cv_predict'
CV_TUNE = 'cv_tune'
PREDICT = 'predict'
FIT = 'fit'
SCORE = 'score'
PROB = 'prob'


NT = 'NT'
NP = 'NP'
TP = 'TP'
TN = 'TN'
FP = 'FP'
FN = 'FN'
P = 'P'
R = 'R'
F1 = 'F1'
MICRO = 'micro'
ANNOTATOR_A = 'annotator A'
ANNOTATOR_B = 'annotator B'
METRIC = "metric"

PARAMS_FILE = 'parameters.pkl'
STATE_DICT = 'state_dict.pt'
PREDICTIONS_FILE = 'predictions.pkl'
PREDICTIONS_JSON = 'predictions.json'

SUBTYPE_DEFAULT = "none"

SPACY_MODEL = 'en_core_web_sm'

ANATOMY = 'Anatomy'

LESION_FINDING = "Lesion-Description"
LESION_ANATOMY = "Lesion-Anatomy"
LESION_SIZE = "Lesion-Size"
LESION_SIZE_TREND = "Lesion-Size-Trend"
LESION_COUNT = "Lesion-Count"
LESION_MALIGNANCY = "Lesion-Malignancy"
LESION_METASTASIS = "Lesion-Metastasis"
LESION_NEOPLASM = "Lesion-Neoplasm"
LESION_ASSERTION = "Lesion-Assertion"
LESION_CHARACTERISTIC = "Lesion-Characteristic"

LESION_SPAN_ONLY_ARGUMENTS = [LESION_ANATOMY, LESION_SIZE, LESION_COUNT, LESION_CHARACTERISTIC]

MEDICAL_PROBLEM = "Medical-Problem"
MEDICAL_ANATOMY = "Medical-Anatomy"
MEDICAL_ASSERTION = "Medical-Assertion"
MEDICAL_COUNT = "Medical-Count"

MEDICAL_SPAN_ONLY_ARGUMENTS = [MEDICAL_ANATOMY, MEDICAL_COUNT]

INDICATION_DESCRIPTION = "Indication-Description"

EVENT_TYPES = [LESION_FINDING, MEDICAL_PROBLEM, INDICATION_DESCRIPTION]
SPAN_ONLY_ARGUMENTS = LESION_SPAN_ONLY_ARGUMENTS + MEDICAL_SPAN_ONLY_ARGUMENTS

FINDING = 'Finding'



INCIDENTAL = 'incidental'


ANATOMY_SUBTYPES = '''Abdomen
Adrenal_gland
Back
Bile_Duct
Bladder
Brain
Breast
Cardiovascular_system
Diaphragm
Digestive_system
Ear
Esophagus
Eye
Fallopian_tube
Gallbladder
Head
Heart
Integumentary_system
Intestine
Kidney
Laryngeal
Liver
Lower_limb
Lung
Lymphatic_system
Mediastinum
Mouth
Musculoskeletal_system
Nasal_sinus
Neck
Nervous_system
Nose
Ovary
Pancreas
Pelvis
Penis
Pericardial_sac
Peritoneal_sac
Pharynx
Pleural_sac
Prostate
Retroperitoneal
Seminal_vesicle
Spleen
Stomach
Testis
Thorax
Thyroid
Tracheobronchial
Upper_limb
Urethra
Uterus
Vagina
Vas_deferens
Vulva
Whole_body
'''.splitlines()

ANATOMY_ABBREVIATION_PATS = [('_', ' '), (' ?system', ""), ("Cardiovascular", 'Cardio'), ('Musculoskeletal', 'MSK')]
ANATOMY_ABBREVIATION = {v: v for v in ANATOMY_SUBTYPES}
for pat, rep in ANATOMY_ABBREVIATION_PATS:
    ANATOMY_ABBREVIATION = {k: re.sub(pat, rep, v) for k, v in ANATOMY_ABBREVIATION.items()}

DARK_RED = tuple(np.array([184,    84, 80])/255)
DARK_BLUE = tuple(np.array([108,  142, 191])/255)
DARK_GOLD = tuple(np.array([224,  172, 46])/255)
DARK_GREEN = tuple(np.array([127, 177, 98])/255)
DARK_GRAY = tuple(np.array([102, 102, 102])/255)
DARK_ORANGE = tuple(np.array([215, 155, 0])/255)
ALMOST_BLACK = tuple(np.array([48, 48, 48])/255)

#Medical-Anatomy
#Lesion-Anatomy
