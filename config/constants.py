
from collections import OrderedDict, Counter
import numpy as np
import re



def ATTR_TYPE_MAP(x):
    return f"{x}Val"

OVERALL = "OVERALL"

LABEL_DEFINITION_FILE = "label_definition.pkl"

UW = 'uw'
MIMIC = 'mimic'

'''
Determinants
'''
ALCOHOL = 'Alcohol'
COUNTRY = 'Country'
DRUG = 'Drug'
EMPLOYMENT = 'Employment'
ENVIRO_EXPOSURE = 'EnviroExposure'
GENDER_ID = 'GenderID'
INSURANCE = 'Insurance'
LIVING_STATUS = 'LivingStatus'
PHYS_ACTIVITY = 'PhysActivity'
RACE = 'Race'
SEXUAL_ORIENT = 'SexualOrient'
TOBACCO = 'Tobacco'


'''
Entities
'''

TRIGGER = 'Trigger'

STATUS                  = "Status"


# Span and class - new
STATUS_TIME             = 'StatusTime'
STATUS_TIME_VAL         = 'StatusTimeVal'

DEGREE                  = 'Degree'
DEGREE_VAL              = 'DegreeVal'

STATUS_EMPLOY           = 'StatusEmploy'
STATUS_EMPLOY_VAL       = 'StatusEmployVal'

STATUS_INSURE           = 'StatusInsure'
STATUS_INSURE_VAL       = 'StatusInsureVal'

TYPE_GENDER_ID          = 'TypeGenderID'
TYPE_GENDER_ID_VAL      = 'TypeGenderIDVal'

TYPE_LIVING             = 'TypeLiving'
TYPE_LIVING_VAL         = 'TypeLivingVal'

TYPE_SEXUAL_ORIENT      = 'TypeSexualOrient'
TYPE_SEXUAL_ORIENT_VAL  = 'TypeSexualOrientVal'



# Span only - new
AMOUNT      = 'Amount'
DURATION    = 'Duration'
FREQUENCY   = 'Frequency'
HISTORY     = 'History'
METHOD      = 'Method'
TYPE        = 'Type'

EMPLOYED = "employed"
UNEMPLOYED = "unemployed"
RETIRED = "retired"
ON_DISABILITY = "on_disability"
STUDENT = "student"
HOMEMAKER = "homemaker"
YES = "yes"
NO = "no"
NONE = "none"
CURRENT = "current"
PAST = "past"
FUTURE = "future"
TRANSGENDER = "transgender"
CISGENDER = "cisgender"
ALONE = "alone"
WITH_FAMILY = "with_family"
WITH_OTHERS = "with_others"
HOMELESS = 'homeless'
HOMOSEXUAL="homosexual"
BISEXUAL = "bisexual"
HETEROSEXUAL = "heterosexual"

EVENT_TYPES = [ALCOHOL, DRUG, TOBACCO, LIVING_STATUS, EMPLOYMENT]

LABELED_ARGUMENTS = [STATUS_TIME, STATUS_EMPLOY, TYPE_LIVING]
SPAN_ONLY_ARGUMENTS = [AMOUNT, DURATION, FREQUENCY, HISTORY, METHOD, TYPE]

ARGUMENT_TYPES = [AMOUNT, DURATION, FREQUENCY, HISTORY, METHOD, STATUS, TYPE]

#ENTITY_TYPES = EVENT_TYPES + ARGUMENTS

# SUBTYPES_BY_ARGUMENT = {}
# SUBTYPES_BY_ARGUMENT[STATUS_TIME] = [NONE, CURRENT, PAST, FUTURE]
# SUBTYPES_BY_ARGUMENT[STATUS_EMPLOY] = [EMPLOYED, UNEMPLOYED, RETIRED, ON_DISABILITY, STUDENT, HOMEMAKER]
# SUBTYPES_BY_ARGUMENT[TYPE_LIVING] = [ALONE, WITH_FAMILY, WITH_OTHERS]



STATUS_TIME_CLASSES = [NONE, CURRENT, PAST, FUTURE]
STATUS_EMPLOY_CLASSES = [EMPLOYED, UNEMPLOYED, RETIRED, ON_DISABILITY, STUDENT, HOMEMAKER]
TYPE_LIVING_CLASSES = [ALONE, WITH_FAMILY, WITH_OTHERS, HOMELESS]

SUBTYPES =  STATUS_TIME_CLASSES + STATUS_EMPLOY_CLASSES + TYPE_LIVING_CLASSES


ARGUMENTS_BY_EVENT_TYPE = {}
ARGUMENTS_BY_EVENT_TYPE[ALCOHOL] =       [STATUS_TIME, TYPE, DURATION, HISTORY, METHOD, AMOUNT, FREQUENCY]
ARGUMENTS_BY_EVENT_TYPE[DRUG] =          ARGUMENTS_BY_EVENT_TYPE[ALCOHOL]
ARGUMENTS_BY_EVENT_TYPE[TOBACCO] =       ARGUMENTS_BY_EVENT_TYPE[ALCOHOL]
ARGUMENTS_BY_EVENT_TYPE[LIVING_STATUS] = [STATUS_TIME, TYPE_LIVING, DURATION, HISTORY]
ARGUMENTS_BY_EVENT_TYPE[EMPLOYMENT] =    [STATUS_EMPLOY, DURATION, HISTORY, TYPE]

ARGUMENT2ROLE = {}
ARGUMENT2ROLE[STATUS_TIME] = STATUS
ARGUMENT2ROLE[STATUS_EMPLOY] = STATUS
ARGUMENT2ROLE[TYPE_LIVING] = TYPE


# SUBTYPES = [v for k, V in SUBTYPES_BY_ARGUMENT.items() for v in V]

REQUIRED_ARGUMENTS = OrderedDict()
REQUIRED_ARGUMENTS[ALCOHOL] = [STATUS]
REQUIRED_ARGUMENTS[DRUG] = [STATUS]
REQUIRED_ARGUMENTS[TOBACCO] = [STATUS]
REQUIRED_ARGUMENTS[LIVING_STATUS] = [TYPE]
REQUIRED_ARGUMENTS[EMPLOYMENT] = [STATUS]


SUBTYPE_DEFAULT = "None"



SPAN_ONLY = "Span_only"


NO_SUBTYPE = "no_subtype"
NO_CONCAT = "no_concat"
CONCAT_LOGITS = "concat_logits"
CONCAT_PROBS = "concat_probs"
LABEL_BIAS = "label_bias"


CORPUS_FILE = "corpus.pkl"
SCORES_FILE = 'scores.csv'
MODEL_FILE = "model.pt"

TEXT_FILE_EXT = 'txt'
ANN_FILE_EXT = 'ann'
EVENT = "event"
RELATION = "relation"
TEXTBOUND = "textbound"
ATTRIBUTE = "attribute"

ENCODING = 'utf-8'

NA = 'N/A'

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
ARGUMENT = "argument"
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
MIN_DIST = 'min_dist'

NONE = 'none'
ID = "id"
SUBSET = "subset"

TRIGGER = 'Trigger'

# TYPE = "type"
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
TRAIN_DEV = 'train_dev'
EVAL = 'eval'
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

SUBTYPE_DEFAULT = "None"



SPACY_MODEL = 'en_core_web_sm'

MIMIC = "mimic"

DARK_RED = tuple(np.array([184,    84, 80])/255)
DARK_BLUE = tuple(np.array([108,  142, 191])/255)
DARK_GOLD = tuple(np.array([224,  172, 46])/255)
DARK_GREEN = tuple(np.array([127, 177, 98])/255)
DARK_GRAY = tuple(np.array([102, 102, 102])/255)
DARK_ORANGE = tuple(np.array([215, 155, 0])/255)




'''
Labeling
'''
BEGIN = 'B-'
INSIDE = 'I-'
OUTSIDE = 'O'
UNKNOWN = 'unknown'
NEG_LABEL = 0


START_TOKEN = '<s>'
END_TOKEN = '</s>'
UNK = '<unk>'














TRIGGER = "Trigger"
LAB_TYPE = 'label_type'
ARGUMENT = 'Argument'
LAB_DEF = 'label_definition'
LAB_MAP = 'label_map'
SEQ = 'sequence'
SENT = 'sentence'


MULTITASK_LABEL_DEF = OrderedDict()
MULTITASK_LABEL_DEF[TRIGGER] = OrderedDict()
MULTITASK_LABEL_DEF[TRIGGER][LAB_TYPE] = SENT
MULTITASK_LABEL_DEF[TRIGGER][ARGUMENT] = TRIGGER
MULTITASK_LABEL_DEF[TRIGGER][LAB_MAP] = OrderedDict()
MULTITASK_LABEL_DEF[TRIGGER][LAB_MAP][ALCOHOL] =       [OUTSIDE, ALCOHOL]
MULTITASK_LABEL_DEF[TRIGGER][LAB_MAP][DRUG] =          [OUTSIDE, DRUG]
MULTITASK_LABEL_DEF[TRIGGER][LAB_MAP][TOBACCO] =       [OUTSIDE, TOBACCO]
MULTITASK_LABEL_DEF[TRIGGER][LAB_MAP][LIVING_STATUS] = [OUTSIDE, LIVING_STATUS]
MULTITASK_LABEL_DEF[TRIGGER][LAB_MAP][EMPLOYMENT] =    [OUTSIDE, EMPLOYMENT]

# Label map-status
MULTITASK_LABEL_DEF[STATUS_TIME] = OrderedDict()
MULTITASK_LABEL_DEF[STATUS_TIME][LAB_TYPE] = SENT
MULTITASK_LABEL_DEF[STATUS_TIME][ARGUMENT] = STATUS_TIME
MULTITASK_LABEL_DEF[STATUS_TIME][LAB_MAP] = OrderedDict()
MULTITASK_LABEL_DEF[STATUS_TIME][LAB_MAP][ALCOHOL] =       [OUTSIDE, NONE, PAST, CURRENT]
MULTITASK_LABEL_DEF[STATUS_TIME][LAB_MAP][DRUG] =          [OUTSIDE, NONE, PAST, CURRENT]
MULTITASK_LABEL_DEF[STATUS_TIME][LAB_MAP][TOBACCO] =       [OUTSIDE, NONE, PAST, CURRENT]
MULTITASK_LABEL_DEF[STATUS_TIME][LAB_MAP][LIVING_STATUS] = [OUTSIDE, NONE, PAST, CURRENT]

MULTITASK_LABEL_DEF[STATUS_EMPLOY] = OrderedDict()
MULTITASK_LABEL_DEF[STATUS_EMPLOY][LAB_TYPE] = SENT
MULTITASK_LABEL_DEF[STATUS_EMPLOY][ARGUMENT] = STATUS_EMPLOY
MULTITASK_LABEL_DEF[STATUS_EMPLOY][LAB_MAP] = OrderedDict()
MULTITASK_LABEL_DEF[STATUS_EMPLOY][LAB_MAP][EMPLOYMENT] =    [OUTSIDE, UNEMPLOYED, ON_DISABILITY, HOMEMAKER, STUDENT, RETIRED, EMPLOYED]

MULTITASK_LABEL_DEF[TYPE_LIVING] = OrderedDict()
MULTITASK_LABEL_DEF[TYPE_LIVING][LAB_TYPE] = SENT
MULTITASK_LABEL_DEF[TYPE_LIVING][ARGUMENT] = TYPE_LIVING
MULTITASK_LABEL_DEF[TYPE_LIVING][LAB_MAP] = OrderedDict()
MULTITASK_LABEL_DEF[TYPE_LIVING][LAB_MAP][LIVING_STATUS] = [OUTSIDE, ALONE, WITH_FAMILY, WITH_OTHERS, HOMELESS]

# Label map-entity
MULTITASK_LABEL_DEF[ENTITY] = OrderedDict()
MULTITASK_LABEL_DEF[ENTITY][LAB_TYPE] = SEQ
MULTITASK_LABEL_DEF[ENTITY][LAB_MAP] = OrderedDict()
MULTITASK_LABEL_DEF[ENTITY][LAB_MAP][ALCOHOL] =       [OUTSIDE, DURATION, HISTORY, TYPE, AMOUNT, FREQUENCY, METHOD]
MULTITASK_LABEL_DEF[ENTITY][LAB_MAP][DRUG] =          [OUTSIDE, DURATION, HISTORY, TYPE, AMOUNT, FREQUENCY, METHOD]
MULTITASK_LABEL_DEF[ENTITY][LAB_MAP][TOBACCO] =       [OUTSIDE, DURATION, HISTORY, TYPE, AMOUNT, FREQUENCY, METHOD]
MULTITASK_LABEL_DEF[ENTITY][LAB_MAP][LIVING_STATUS] = [OUTSIDE, DURATION, HISTORY]
MULTITASK_LABEL_DEF[ENTITY][LAB_MAP][EMPLOYMENT] =    [OUTSIDE, DURATION, HISTORY, TYPE]
