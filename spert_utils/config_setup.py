
from collections import Counter, OrderedDict
import json
import logging
import pandas as pd
import os
from pathlib import Path

PREDICTION_FILE_PATTERN = 'prediction*.json'
MODEL_DIRECTORY_PATTERN = "final_model"
MODEL_FILE_PATTERN = "pytorch_model.bin"

NON_EVAL_ARGS = ["train_path", "valid_path", "log_path", "save_path",
                "train_batch_size", "neg_entity_count", "neg_relation_count", "epochs", "lr", "lr_warmup",
                "weight_decay", "max_grad_norm", "final_eval"]
CONFIG_MAP = {}
CONFIG_MAP["model_path"] = "save_path"
CONFIG_MAP["tokenizer_path"] = "save_path"

from spert_utils.spert_io import ID, TOKENS, OFFSETS, ENTITIES, RELATIONS, SUBTYPES, TYPE, START, END, HEAD, TAIL



def dict_to_config_file(params, path):

    print(params)
    print(path)

    x = []
    x.append('[1]')
    for k, v in params.items():
            x.append(f"{k} = {v}")
    x = "\n".join(x)


    with open(path, 'w') as f:
        f.write(x)

    return x

# def create_event_types_path(entities, subtypes, relations, path, sent_labels=None):
#
#     '''
#     {"entities":
#       {
#         "Task": {"short": "Task", "verbose": "Task"},
#         "Generic": {"short": "Generic", "verbose": "Generic"}
#       },
#     "relations":
#       {
#         "Used-for": {"short": "Used-for", "verbose": "Used-for", "symmetric": false},
#         "Conjunction": {"short": "Conjunction", "verbose": "Conjunction", "symmetric": true}
#         }
#       }
#       '''
#
#     d = {}
#     d["entities"] = {}
#     for x in entities:
#         d["entities"][x] = {"short": x, "verbose": x}
#
#     d["subtypes"] = {}
#     for x in subtypes:
#         d["subtypes"][x] = {"short": x, "verbose": x}
#
#     d["relations"] = {}
#     for x in relations:
#         d["relations"][x] = {"short": x, "verbose": x,  "symmetric": False}
#
#     if sent_labels is not None:
#         d["sent_labels"] = sent_labels
#
#     json.dump(d, open(path, 'w'), indent=4)
#
#     return d


def create_event_types_path(entities, subtypes, relations, sent_labels=None, path=None):

    '''
    {"entities":
      {
        "Task": {"short": "Task", "verbose": "Task"},
        "Generic": {"short": "Generic", "verbose": "Generic"}
      },
    "relations":
      {
        "Used-for": {"short": "Used-for", "verbose": "Used-for", "symmetric": false},
        "Conjunction": {"short": "Conjunction", "verbose": "Conjunction", "symmetric": true}
        }
      }
      '''

    d = {}
    d["entities"] = {}
    for x in entities:
        d["entities"][x] = {"short": x, "verbose": x}

    d["subtypes"] = {}
    if isinstance(subtypes, dict):
        for k, V in subtypes.items():
            d["subtypes"][k] = {}
            for v in V:
                d["subtypes"][k][v] = {"short": v, "verbose": v}

    else:
        for x in subtypes:
            d["subtypes"][x] = {"short": x, "verbose": x}

    d["relations"] = {}
    for x in relations:
        d["relations"][x] = {"short": x, "verbose": x,  "symmetric": False}

    if sent_labels is not None:
        d["sent_labels"] = sent_labels

    json.dump(d, open(path, 'w'), indent=4)

    return d


# def get_dataset_stats(dataset_path, dest_path, name='none'):
#
#     data = json.load(open(dataset_path, 'r'))
#     sent_count = 0
#     word_count = 0
#     entity_counter = Counter()
#     relation_counter = Counter()
#     subtype_counter = Counter()
#     for sent in data:
#
#         tokens = sent[TOKENS]
#         entities = sent[ENTITIES]
#         relations = sent[RELATIONS]
#         subtypes = sent[SUBTYPES]
#
#
#
#         sent_count += 1
#         word_count += len(tokens)
#         for i, entity in enumerate(entities):
#             subtype = subtypes[i]
#
#             k = (entity[TYPE], subtype[TYPE])
#
#             entity_counter[k] += 1
#
#         for relation in relations:
#             head_index = relation[HEAD]
#             tail_index = relation[TAIL]
#
#             head_entity = entities[head_index]
#             tail_entity = entities[tail_index]
#
#             head_subtype = subtypes[head_index]
#             tail_subtype = subtypes[tail_index]
#
#             k = (head_entity[TYPE], head_subtype[TYPE],
#                  tail_entity[TYPE], tail_subtype[TYPE])
#
#             relation_counter[k] += 1
#
#         if "subtypes" in sent:
#             for subtype in sent["subtypes"]:
#                 subtype_counter[subtype["type"]] += 1
#
#     logging.info("")
#     logging.info(f"Data set summary")
#
#     columns = ["entity_type", "subtype", "count"]
#     counts = [tuple(list(k) + [v]) for k, v in entity_counter.items()]
#     df = pd.DataFrame(counts, columns=columns)
#     df = df.sort_values(["entity_type", "subtype"])
#     f = os.path.join(dest_path, f"{name}_entity_counts.csv")
#     df.to_csv(f, index=False)
#     logging.info(f"")
#     logging.info(f"Entity counts:\n{df}")
#
#     df = pd.DataFrame(subtype_counter.items())
#     f = os.path.join(dest_path, f"{name}_subtype_counts.csv")
#     df.to_csv(f, index=False)
#     logging.info(f"")
#     logging.info(f"Subtype counts:\n{df}")
#
#
#     columns = ["head_type", "head_subtype", "tail_type", "tail_subtype", "count"]
#     counts = [tuple(list(k) + [v]) for k, v in relation_counter.items()]
#     df = pd.DataFrame(counts, columns=columns)
#     df = df.sort_values(["head_type", "head_subtype", "tail_type"])
#     f = os.path.join(dest_path, f"{name}_relation_counts.csv")
#     df.to_csv(f, index=False)
#     logging.info(f"")
#     logging.info(f"Relation counts:\n{df}")
#
#
#     return None

def get_dataset_stats(dataset_path, dest_path, name='none'):

    is_subtype_multi_label = False

    data = json.load(open(dataset_path, 'r'))
    sent_count = 0
    word_count = 0
    entity_counter = Counter()
    relation_counter = Counter()
    subtype_counter = Counter()
    for sent in data:
        sent_count += 1
        word_count += len(sent["tokens"])
        for entity in sent["entities"]:
            entity_counter[entity["type"]] += 1
        for relation in sent["relations"]:
            relation_counter[relation["type"]] += 1

        if "subtypes" in sent:
            for subtype in sent["subtypes"]:
                if isinstance(subtype["type"], dict):
                    is_subtype_multi_label = True
                    for k, v in subtype["type"].items():
                        subtype_counter[(k, v)] += 1
                else:
                    subtype_counter[subtype["type"]] += 1

    logging.info("")
    logging.info(f"Data set summary")

    df = pd.DataFrame(entity_counter.items())
    f = os.path.join(dest_path, f"{name}_entity_counts.csv")
    df.to_csv(f, index=False)
    logging.info(f"")
    logging.info(f"Entity counts:\n{df}")


    if is_subtype_multi_label:
        subtype_counter = [tuple(list(k)+[v]) for k, v in subtype_counter.items()]
        df = pd.DataFrame(subtype_counter)
    else:
        df = pd.DataFrame(subtype_counter.items())
    f = os.path.join(dest_path, f"{name}_subtype_counts.csv")
    df.to_csv(f, index=False)
    logging.info(f"")
    logging.info(f"Subtype counts:\n{df}")

    df = pd.DataFrame(relation_counter.items())
    f = os.path.join(dest_path, f"{name}_relation_counts.csv")
    df.to_csv(f, index=False)
    logging.info(f"")
    logging.info(f"Relation counts:\n{df}")

    return None

def get_prediction_file(dir, pattern=PREDICTION_FILE_PATTERN):

    files = list(Path(dir).rglob(pattern))

    # make sure at least one configfile found
    assert len(files) > 0

    # get most recent
    file = max(files, key=os.path.getctime)

    return file


def get_model_dir(dir, pattern=MODEL_FILE_PATTERN):

    files = list(Path(dir).rglob(pattern))

    # make sure at least one configfile found
    assert len(files) > 0

    # get most recent
    file = max(files, key=os.path.getctime)

    model_dir = file.parent

    return model_dir


def update_model_config(model_config, previous_config_file, new_config_file, \
        remove=NON_EVAL_ARGS, config_map=CONFIG_MAP):

    lines = open(previous_config_file, "r").readlines()


    # header = previous_config.pop(0)

    # print(previous_config)



    header = None
    config = OrderedDict()
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            header = stripped_line
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            config[key] = value


    for k, v in config_map.items():
        config[k] = config[v]

    removed = []
    for k in remove:
        if k in config:
            del config[k]
            removed.append(k)

    added = []
    overridden = []
    for k, v in model_config.items():
        if k in config:
            overridden.append(k)
        else:
            added.append(k)
        config[k] = v


    assert "model_path" in config
    assert "tokenizer_path" in config

    logging.info(f"Update model config")
    logging.info(f"Previous config file:    {previous_config_file}")
    logging.info(f"New config file:         {new_config_file}")
    logging.info(f"Parameters removed:      ({len(removed)}) {removed}")
    logging.info(f"Parameters added:        ({len(added)}) {added}")
    logging.info(f"Parameters overridden:   ({len(overridden)}) {overridden}")
    logging.info(f"Mapped config: ")
    for k, v in config_map.items():
        logging.info(f"\t{k} = {v}")


    x = []
    if header is not None:
        x.append(header)
    for k, v in config.items():
            x.append(f"{k} = {v}")
    x = "\n".join(x)


    with open(new_config_file, 'w') as f:
        f.write(x)

    return config
