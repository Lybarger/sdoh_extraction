

import json
import pandas as pd
import os
import numpy as np
from collections import Counter, OrderedDict




from config.constants import TYPE, SUBTYPE, START, END, TOKENS, ENTITIES, RELATIONS, TRIGGER, ENTITY, TRIGGER_SUBTYPE, ENTITY_SUBTYPE
from config.constants import NT, NP, TP, METRIC, COUNT, FP, FN
from config.constants import EXACT, PARTIAL, OVERLAP, LABEL

from corpus.corpus_brat import CorpusBrat
from corpus.labels import Entity


def PRF(df):

    df["P"] = df[TP]/(df[NP].astype(float))
    df["R"] = df[TP]/(df[NT].astype(float))
    df["F1"] = 2*df["P"]*df["R"]/(df["P"] + df["R"])

    return df

def augment_dict_keys(d, x):

    # get current keys
    keys = list(d.keys())

    # iterate over original keys
    for k_original in keys:

        # create new key
        k_new = k_original
        if isinstance(k_new, str):
            k_new = [k_new]
        k_new = tuple(list(k_new) + [x])

        # update dictionary
        d[k_new] = d.pop(k_original)

    return d

def get_entity_counts(entities, entity_scoring=EXACT, include_subtype=False,
                                                return_counts_by_entity=False):
    """
    Get histogram of entity labels

    Parameters
    ----------
    entities: list of entities, [Entity, Entity,...]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    """

    assert isinstance(entities, list)

    counter = Counter()
    counts_by_entity = []
    for i, entity in enumerate(entities):

        assert isinstance(entity, Entity)

        # key for counter
        k = tuple([entity.type_])
        if include_subtype:
            k = tuple(list(k) + [entity.subtype])

        v = None

        # count spans
        if entity_scoring in [EXACT, OVERLAP, LABEL]:
            v = 1

        # count tokens
        elif entity_scoring in [PARTIAL]:
            v = entity.token_end - entity.token_start

        else:
            raise ValueError(f"invalid entities scoring: {entity_scoring}")

        assert v is not None

        counter[k] += v
        counts_by_entity.append(v)


    if entity_scoring in [EXACT, OVERLAP, LABEL]:
        assert sum(counter.values()) == len(entities)
        assert sum(counts_by_entity) == len(entities)

    if return_counts_by_entity:
        return counts_by_entity
    else:
        return counter

def get_event_counts(events, \
                        trigger_scoring = EXACT,
                        argument_scoring = EXACT,
                        include_subtype = False):
    """
    Get histogram of entity labels
    """

    assert trigger_scoring in [EXACT, OVERLAP]
    assert argument_scoring in [EXACT, OVERLAP, PARTIAL, LABEL]

    counter = Counter()
    for event in events:

        # Get trigger
        trigger = event.arguments[0]

        # iterate over arguments
        for argument in event.arguments[1:]:

            # key for counter

            if include_subtype:
                k = (trigger.type_, trigger.subtype, argument.type_, argument.subtype)
            else:
                k = (trigger.type_, argument.type_)

            # count spans
            if argument_scoring in [EXACT, OVERLAP, LABEL]:
                counter[k] += 1

            # count tokens
            elif argument_scoring in [PARTIAL]:
                counter[k] += argument.token_end - argument.token_start

            else:
                raise ValueError(f"invalid tail scoring: {argument_scoring}")

    return counter


def get_overlap(i1, i2, j1, j2):
    """
    Get overlap between spans
    """

    A = set(range(i1, i2))
    B = set(range(j1, j2))
    overlap = A.intersection(B)
    overlap = sorted(list(overlap))

    return overlap

def get_overlap_count(i1, i2, j1, j2):
    """
    Determine if any overlap
    """
    overlap = get_overlap(i1, i2, j1, j2)
    return len(overlap)

def has_overlap(i1, i2, j1, j2):
    """
    Determine if any overlap
    """
    overlap = get_overlap(i1, i2, j1, j2)
    return len(overlap) > 0


def separate_matches(X, match_indices):

    #assert max(match_indices) < len(X)

    # iterate over gold entities
    match = []
    diff = []
    for i, x in enumerate(X):
        if i in match_indices:
            match.append(x)
        else:
            diff.append(x)

    assert len(match) + len(diff) == len(X)

    return (match, diff)


def get_entity_diff(gold, predict, entity_scoring=EXACT, include_subtype=False):
    """
    Get histogram of matching entities

    Parameters
    ----------
    gold: list of entities, [Entity, Entity,...]
    predict: list of entities, [Entity, Entity,...]
    entity_scoring: scoring type as str in ["exact", "overlap"]
    include_subtype: include subtype in result, as bool
    """

    assert entity_scoring in [EXACT, OVERLAP]

    assert isinstance(gold, list)
    assert isinstance(predict, list)

    # iterate over gold
    I_matches = set([])
    J_matches = set([])

    # iterate over gold entities
    for i, g in enumerate(gold):
        assert isinstance(g, Entity)

        # iterate over predicted entities
        for j, p in enumerate(predict):
            assert isinstance(p, Entity)

            # get entity comparison value as int
            v = compare_entities(g, p, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

            match = (v > 0) and \
                    (i not in I_matches) and \
                    (j not in J_matches)

            # include matched values
            if match:

                assert i not in I_matches
                assert j not in J_matches

                I_matches.add(i)
                J_matches.add(j)

    # separate matches an difference
    gold_match,    gold_diff    = separate_matches(gold,    I_matches)
    predict_match, predict_diff = separate_matches(predict, J_matches)

    return (gold_match, gold_diff, predict_match, predict_diff)

def get_event_diff(gold, predict, scoring=EXACT, include_subtype=False):

    assert scoring in [EXACT, OVERLAP]

    I_matches = set([])
    J_matches = set([])

    gold_match = []
    gold_diff = []
    predict_match = []
    predict_diff = []

    for i, g in enumerate(gold):

        gold_trigger =   g.arguments[0]
        gold_arguments = g.arguments[1:]

        for j, p in enumerate(predict):

            predict_trigger =   p.arguments[0]
            predict_arguments = p.arguments[1:]

            trigger_match = compare_entities(gold_trigger, predict_trigger, \
                                        entity_scoring = scoring,
                                        include_subtype = include_subtype)

            match = trigger_match and \
                    (i not in I_matches) and \
                    (j not in J_matches)

            # include matched values
            if match:

                assert i not in I_matches
                assert j not in J_matches

                I_matches.add(i)
                J_matches.add(j)

                g_match, g_diff, p_match, p_diff = get_entity_diff( \
                        gold = gold_arguments,
                        predict = predict_arguments,
                        entity_scoring = scoring,
                        include_subtype = include_subtype)

                gold_match.extend(      [(gold_trigger,     x) for x in g_match])
                gold_diff.extend(       [(gold_trigger,     x) for x in g_diff])
                predict_match.extend(   [(predict_trigger,  x) for x in p_match])
                predict_diff.extend(    [(predict_trigger,  x) for x in p_diff])


    return (gold_match, gold_diff, predict_match, predict_diff)

def compare_entities(gold, predict, entity_scoring=EXACT, include_subtype=False):
    """
    Get histogram of matching entities

    Parameters
    ----------
    gold: Entity
    predict: Entity
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    """

    assert entity_scoring in [EXACT, PARTIAL, OVERLAP, LABEL]

    assert isinstance(gold, Entity)
    assert isinstance(predict, Entity)

    # assess label match
    type_match = gold.type_ == predict.type_
    subtype_match = gold.subtype == predict.subtype
    if include_subtype:
        type_match = type_match and subtype_match

    y = 0

    if type_match:
        g1, g2 = gold.token_start, gold.token_end
        p1, p2 = predict.token_start, predict.token_end

        indices_overlap = get_overlap_count(g1, g2, p1, p2)

        # exact match
        # count spans
        if (entity_scoring == EXACT) and ((g1, g2) == (p1, p2)):
            y = 1

        # partial match
        # count tokens
        elif (entity_scoring == PARTIAL) and indices_overlap:
            y = indices_overlap

        # any overlap match
        # count spans
        elif (entity_scoring == OVERLAP) and indices_overlap:
            y = 1

        # subtype labels match
        # count spans
        elif (entity_scoring == LABEL) and subtype_match:
            y = 1

    return y









def get_entity_matches(gold, predict, entity_scoring=EXACT, include_subtype=False, \
                                    return_counts_by_entity=False):
    """
    Get histogram of matching entities

    Parameters
    ----------
    gold: list of entities, [Entity, Entity,...]
    predict: list of entities, [Entity, Entity,...]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    """

    assert entity_scoring in [EXACT, PARTIAL, OVERLAP, LABEL]

    assert isinstance(gold, list)
    assert isinstance(predict, list)

    counter = Counter()
    counts_by_entity = []
    grand_total = 0


    #print('='*80)
    # iterate over gold
    I = set([])
    J = set([])

    for i, g in enumerate(gold):
        assert isinstance(g, Entity)

        matching_j = []
        total_v = 0

        # iterate over predicted entities
        for j, p in enumerate(predict):
            assert isinstance(p, Entity)

            # get entity comparison value as int
            v = compare_entities(g, p, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)

            # allow many to many span matching
            if entity_scoring in [PARTIAL]:
                match = (v > 0)

            # only allow each span to match once
            # entity_scoring in [EXACT, OVERLAP, LABEL]
            else:
                match = (v > 0) and (i not in I) and (j not in J)
                #print(match, v, i, I, j, J)


            # include matched values
            if match:

                #assert i not in I
                #assert j not in J

                I.add(i)
                J.add(j)

                # key for counter
                k = tuple([g.type_])
                if include_subtype:
                    k = tuple(list(k) + [g.subtype])

                assert isinstance(k, tuple), type(k)
                counter[k] += v
                grand_total += v

                matching_j.append(j)
                total_v += v

        counts_by_entity.append((matching_j, total_v))

    if entity_scoring not in [PARTIAL]:
        assert grand_total <= len(predict)

    if return_counts_by_entity:
        return counts_by_entity
    else:
        return counter

def get_event_matches(gold, predict, \
                                trigger_scoring = EXACT,
                                argument_scoring = EXACT,
                                include_subtype = False):

    assert trigger_scoring in [EXACT, OVERLAP]
    assert argument_scoring in [EXACT, OVERLAP, PARTIAL, LABEL]

    I = set([])
    J = set([])
    counter = Counter()
    for i, g in enumerate(gold):

        gold_trigger =   g.arguments[0]
        gold_arguments = g.arguments[1:]


        trigger_key = tuple([gold_trigger.type_])
        if include_subtype:
            trigger_key = tuple(list(trigger_key) + [gold_trigger.subtype])


        for j, p in enumerate(predict):

            predict_trigger =   p.arguments[0]
            predict_arguments = p.arguments[1:]

            trigger_match = compare_entities(gold_trigger, predict_trigger, \
                                        entity_scoring = trigger_scoring,
                                        include_subtype = include_subtype)


            # allow many to many span matching
            if trigger_scoring in [PARTIAL]:
                match = trigger_match

            # only allow each span to match once
            # trigger_scoring in [EXACT, OVERLAP, LABEL]
            else:
                match = trigger_match and (i not in I) and (j not in J)

            # include matched values
            if match:

                I.add(i)
                J.add(j)

                cntr = get_entity_matches(gold_arguments, predict_arguments, \
                                        entity_scoring = argument_scoring,
                                        include_subtype = include_subtype)

                for entity_key, v in cntr.items():

                    # key for counter
                    assert isinstance(entity_key, tuple), type(k)

                    k = tuple(list(trigger_key) + list(entity_key))
                    counter[k] += v

    return counter







def get_entity_df(counts, include_subtype=False):


    count_cols = [NT, NP, TP]

    cols = [TYPE]
    if include_subtype:
        cols = cols + [SUBTYPE]


    counts = [list(k) + [v] for k, v in counts.items()]

    df = pd.DataFrame(counts, columns= cols + [METRIC, COUNT])

    if include_subtype:
        df[SUBTYPE].fillna(value="none", inplace=True)

    df = pd.pivot_table(df, values=COUNT, index=cols, columns=METRIC)


    df = df.fillna(0).astype(int)
    df = df.reset_index()

    df = df.sort_values(cols)
    for c in count_cols:
        if c not in df:
            df[c] = 0
    df = df[cols + count_cols]
    df = PRF(df)
    df = df.fillna(0)

    return df

def get_event_df(counts, include_subtype=False):

    count_cols = [NT, NP, TP]

    if include_subtype:
        cols = [TRIGGER, TRIGGER_SUBTYPE, ENTITY, ENTITY_SUBTYPE]
    else:
        cols = [TRIGGER, ENTITY]


    counts = [list(k) + [v] for k, v in counts.items()]
    df = pd.DataFrame(counts, columns= cols + [METRIC, COUNT])

    if include_subtype:
        df[TRIGGER_SUBTYPE].fillna(value="none", inplace=True)
        df[ENTITY_SUBTYPE].fillna(value="none", inplace=True)


    df = pd.pivot_table(df, values=COUNT, index=cols, columns=METRIC)
    df = df.fillna(0).astype(int)
    df = df.reset_index()
    for c in count_cols:
        if c not in df:
            df[c] = 0
    df = df[cols + count_cols]
    df = PRF(df)
    df = df.fillna(0)

    return df







def score_entities(gold, predict, entity_scoring=EXACT, include_subtype=False):
    '''
    Evaluate predicted entities against true entities

    Parameters
    ----------
    gold: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    predict: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    '''

    assert len(gold) == len(predict)

    nt = Counter()
    np = Counter()
    tp = Counter()

    # iterate over documents
    for g, p in zip(gold, predict):

        nt += get_entity_counts(g, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = include_subtype)
        np += get_entity_counts(p, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = include_subtype)

        tp += get_entity_matches(g, p, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype)


    counter = Counter()
    for name, d in [(NT, nt), (NP, np), (TP, tp)]:
        d = augment_dict_keys(d, name)
        counter.update(d)

    df = get_entity_df(counter, include_subtype=include_subtype)

    return df

def label_name(entity):

    if (entity.subtype is None) or (entity.subtype == entity.type_):
        return entity.type_
    else:
        return f"{entity.type_} ({entity.subtype})"


def score_entities_detailed(gold, predict, entity_scoring=EXACT, include_subtype=False):
    '''
    Evaluate predicted entities against true entities

    Parameters
    ----------
    gold: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    predict: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    '''

    assert len(gold) == len(predict)




    docs = []

    # iterate over documents
    for g, p in zip(gold, predict):

        nt = get_entity_counts(g, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = include_subtype,
                                    return_counts_by_entity = True)

        np = get_entity_counts(p, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = include_subtype,
                                    return_counts_by_entity = True)

        tp = get_entity_matches(g, p, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype,
                                return_counts_by_entity = True)

        doc = []
        matched_predict = []
        for i, (matched_j, v) in enumerate(tp):

            d = OrderedDict()

            a = g[i]
            d[NT] = nt[i]
            d[NP] = 0
            d[TP] = v
            d[FP] = -1
            d[FN] = -1


            d["label"] = label_name(a)
            d["gold_label"] = label_name(a)
            d["gold_text"] = a.text
            d["gold_start"] = a.char_start
            d["gold_end"] = a.char_end

            matched_predict.extend(matched_j)
            for k, j in enumerate(matched_j):
                #assert len(matched_j) <= 3, len(matched_j)
                #for k in range(3):

                #if k < len(matched_j):


                assert v > 0
                #j = matched_j[k]

                a = p[j]
                d[f"predict_{k}_label"] = label_name(a)
                d[f"predict_{k}_text"] = a.text
                d[f"predict_{k}_start"] = a.char_start
                d[f"predict_{k}_end"] = a.char_end
                d[NP] += np[j]
                #else:
                #    d[f"predict_{k}_label"] = ''
                #    d[f"predict_{k}_text"] = ''
                #    d[f"predict_{k}_start"] = None
                #    d[f"predict_{k}_end"] = None

            d[FP] = d[NP] - d[TP]
            d[FN] = d[NT] - d[TP]


            doc.append(d)


        matched_predict = set(matched_predict)


        for j, v in enumerate(np):
            if j not in matched_predict:
                d = OrderedDict()


                a = p[j]

                d[NT] = 0
                d[NP] = v
                d[TP] = 0
                d[FP] = d[NP] - d[TP]
                d[FN] = d[NT] - d[TP]

                d["label"] = label_name(a)
                d["gold_label"] = ''
                d["gold_text"] = ''
                d["gold_start"] = None
                d["gold_end"] = None

                k = 0
                d[f"predict_{k}_label"] = label_name(a)
                d[f"predict_{k}_text"] = a.text
                d[f"predict_{k}_start"] = a.char_start
                d[f"predict_{k}_end"] = a.char_end

                #for k in range(1,3):
                #    d[f"predict_{k}_label"] = ''
                #    d[f"predict_{k}_text"] = ''
                #    d[f"predict_{k}_start"] = None
                #    d[f"predict_{k}_end"] = None

                doc.append(d)


        docs.append(doc)


    return docs



def score_events(gold, predict, \
                        trigger_scoring = EXACT,
                        argument_scoring = EXACT,
                        include_subtype = False):
    '''
    Evaluate predicted events against true events

    Parameters
    ----------
    gold: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    predict: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    '''

    assert len(gold) == len(predict)

    nt = Counter()
    np = Counter()
    tp = Counter()

    # iterate over documents
    for g, p in zip(gold, predict):

        nt += get_event_counts(g, \
                                trigger_scoring = trigger_scoring,
                                argument_scoring = argument_scoring,
                                include_subtype = include_subtype)

        np += get_event_counts(p, \
                                trigger_scoring = trigger_scoring,
                                argument_scoring = argument_scoring,
                                include_subtype = include_subtype)

        tp += get_event_matches(g, p, \
                                trigger_scoring = trigger_scoring,
                                argument_scoring = argument_scoring,
                                include_subtype = include_subtype)
    counter = Counter()
    for name, d in [(NT, nt), (NP, np), (TP, tp)]:
        d = augment_dict_keys(d, name)
        counter.update(d)

    df = get_event_df(counter, include_subtype=include_subtype)

    return df

def score_full(gold_entities, predict_entities, gold_events, predict_events, \
                scoring = EXACT,
                entity_scoring = None,
                trigger_scoring = None,
                argument_scoring = None,
                path = None,
                description = None,
                ids = None):

    if scoring is None:
        assert entity_scoring is not None
        assert trigger_scoring is not None
        assert argument_scoring is not None

    elif scoring == EXACT:
        entity_scoring = EXACT
        trigger_scoring = EXACT
        argument_scoring = EXACT
        description = EXACT

    elif scoring == OVERLAP:
        entity_scoring = OVERLAP
        trigger_scoring = OVERLAP
        argument_scoring = OVERLAP
        description = OVERLAP

    elif scoring == PARTIAL:
        entity_scoring = PARTIAL
        trigger_scoring = OVERLAP
        argument_scoring = PARTIAL
        description = PARTIAL

    elif scoring == LABEL:
        entity_scoring = LABEL
        trigger_scoring = OVERLAP
        argument_scoring = LABEL
        description = LABEL
    else:
        raise ValueError(f"invalid scoring type: {scoring}")


    df_entity_type = score_entities(gold_entities, predict_entities, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = False)

    df_entity_subtype = score_entities(gold_entities, predict_entities, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = True)

    df_argument_type = score_events(gold_events, predict_events, \
                                    trigger_scoring = trigger_scoring,
                                    argument_scoring = argument_scoring,
                                    include_subtype = False)

    df_argument_subtype = score_events(gold_events, predict_events, \
                                    trigger_scoring = trigger_scoring,
                                    argument_scoring = argument_scoring,
                                    include_subtype = True)

    df_dict = OrderedDict()
    df_dict["entity_type"] =    df_entity_type
    df_dict["entity_subtype"] = df_entity_subtype
    df_dict["argument_type"] = df_argument_type
    df_dict["argument_subtype"] = df_argument_subtype


    if path is not None:
        for k, df in df_dict.items():
            f = os.path.join(path, f"{k}_{description}.csv")
            df.to_csv(f, index=False)

    return df_dict

def score_docs(gold_docs, predict_docs, \
                            scoring = [EXACT],
                            entity_scoring = None,
                            trigger_scoring = None,
                            argument_scoring = None,
                            destination = None,
                            description = None):

    """
    Score entities
    """

    assert isinstance(gold_docs, dict)
    assert isinstance(predict_docs, dict)
    g = sorted(list(gold_docs.keys()))
    p = sorted(list(predict_docs.keys()))
    assert len(g) == len(p), f"{len(g)} vs {len(p)}"
    assert g == p, f"{g} vs {p}"


    gold_entities = []
    predict_entities = []
    gold_events = []
    predict_events = []
    ids = []
    for id in gold_docs:
        gold_doc = gold_docs[id]
        predict_doc = predict_docs[id]

        gold_entities.append(gold_doc.entities())
        predict_entities.append(predict_doc.entities())

        gold_events.append(gold_doc.events())
        predict_events.append(predict_doc.events())

        ids.append(id)



    """
    Score events
    """

    if isinstance(scoring, str):
        scoring = [scoring]

    df_dict = OrderedDict()
    for s in scoring:

        df_dict[s] = score_full( \
                    gold_entities = gold_entities,
                    predict_entities = predict_entities,
                    gold_events = gold_events,
                    predict_events = predict_events,
                    scoring = s,
                    entity_scoring = entity_scoring,
                    trigger_scoring = trigger_scoring,
                    argument_scoring = argument_scoring,
                    path = destination,
                    description = description,
                    ids = ids)



    return df_dict



def score_brat_events(gold_dir, predict_dir, \
                            corpus_class = CorpusBrat,
                            sample_count = None,
                            scoring = [EXACT],
                            entity_scoring = None,
                            trigger_scoring = None,
                            argument_scoring = None,
                            destination = None,
                            description = None):


    gold_corpus = corpus_class()
    gold_corpus.import_dir(gold_dir, n=sample_count)

    predict_corpus = corpus_class()
    predict_corpus.import_dir(predict_dir, n=sample_count)

    gold_docs = gold_corpus.docs()
    predict_docs = predict_corpus.docs()

    df_dict = score_docs(gold_docs, predict_docs, \
                            scoring = scoring,
                            entity_scoring = entity_scoring,
                            trigger_scoring = trigger_scoring,
                            argument_scoring = argument_scoring,
                            destination = destination,
                            description = description)

    return df_dict



def get_token_count(tokens, length_max=None):

    token_count = len(tokens)

    if length_max is not None:
        token_count = min(token_count, length_max)
    return token_count


def entity_error_analysis(gold, predict, entity_scoring=EXACT, include_subtype=False, entity_types=None, length_max=None):
    '''
    Evaluate predicted entities against true entities

    Parameters
    ----------
    gold: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    predict: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    '''

    assert len(gold) == len(predict)

    entity_filter = lambda entities, types: [entity for entity in entities if entity.type_ in types]



    # iterate over documents
    count_nt = Counter()
    count_np = Counter()
    count_tp = Counter()
    false_negatives = []
    for g, p in zip(gold, predict):

        if entity_types is not None:
            g = entity_filter(g, entity_types)
            p = entity_filter(p, entity_types)

        nt = get_entity_counts(g, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = include_subtype,
                                    return_counts_by_entity = True)

        np = get_entity_counts(p, \
                                    entity_scoring = entity_scoring,
                                    include_subtype = include_subtype,
                                    return_counts_by_entity = True)

        tp = get_entity_matches(g, p, \
                                entity_scoring = entity_scoring,
                                include_subtype = include_subtype,
                                return_counts_by_entity = True)

        assert len(nt) == len(g)
        assert len(tp) == len(g)
        for g_, nt_, tp_ in zip(g, nt, tp):

            token_count = get_token_count(g_.tokens, length_max=length_max)

            count_nt[token_count] += nt_

            idxs, n = tp_
            if n:
                assert len(idxs) == 1
                idx = idxs[0]
                count_tp[token_count] += n
            else:
                false_negatives.append(g_)


        assert len(np) == len(p)
        for p_, np_ in zip(p, np):

            token_count = get_token_count(p_.tokens, length_max=length_max)

            count_np[token_count] += np_

    counts = list(count_nt.keys()) + list(count_np.keys())
    counts = sorted(list(set(counts)))

    rows = []
    for c in counts:
        nt = count_nt[c]
        np = count_np[c]
        tp = count_tp[c]
        rows.append((c, nt, np, tp))
    df = pd.DataFrame(rows, columns=["token_count", NT, NP, TP])
    df = PRF(df)
    df = df.fillna(0)

    rows = []
    for entity in false_negatives:
        d = OrderedDict()
        d["type"] = entity.type_
        d["subtype"] = 'none' if entity.subtype is None else entity.subtype
        d["token_count"] = len(entity.tokens)
        d["text"] = entity.text.lower()
        rows.append(d)
    df_fn = pd.DataFrame(rows)
    columns = df_fn.columns.tolist()
    df_fn["count"] = 1
    #df_fn =  df_fn.groupby(columns).sum()
    #df_fn = df_fn.groupby(df_fn.columns.tolist()).size().reset_index().\
    #rename(columns={0:'records'})
    agg = {"count":"sum"}
    #df_fn = df_fn.groupby(["type", "subtype", "text"]).agg(agg)
    df_fn = df_fn.groupby(columns).agg(agg)
    df_fn = df_fn.reset_index()
    df_fn = df_fn.sort_values(["count", "token_count"], ascending=[False, True])

    return df, df_fn


"""

def score_relations(gold, predict, \
                    head_scoring = EXACT,
                    tail_scoring = EXACT,
                    include_subtype = False):


    nt = get_relation_counts(gold, \
                            head_scoring = head_scoring,
                            tail_scoring = tail_scoring,
                            include_subtype = include_subtype)

    np = get_relation_counts(predict, \
                            head_scoring = head_scoring,
                            tail_scoring = tail_scoring,
                            include_subtype = include_subtype)

    tp = get_relation_matches(gold, predict, \
                                    head_scoring = head_scoring,
                                    tail_scoring = tail_scoring,
                                    include_subtype = include_subtype)
    counter = Counter()
    for name, d in [(NT, nt), (NP, np), (TP, tp)]:
        d = augment_dict_keys(d, name)
        counter.update(d)

    return counter
"""
