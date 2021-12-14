

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from collections import OrderedDict, Counter
import hashlib
import logging
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



from spert_utils.spert_io import spert2doc_dict, spert_doc2brat_dicts
from config.constants import ENCODING, ARG_1, ARG_2, ROLE, TYPE, SUBTYPE, EVENT_TYPE, ENTITIES, COUNT, RELATIONS, ENTITIES, EVENTS
from config.constants import SPACY_MODEL, SUBTYPE_DEFAULT, TRIGGER
from corpus.corpus import Corpus
from corpus.document_brat import DocumentBrat
from corpus.brat import get_brat_files
from corpus.tokenization import get_tokenizer, map2ascii
from utils.proj_setup import make_and_clear
import spacy


def counter2df(counter, columns=None):

    X = []

    for k, counts in counter.items():
        if isinstance(k, (list, tuple)):
            k = list(k)
        else:
            k = [k]
        X.append(k + [counts])

    df = pd.DataFrame(X, columns=columns)

    return df

class CorpusBrat(Corpus):

    def __init__(self, document_class=DocumentBrat, spacy_model=SPACY_MODEL):

        self.document_class = document_class
        self.spacy_model = spacy_model

        Corpus.__init__(self)


    def import_dir(self, path, \
                        n = None,
                        skip = None,
                        ann_map = None,
                        tag_function = None):

        tokenizer = spacy.load(self.spacy_model)

        '''
        Import BRAT directory
        '''

        # Find text and annotation files
        text_files, ann_files = get_brat_files(path)
        file_list = list(zip(text_files, ann_files))
        file_list.sort(key=lambda x: x[1])

        logging.info(f"Importing BRAT directory: {path}")

        if n is not None:
            logging.warn("="*72)
            logging.warn("Only process processing first {} files".format(n))
            logging.warn("="*72)
            file_list = file_list[:n]

        if skip is not None:
            logging.warn("="*72)
            logging.warn(f"Skipping ids: {skip}")
            logging.warn("="*72)

        logging.info(f"BRAT file count: {len(file_list)}")

        pbar = tqdm(total=len(file_list), desc='BRAT import')

        # Loop on annotated files
        for fn_txt, fn_ann in file_list:

            # Read text file
            with open(fn_txt, 'r', encoding=ENCODING) as f:
                text = f.read()

            # Read annotation file
            with open(fn_ann, 'r', encoding=ENCODING) as f:
                ann = f.read()

            if ann_map is not None:
                for pat, val in ann_map:
                    ann = re.sub(pat, val, ann)

            # Use filename as ID
            id = os.path.splitext(os.path.relpath(fn_txt, path))[0]

            if (skip is None) or (id not in skip):

                if tag_function is None:
                    tags = None
                else:
                    tags = tag_function(id)


                doc = self.document_class( \
                    id = id,
                    text = text,
                    ann = ann,
                    tags = tags,
                    tokenizer = tokenizer
                    )

                # Build corpus
                assert doc.id not in self.docs_
                self.docs_[doc.id] = doc

            pbar.update(1)

        pbar.close()


    def sentence_count(self, include=None, exclude=None):

        count = 0
        for doc in self.docs(include=include, exclude=exclude):
            count += doc.sentence_count()
        return count

    def word_count(self, include=None, exclude=None):

        count = 0
        for doc in self.docs(include=include, exclude=exclude):
            count += doc.word_count()
        return count


    def entities(self, include=None, exclude=None, as_dict=False, by_sent=False, entity_types=None):
        """
        Get entities by document
        """

        y = OrderedDict()
        for doc in self.docs(include=include, exclude=exclude):
            y[doc.id] = doc.entities(as_dict=False, by_sent=by_sent, entity_types=entity_types)
        if as_dict:
            pass
        else:
            y = list(y.values())
        return y

    def relations(self, include=None, exclude=None, as_dict=False, by_sent=False, entity_types=None):
        """
        Get relations by document
        """

        y = OrderedDict()
        for doc in self.docs(include=include, exclude=exclude):
            y[doc.id] = doc.relations(by_sent=by_sent, entity_types=entity_types)
        if as_dict:
            pass
        else:
            y = list(y.values())
        return y

    def events(self, include=None, exclude=None, as_dict=False, by_sent=False, event_types=None, entity_types=None):
        """
        Get events by document
        """

        y = OrderedDict()
        for doc in self.docs(include=include, exclude=exclude):
            y[doc.id] = doc.events( \
                                by_sent = by_sent,
                                event_types = event_types,
                                entity_types = entity_types)
        if as_dict:
            pass
        else:
            y = list(y.values())
        return y


    def events2spert(self, include=None, exclude=None, event_types=None, entity_types=None, \
            skip_duplicate_spans=True, include_doc_text=False,
            flat=True, path=None, sample_count=None):
        """
        Get events by document
        """

        logging.warn(f"events2spert")

        y = []
        entity_counter = Counter()
        relation_counter = Counter()
        for i, doc in enumerate(self.docs(include=include, exclude=exclude)):
            y_, ec, rc = doc.events2spert( \
                                event_types = event_types,
                                entity_types = entity_types,
                                skip_duplicate_spans = skip_duplicate_spans,
                                include_doc_text = include_doc_text)
            entity_counter += ec
            relation_counter += rc
            if flat:
                y.extend(y_)
            else:
                y.append(y_)

            if (sample_count is not None) and (i > sample_count):
                break


        if path is not None:
            json.dump(y, open(path, 'w'))

        counts = [(type, keep, count) for (type, keep), count in entity_counter.items()]
        df = pd.DataFrame(counts, columns=["type", "keep", "count"])
        df["frequency"] = df["count"] / df["count"].sum()
        df = df.sort_values(["type", "keep"])
        logging.info(f"")
        logging.info(f"Entity counts:\n{df}")

        counts = [(head, tail, keep, count) for (head, tail, keep), count in relation_counter.items()]
        df = pd.DataFrame(counts, columns=["head", "tail", "keep", "count"])
        df["frequency"] = df["count"] / df["count"].sum()
        df = df.sort_values(["head", "tail", "keep"])
        logging.info(f"")
        logging.info(f"Relation counts:\n{df}")


        return y




    def quality_check(self, path=None, annotator_position=None,
                    labeled_arguments=None, required_arguments=None,
                    id_pattern=None,
                    include=None, exclude=None):

        dfs = []
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):
            df = doc.quality_check( \
                            annotator_position = annotator_position,
                            labeled_arguments = labeled_arguments,
                            required_arguments = required_arguments)
            dfs.append(df)

        df = pd.concat(dfs, axis=0)
        df = df.sort_values(["annotator", "id", "line"], ascending=True)

        if id_pattern is not None:
            df["keep"] = df["id"].apply(lambda x: bool(re.search(id_pattern, x)))
            df = df[df["keep"]]
            del df["keep"]



        if path is not None:
            f = os.path.join(path, "quality_check_all.csv")
            df.to_csv(f, index=False)

            annotators = df["annotator"].unique()
            for annotator in annotators:
                df_temp = df[df["annotator"] == annotator]
                #f = os.path.join(path, f"quality_check_{annotator}.csv")
                #df_temp.to_csv(f, index=False)

                f = os.path.join(path, f"quality_check_{annotator}.xlsx")
                df_temp.to_excel(f, engine='openpyxl',
                            index=False,
                            freeze_panes=(1,1))

        return df

    def annotation_summary(self, path=None, include=None, exclude=None):

        counter = Counter()
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):
            counter += doc.annotation_summary()

        df = counter2df(counter)

        if path is not None:
            f = os.path.join(path, "annotation_summary.csv")
            df.to_csv(f)

        return df

    def label_summary(self, path=None, include=None, exclude=None):

        counters = OrderedDict()
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):
            counters_doc = doc.label_summary()

            for k, v in counters_doc.items():
                if k not in counters:
                    counters[k] = Counter()
                counters[k] += v

        dfs = OrderedDict()
        for k, v in counters.items():
            if k == ENTITIES:
                columns = [TYPE, SUBTYPE, COUNT]
            elif k == RELATIONS:
                columns = [ARG_1, ARG_2, ROLE, COUNT]
            elif k == EVENTS:
                columns = [EVENT_TYPE, TYPE, SUBTYPE, COUNT]
            else:
                columns = None

            dfs[k] = counter2df(v, columns=columns)


        if path is not None:
            for k, df in dfs.items():
                f = os.path.join(path, f"label_summary_{k}.csv")
                df.to_csv(f)

        return dfs

    def write_brat(self, path, include=None, exclude=None):

        make_and_clear(path, recursive=True)
        for doc in self.docs(include=include, exclude=exclude):
            doc.write_brat(path)

    def tag_summary(self, path, include=None, exclude=None):

        summary = []
        for doc in self.docs(include=include, exclude=exclude):
            summary.append((doc.id,doc.tags))

        df = pd.DataFrame(summary,columns=["id", "tags"])
        f = os.path.join(path, "tag_summary.csv")
        df.to_csv(f, index=False)

    def snap_textbounds(self, include=None, exclude=None):
        for doc in self.docs(include=include, exclude=exclude):
            doc.snap_textbounds()

    # OVERRIDE
    def y(self, include=None, exclude=None):

        y = []
        for doc in self.docs(include=include, exclude=exclude):
            y.append(doc.y())
        return y


    # OVERRIDE
    def Xy(self, include=None, exclude=None):
        X = []
        y = []
        for doc in self.docs(include=include, exclude=exclude):
            X_, y_ = doc.Xy()
            X.append(X_)
            y.append(y_)

        return (X, y)

    def map_(self, event_map=None, relation_map=None, tb_map=None, attr_map=None, include=None, exclude=None, path=None):

        counter = Counter()
        for doc in self.docs(include=include, exclude=exclude):
            counter += doc.map_( \
                    event_map = event_map,
                    relation_map = relation_map,
                    tb_map = tb_map,
                    attr_map = attr_map)

        df = counter2df(counter, columns=["Type", "Original", "New", COUNT])
        if path is not None:
            f = os.path.join(path, "mapped_values.csv")
            df.to_csv(f)

        return df

    def swap_spans(self, argument_source, argument_target, include=None, exclude=None):

        counter = Counter()
        for doc in self.docs(include=include, exclude=exclude):
            counter += doc.swap_spans( \
                        argument_source = argument_source,
                        argument_target = argument_target)

        logging.info(f"Swap spans")
        for k, v in counter.items():
            logging.info(f"{k} = {v}")

        return True

    def span_histogram(self, path=None, filename="span_histogram.csv", entity_types=None):

        entities = self.entities(entity_types=entity_types)

        counter = Counter()
        for doc in entities:
            for entity in doc:
                assert (entity_types is None) or (entity.type_ in entity_types)
                text = entity.text.lower()
                counter[(entity.type_, entity.subtype, text)] += 1

        counts = [(type, subtype, text, count) for (type, subtype, text), count in counter.items()]
        df = pd.DataFrame(counts, columns=["type", "subtype", "text", "count"])

        df.sort_values('count', ascending=False, inplace=True)

        fn = os.path.join(path, filename)
        df.to_csv(fn)


        return df

    def transfer_subtype_value(self, argument_pairs, include=None, exclude=None, path=None):


        counts = Counter()
        for doc in self.docs(include=include, exclude=exclude):
            counts += doc.transfer_subtype_value(argument_pairs)

        counts = [{'argument_type': a, 'argument_value': b, 'count': c} \
                                                for (a, b), c in counts.items()]
        df = pd.DataFrame(counts)
        if path is not None:
            f = os.path.join(path, "transfer_subtype_value_counts.csv")
            df.to_csv(f, index=False)

        return df


    def import_spert_corpus(self, path, argument_pairs):



        tokenizer = spacy.load(self.spacy_model)


        spert_doc_dict = spert2doc_dict(path)

        for id, spert_doc in spert_doc_dict.items():
            text, event_dict, relation_dict, tb_dict, attr_dict = spert_doc2brat_dicts(spert_doc, argument_pairs)

            doc = self.document_class( \
                id = id,
                text = text,
                ann = None,
                tags = None,
                tokenizer = tokenizer,
                event_dict = event_dict,
                relation_dict = relation_dict,
                tb_dict = tb_dict,
                attr_dict = attr_dict,
                )

            # Build corpus
            assert doc.id not in self.docs_
            self.docs_[doc.id] = doc
