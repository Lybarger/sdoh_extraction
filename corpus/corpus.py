


import json
import os
import joblib
import re
import shutil
import pandas as pd
from multiprocessing import Pool
import traceback
from tqdm import tqdm
import numpy as np
from collections import OrderedDict, Counter
import logging
from pathlib import Path
import itertools

from sklearn.model_selection import train_test_split


from corpus.tokenization import get_tokenizer, get_tokens
from corpus.document import Document
from config.constants import ENCODING, TEXT_FILE_EXT, ANN_FILE_EXT, QC, ID, SUBSET, TRAIN, TEST, DEV
from corpus.brat import write_txt, write_ann
from utils.proj_setup import make_and_clear
from utils.random_sample import random_sample



def batch_documents(doc):
    '''
    Process batch of documents
    '''

    # Executed with error handling
    try:
        return Document(**doc)

    # Catch exceptions
    except Exception as e:
        print('Caught exception in worker thread:')

        for k, v in doc.items():
            print('{}:\t{}'.format(k, v))

        # Print exception
        traceback.print_exc()

        raise e


def include_keep(tags, include):

    # assume keep is true by default
    keep = True

    # exclude labels provided
    if (include is not None):

        # require all include tags to be present
        if not include.issubset(tags):
            keep = False

    return keep


def exclude_keep(tags, exclude):

    # assume keep is true by default
    keep = True

    # exclude labels provided
    if (exclude is not None):

        # at least some overlap between exclude and tags
        if len(exclude.intersection(tags)) > 0:
            keep = False

    return keep


class Corpus:
    '''
    Corpus container (collection of documents)
    '''
    def __init__(self):

        self.docs_ = OrderedDict()

    def __len__(self):
        return len(self.docs_)

    def __getitem__(self, key):
        return self.docs_[key]

    def __setitem__(self, key, item):
        self.docs_[key] = item

    def __delitem__(self, key):
        del self.docs_[key]

    def add_doc(self, doc):
        '''
        Add new document to corpus
        '''

        # Prevent duplicate document IDs
        assert doc.id not in self.docs_, \
        "corpus ids:\n{}\ndoc id:\t{}".format(self.docs_.keys(), doc.id)

        # Add document to corpus
        self.docs_[doc.id] = doc

        return True

    def doc_filter(self, include=None, exclude=None):
        '''
        Get filtered set of documents
        '''

        if isinstance(include, str):
            include = [include]

        if isinstance(exclude, str):
            exclude = [exclude]

        if (include is not None):
            include = set(include)

        if (exclude is not None):
            exclude = set(exclude)

        docs_out = OrderedDict()
        for id, doc in self.docs_.items():

            # go to document tags
            tags = doc.tags
            if tags is None:
                tags = set([])
            if not isinstance(tags, set):
                tags = set(tags)

            keep = True
            keep = keep and include_keep(tags, include)
            keep = keep and exclude_keep(tags, exclude)

            if keep:
                docs_out[id] = doc

        if (include is not None) or (exclude is not None):
            logging.info('Document filter')
            logging.info('\tinclude:         {}'.format(include))
            logging.info('\texclude:         {}'.format(exclude))
            logging.info('\tcount, all:      {}'.format(len(self)))
            logging.info('\tcount, filtered: {}'.format(len(docs_out)))

        return docs_out


    def id2stem(self, id):
        '''
        Convert document ID to filename stem
        '''
        return id

    def docs(self, as_dict=False, include=None, exclude=None):
        '''
        Get documents
        '''

        # Get filtered documents
        docs = self.doc_filter(include=include, exclude=exclude)

        # Output documents as dict (no change to output needed)
        if as_dict:
            pass
        else:
            docs = list(docs.values())

        return docs

    def X(self, include=None, exclude=None):

        X = []
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):
            X.append(doc.X())

        return X

    def ids(self, as_stem=False, include=None, exclude=None):
        '''
        Get tokenized documents
        '''
        ids = []
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):

            id = doc.id
            if as_stem:
                id = self.id2stem(id)
            ids.append(id)

        return ids

    def doc_count(self, include=None, exclude=None):
        '''
        Get document count
        '''

        return len(self.docs(include=include, exclude=exclude))


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


    def tags_by_doc(self, path=None, include=None, exclude=None):

        tags = OrderedDict()
        for id, doc in self.docs(as_dict=True, include=include, exclude=exclude).items():
            tags[id] = tuple(doc.tags)


        if path is not None:
            f = os.path.join(path, "tags_by_doc.csv")
            df = pd.DataFrame(tags.items(), columns=["id", "tags"])
            df.to_csv(f, index=False)

        return tags

    def tag_histogram(self, path=None, include=None, exclude=None):

        tags_by_doc = self.tags_by_doc(include=include, exclude=exclude)
        tags = [tag for tags in tags_by_doc.values() for tag in tags]
        counter = Counter(tags)

        df = pd.DataFrame(counter.items(), columns=["tag", "count"])


        if path is not None:
            f = os.path.join(path, "tag_histogram.csv")
            df.to_csv(f, index=False)

        return df

    def histogram(self, tokenizer, path=None, include=None, exclude=None):

        sent_lengths = Counter()
        doc_lengths = Counter()
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):
            tokens = get_tokens(doc.text, tokenizer)

            for sent in tokens:
                sent_lengths[len(sent)] += 1
            doc_lengths[len(tokens)] += 1

        df_sent = pd.DataFrame(sent_lengths.items(), columns=["sentence_length", "count"])
        df_sent.sort_values("sentence_length", ascending=True, inplace=True)

        df_doc = pd.DataFrame(doc_lengths.items(), columns=["document_length", "count"])
        df_doc.sort_values("document_length", ascending=True, inplace=True)

        if path is not None:
            f = os.path.join(path, "sentence_lengths.csv")
            df_sent.to_csv(f, index=False)

            f = os.path.join(path, "document_lengths.csv")
            df_doc.to_csv(f, index=False)

        return (df_sent, df_doc)

    def assign_splits(self, \
                train_size = None,
                dev_size = None,
                test_size = None,
                splits_file = None,
                random_state = None,
                shuffle = True,
                path = None,
                include = None,
                exclude = None):


        logging.info(f"Split assignment")


        ids = [doc.id for doc in self.docs(include=include, exclude=exclude)]

        if splits_file is None:

            total_size = train_size + dev_size + test_size
            assert total_size == 1.0, total_size


            ids_train, ids_dev_test = train_test_split(ids, \
                                        test_size = 1 - train_size,
                                        random_state = random_state,
                                        shuffle = shuffle)

            ids_dev, ids_test = train_test_split(ids_dev_test, \
                                        test_size = test_size/(test_size + dev_size),
                                        random_state = random_state,
                                        shuffle = shuffle)


            train = list(zip(ids_train, [TRAIN]*len(ids_train)))
            dev =   list(zip(ids_dev,   [DEV]*len(ids_dev)))
            test =  list(zip(ids_test,  [TEST]*len(ids_test)))

            assert sorted(ids_train + ids_dev + ids_test) == sorted(ids)

            df = pd.DataFrame(train + dev + test, columns=[ID, SUBSET])

            logging.info(f"Generating NEW random assignments ")
            logging.info(f"Random assignment: ")
            logging.info(f"\ttrain_size:   {train_size}")
            logging.info(f"\tdev_size:     {dev_size}")
            logging.info(f"\ttest_size:    {test_size}")
            logging.info(f"\trandom_state: {random_state}")

        else:
            df = pd.read_csv(splits_file)

            assert ID in df
            assert SUBSET in df

            df_temp = df.groupby([SUBSET]).count()
            df_temp["Relative Frequency"] = df_temp[ID]/df_temp[ID].sum()

            logging.info(f"Loading EXISTING assignments: {splits_file}")
            logging.info(f"Subset distribution:\n{df_temp}")



        if path is not None:
            f = os.path.join(path, "splits.csv")
            df.to_csv(f, index=False)


        ids = set(ids)
        n_all = len(ids)
        for index, row in df.iterrows():
            id = row[ID]
            self.docs_[id].tags.add(row[SUBSET])
            ids.remove(id)
        logging.info(f"ID count, all: {n_all}")
        logging.info(f"ID count, split: {len(df)}")
        logging.info(f"ID count, not assigned: {len(ids)}")
        if len(ids) > 0:
            logging.info(f"Not assigned: {ids}")


        return df

    def add_annotator_tags(self, annotator_position=0, include=None, exclude=None):

        annotators = set([])
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):
            id_parts = doc.id.split(os.path.sep)
            annotator = id_parts[annotator_position]
            annotators.add(annotator)

            if annotator in doc.tags:
                logging.info(f"Annotator already in tags: {annotator}")

            doc.tags.add(annotator)

        annotators = sorted(list(annotators))

        logging.info(f"Adding annotator tags")
        logging.info(f"Annotator count: {len(annotators)}")
        logging.info(f"Annotators:      {annotators}")
        logging.info(f"Annotator distribution:")
        for annotator in annotators:
            n = len(self.docs(include=annotator))
            #logging.info(f"\t{annotator} - {n}")

        return annotators

    def get_docs_by_annotator(self, annotator_position=0, update_id=True, include=None, exclude=None):

        annotators = self.add_annotator_tags( \
                                annotator_position = annotator_position,
                                include = include,
                                exclude = exclude)

        output = OrderedDict()
        for annotator in annotators:

            if annotator not in output:
                output[annotator] = OrderedDict()

            docs = self.docs(as_dict=True, include=annotator)

            for id, doc in docs.items():

                if update_id:
                    id_parts = doc.id.split(os.path.sep)
                    id_parts.pop(annotator_position)

                    id = f"{os.path.sep}".join(id_parts)
                    doc.id = id

                output[annotator][id] = doc


        return output

    def random_sample(self, size, \
            exclude_ids = None,
            seed = 1,
            path = None,
            brat = True,
            encoding = ENCODING,
            annotators = None,
            anno_type = 'single',  #  'single' or 'multiple'
            include_tags = None,
            exclude_tags = None,
            strip_ws = True):

        '''
        Randomly sample documents
        '''

        # Get relevant documents
        docs = self.docs(   \
                        as_dict = True,
                        include = include_tags,
                        exclude = exclude_tags)

        # IDs as list
        ids = sorted(list(docs.keys()))

        sampled_ids = random_sample(ids, size, \
                        seed = seed,
                        exclude = exclude_ids,
                        sort_values = True)

        # Extract samples
        sampled_docs = OrderedDict()
        for id in sampled_ids:
            sampled_docs[id] = docs[id]
        assert len(sampled_docs) == size

        # Write sampled files
        if not path is None:

            assert annotators is not None

            if anno_type == 'single':
                annotators = itertools.cycle(annotators)

            elif anno_type == 'double':
                n = 2
                c = len(annotators)
                assert (c % n) == 0, '''Double annotation routine assumes the number of annotators is even. The code needs to be modified to account for an odd number of annotators'''
                annotator_pairs = [annotators[i:i+n] for i in range(0, c, n)]

                check = []
                for pair in annotator_pairs:
                    assert len(pair) == n
                    check.extend(pair)
                assert sorted(check) == sorted(annotators)

                annotator_pairs = itertools.cycle(annotator_pairs)


            # Loop on documents
            for id_, doc in sampled_docs.items():

                stem = self.id2stem(id_)
                text = doc.text

                if anno_type == 'single':
                    annotator = next(annotators)

                    fn = os.path.join(annotator, stem)
                    write_txt(path, fn, text, strip_ws=strip_ws)
                    if brat:
                        write_ann(path, fn, '')


                elif anno_type == 'multiple':
                    for annotator in annotators:
                        fn = os.path.join(annotator, stem)
                        write_txt(path, fn, text, strip_ws=strip_ws)
                        if brat:
                            write_ann(path, fn, '')


                elif anno_type == 'double':
                    pair = next(annotator_pairs)
                    for annotator in pair:
                        fn = os.path.join(annotator, stem)
                        write_txt(path, fn, text, strip_ws=strip_ws)
                        if brat:
                            write_ann(path, fn, '')


                else:
                    ValueError("invalid annotation type: {}".format(anno_type))

        return sampled_docs
