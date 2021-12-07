

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



class CorpusPredict():

    def __init__(self):

        self.docs_ = OrderedDict()

    def add_doc(self, doc):

        # Build corpus
        assert doc.id not in self.docs_
        self.docs_[doc.id] = doc

    def __str__(self):
        out = OrderedDict()
        out["doc_count"] = self.doc_count()
        out["token_count"] = self.token_count()
        out["sent_count"] = self.sent_count()
        out["entity_count"] = self.entity_count()
        out["event_count"] = self.event_count()
        return 'Corpus(' + ', '.join([f'{k}={v}' for k, v in out.items()]) + ')'

    def doc_count(self):
        return len(self.docs())

    def token_count(self):
        return sum([doc.token_count() for doc in self.docs()])

    def sent_count(self):
        return sum([doc.sent_count() for doc in self.docs()])

    def entity_count(self):
        return sum([doc.entity_count() for doc in self.docs()])

    def event_count(self):
        return sum([doc.event_count() for doc in self.docs()])


    def docs(self, as_dict=False):
        docs = self.docs_

        if as_dict:
            pass
        else:
            docs = list(docs.values())

        return docs


    def entities(self, as_dict=False):
        """
        Get entities by document
        """

        y = OrderedDict()
        for id, doc in self.docs_.items():
            y[doc.id] = doc.entities()

        if as_dict:
            pass
        else:
            y = list(y.values())
        return y


    def events(self, as_dict=False):
        """
        Get events by document
        """

        y = OrderedDict()
        for id, doc in self.docs_.items():
            y[doc.id] = doc.events()

        if as_dict:
            pass
        else:
            y = list(y.values())
        return y
