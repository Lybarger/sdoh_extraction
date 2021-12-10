


import spacy

from corpus.corpus_brat import CorpusBrat
from corpus.document_brat_spert import DocumentBratSpert
from spert_utils.spert_io import spert2doc_dict, spert2brat_dicts
from config.constants import SPACY_MODEL

class CorpusBratSpert(CorpusBrat):

    def __init__(self, document_class=DocumentBratSpert, spacy_model=SPACY_MODEL):


        CorpusBrat.__init__(self, document_class=document_class, spacy_model=spacy_model)


    def import_spert_corpus(self, path):



        tokenizer = spacy.load(self.spacy_model)


        spert_doc_dict = spert2doc_dict(path)


        for id, spert_doc in spert_doc_dict.items():
            event_dict, tb_dict, attr_dict = spert2brat_dicts(spert_doc)

    # # process documents
    # # iterate over documents in corpus
    # corpus = CorpusPredict()
    # for id, spert_doc in by_doc.items():
    #     print(id)
    #     doc = spert2brat_doc(spert_doc, tokenizer=tokenizer)
    #     #corpus.add_doc(doc)
