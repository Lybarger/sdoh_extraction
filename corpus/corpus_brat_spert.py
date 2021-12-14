


import spacy

from corpus.corpus_brat import CorpusBrat
from corpus.document_brat_spert import DocumentBratSpert
from spert_utils.spert_io import spert2doc_dict, spert_doc2brat_dicts
from config.constants import SPACY_MODEL

class CorpusBratSpert(CorpusBrat):

    def __init__(self, document_class=DocumentBratSpert, spacy_model=SPACY_MODEL):


        CorpusBrat.__init__(self, document_class=document_class, spacy_model=spacy_model)


    def import_spert_corpus(self, path, argument_pairs):



        tokenizer = spacy.load(self.spacy_model)


        spert_doc_dict = spert2doc_dict(path)

        for id, spert_doc in spert_doc_dict.items():
            text, event_dict, relation_dict, tb_dict, attr_dict = spert_doc2brat_dicts(spert_doc, argument_pairs)

            doc = self.document_class( \
                id = id,
                text = text,
                event_dict = event_dict,
                relation_dict = relation_dict,
                tb_dict = tb_dict,
                attr_dict = attr_dict,
                tags = None,
                tokenizer = tokenizer
                )

            # Build corpus
            assert doc.id not in self.docs_
            self.docs_[doc.id] = doc
