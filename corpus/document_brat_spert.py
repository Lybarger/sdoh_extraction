


from corpus.document_brat import DocumentBrat


class DocumentBratSpert(DocumentBrat):


    def __init__(self, \
        id,
        text,
        event_dict,
        relation_dict,
        tb_dict,
        attr_dict,
        tags = None,
        tokenizer = None
        ):

        DocumentBrat.__init__(self, \
            id = id,
            text = text,
            ann = None,
            tags = tags,
            tokenizer = tokenizer
            )

        self.get_annotations(event_dict, relation_dict, tb_dict, attr_dict)
        self.get_tokens(text, tokenizer)


    def get_annotations(self, event_dict, relation_dict, tb_dict, attr_dict):

        self.ann = None

        # Extract events, text bounds, and attributes from annotation string
        self.event_dict = event_dict
        self.relation_dict = relation_dict
        self.tb_dict = tb_dict
        self.attr_dict = attr_dict
