
import os
import re
import pandas as pd
import numpy as np



class Document:
    '''
    Document container
    '''
    def __init__(self, \
        id,
        text,
        tags = None,
        ):

        # Make sure text is not None and is string
        assert text is not None
        assert isinstance(text, str)

        # Make sure text has at least 1 non-white space character
        text_wo_ws = ''.join(text.split())
        assert len(text_wo_ws) > 0, '''"{}"'''.format(repr(text))

        self.id = id
        self.text = text

        if tags is None:
            self.tags=set([])
        else:
            self.tags = tags


    def __str__(self):
        return self.text

    def X(self):
        return self.text
