

from collections import Counter
from collections import OrderedDict
import logging



from config.constants import EVENT



class DocumentPredict():


    def __init__(self, id, entities, events, tokens=None, offsets=None):

        self.id = id
        self._entities = entities
        self._events = events
        self.tokens = tokens
        self.offsets = offsets



    def __str__(self):
        out = OrderedDict()
        out["id"] = self.id
        out["token_count"] = self.token_count()
        out["sent_count"] = self.sent_count()
        out["entity_count"] = self.entity_count()
        out["event_count"] = self.event_count()
        return 'Doc(' + ', '.join([f'{k}={v}' for k, v in out.items()]) + ')'


    def token_count(self):
        return sum([len(sent) for sent in self.tokens])

    def sent_count(self):
        return len(self.tokens)

    def entity_count(self):
        return len(self.entities())

    def event_count(self):
        return len(self.events())

    def entities(self):
        '''
        get list of entities for document
        '''

        return self._entities

    def events(self):
        '''
        get list of entities for document
        '''

        return self._events
