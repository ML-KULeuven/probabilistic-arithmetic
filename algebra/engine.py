import uuid

from .categorical import Categorical


class CategoricalEngine:
    def __init__(self):
        random_variables = dict()
        self.id = uuid.uuid4()

    def create_rv(self, name, logits, lower):
        rv = Categorical(name, logits, lower, self)
        self.random_variables[name] = rv
        return rv

    def create_comparison(self, expression):
        raise NotImplementedError()
