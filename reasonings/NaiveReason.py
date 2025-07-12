from BaseReason import BaseReason

class NaiveReason(BaseReason):
    def __init__(self, config, knowledge_model):
        super().__init__(config, knowledge_model)

    def solve_qa(self, question, ref = None):
        return self.knowledge_model.response('Naive', {'question': question}, ref, question)

    def solve_fc(self, question, ref = None):
        pass
