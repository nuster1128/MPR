class BaseReason():
    def __init__(self, config, knowledge_model):
        self.config = config
        self.knowledge_model = knowledge_model
    
    def solve_qa(self, question, ref = None):
        raise NotImplementedError

    def solve_fc(self, question, ref = None):
        raise NotImplementedError

