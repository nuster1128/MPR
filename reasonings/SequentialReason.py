from BaseReason import BaseReason

class SequentialReason(BaseReason):
    def __init__(self, config, knowledge_model):
        super().__init__(config, knowledge_model)

    def solve_qa(self, question, ref = None):
        max_step = self.config['max_step']
        previous_thoughts = []

        # Start Thought
        previous_thoughts.append(self.knowledge_model.response('Sequential_Start', {
            'max_step': max_step,
            'question': question
        }, ref, question))

        # Middle Thought
        for current_step in range(1, max_step-1):
            previous_thoughts.append(self.knowledge_model.response('Sequential_Middle', {
                'current_step': current_step+1,
                'max_step': max_step,
                'question': question,
                'previous_thoughts': '\n'.join(['(Step %d) %s' % (tid+1, t) for tid, t in enumerate(previous_thoughts)])
            }, ref, previous_thoughts[-1]))
        
        return self.knowledge_model.response('Sequential_Final', {
            'question': question,
            'previous_thoughts': '\n'.join(['(Step %d) %s' % (tid+1, t) for tid, t in enumerate(previous_thoughts)])
        }, ref, previous_thoughts[-1])

    def solve_fc(self, question, ref = None):
        pass
