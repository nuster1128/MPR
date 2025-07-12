from BaseReason import BaseReason

class DecompositionReason(BaseReason):
    def __init__(self, config, knowledge_model):
        super().__init__(config, knowledge_model)

    def solve_qa(self, question, ref = None):
        # Divide
        sub_question_list = self.knowledge_model.inference('Decomposition_Divide', {
            'question': question,
            'max_sub_question': self.config['decompose_num']
        }).split('\n')[:self.config['decompose_num']]
        print(sub_question_list)
        sub_answer_list = []

        # Solution
        for qid, sub_question in enumerate(sub_question_list):
            current_state = []
            for qqid in range(len(sub_question_list)):
                if qqid < qid:
                    current_state.append('%d. %s (Answer: %s)' % (qqid+1, sub_question_list[qqid], sub_answer_list[qqid]))
                elif qqid == qid:
                    current_state.append('%d. %s (Current Sub-question)' % (qqid+1, sub_question_list[qqid]))
                else:
                    current_state.append('%d. %s' % (qqid+1, sub_question_list[qqid]))
            current_state = '\n'.join(current_state)
            response = self.knowledge_model.response ('Decomposition_Solve', {
                'question': question,
                'current_state': current_state,
                'sub_question': sub_question
            }, ref, f'(question) {question}\n (sub-question) {sub_question}')
            sub_answer_list.append(response)
        
        # Merge
        current_state = ['%d. %s (Answer: %s)' % (qqid+1, sub_question_list[qqid], sub_answer_list[qqid]) for qqid in range(len(sub_question_list))]
        current_state = '\n'.join(current_state)
        response = self.knowledge_model.response ('Decomposition_Merge', {
            'question': question,
            'current_state': current_state
        }, ref, current_state)

        return response

    def solve_fc(self, question, ref = None):
        pass
