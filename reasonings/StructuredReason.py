from BaseReason import BaseReason

class StructuredReason(BaseReason):
    def __init__(self, config, knowledge_model):
        super().__init__(config, knowledge_model)

    def path_selection(self, path_buffer, question):
        res = self.knowledge_model.inference('Structured_Selection', {
            'question': question,
            'reasoning_paths': '\n'.join(['Path %d: %s' % (pid, ' '.join(['(Step %d) %s' % (sid+1, thoughts) for sid, thoughts in enumerate(path)])) for pid, path in enumerate(path_buffer)])
        })

        if res.isdigit():
            index = eval(res)
            if index < len(path_buffer):
                return index
            else:
                return 0
        else:
            return 0

    def solve_qa(self, question, ref = None):
        max_step = self.config['max_step']
        path_buffer = []

        # Start Thought
        start_thoughts = self.knowledge_model.response('Structured_Start', {
            'question': question
        }, ref, question)
        path_buffer.append([start_thoughts])

        # Middle Thought
        for current_step in range(1, max_step-1):
            current_path_id = self.path_selection(path_buffer, question)
            current_path = path_buffer[current_path_id]
            new_path_list = []
            for branch_id in range(self.config['branch_size']):
                new_thoughts = self.knowledge_model.response('Structured_Middle', {
                    'question': question,
                    'previous_thoughts': '\n'.join(['(Step %d) %s' % (tid+1, t) for tid, t in enumerate(current_path)])
                }, ref, current_path[-1])

                new_path = current_path + [new_thoughts]
                new_path_list.append(new_path)
            path_buffer = path_buffer[:current_path_id] + path_buffer[current_path_id:] + new_path_list

        # Final Thought
        current_path_id = self.path_selection(path_buffer, question)
        current_path = path_buffer[current_path_id]
        return self.knowledge_model.response('Structured_Final', {
            'question': question,
            'previous_thoughts': '\n'.join(['(Step %d) %s' % (tid+1, t) for tid, t in enumerate(current_path)])
        }, ref, current_path[-1])

    def solve_fc(self, question, ref = None):
        pass
