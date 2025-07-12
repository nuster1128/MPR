# ----- Naive Prompt Templates -----
Naive_RAG_Prompt_Template = """Please help me answer the following question based on the given information.
Information:
{references}

Question:
{question}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

Naive_Ignoramus_Prompt_Template = """Please help me answer the following question.
Question:
{question}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

# ----- Decomposition Prompt Templates -----
Decomposition_Divide_RAG_Prompt_Template = """To better answer the following question, please break it down into several sub-questions.
Question:
{question}

Requirements:
1. The sub-questions should be concise.
2. Each sub-questions is on a separate line, without any other descriptions.
3. You can decompose the problem into 1 to {max_sub_question} sub-questions, and any content beyond 5 lines will be ignored."""

Decomposition_Divide_Ignoramus_Prompt_Template = """To better answer the following question, please break it down into several sub-questions.
Question:
{question}

Requirements:
1. The sub-questions should be concise.
2. Each sub-questions is on a separate line, without any other descriptions.
3. You can decompose the problem into 1 to {max_sub_question} sub-questions, and any content beyond 5 lines will be ignored."""


Decomposition_Solve_RAG_Prompt_Template = """In order to better answer the following question, we have decomposed them into several sub-questions.
Please help me generate the answer to the current sub-question based on the given information.
Question:
{question}

Sub-questions:
{current_state}

Current Sub-question:
{sub_question}

Information:
{references}

Requirements:
1. The answer should be concise but informative.
2. You should only output the answer in one line (no code block), without any other descriptions."""

Decomposition_Solve_Ignoramus_Prompt_Template = """In order to better answer the following question, we have decomposed them into several sub-questions.
Please help me generate the answer to the current sub-question.
Question:
{question}

Sub-questions:
{current_state}

Current Sub-question:
{sub_question}

Requirements:
1. The answer should be concise but informative.
2. You should only output the answer in one line (no code block), without any other descriptions."""

Decomposition_Merge_RAG_Prompt_Template = """In order to better answer the following question, we have decomposed them into several sub-questions and answered them separately.
Please help me generate the final answer to the question based on the sub-questions and the given information.
Question:
{question}

Sub-questions and Answers:
{current_state}

Information:
{references}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

Decomposition_Merge_Ignoramus_Prompt_Template = """In order to better answer the following question, we have decomposed them into several sub-questions and answered them separately.
Please help me generate the final answer to the question based on the sub-questions.
Question:
{question}

Sub-questions and Answers:
{current_state}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

# ----- Sequential Prompt Templates -----

Sequential_Start_RAG_Prompt_Template = """To better answer the following question, let's think step by step.
The current step is 1, and you should provide the final answer at step {max_step}.
Please generate your thoughts for the current step, and you may refer to the given information.

Information:
{references}

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Sequential_Start_Ignoramus_Prompt_Template = """To better answer the following question, let's think step by step.
The current step is 1, and you should provide the final answer at step {max_step}.
Please generate your thoughts for the current step.

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Sequential_Middle_RAG_Prompt_Template = """To better answer the following questions, let's think step by step.
The current step is {current_step}, and you should provide the final answer at step {max_step}.
Please generate your thoughts for the current step, and you may refer to the given information and your previous thought.

Information:
{references}

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Sequential_Middle_Ignoramus_Prompt_Template = """To better answer the following questions, let's think step by step.
The current step is {current_step}, and you should provide the final answer at step {max_step}.
Please generate your thoughts for the current step, and you may refer to your previous thought.

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Sequential_Final_RAG_Prompt_Template = """To better answer the following question, let's think step by step.
Please help me generate the answer to the question based on the given information and your previous thoughts.

Information:
{references}

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

Sequential_Final_Ignoramus_Prompt_Template = """To better answer the following question, let's think step by step.
Please help me generate the answer to the question based on your previous thoughts.

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

# ----- Structured Prompt Templates -----

Structured_Start_RAG_Prompt_Template = """To better answer the following question, let's think step by step.
Please generate your thoughts for the current step, and you may refer to the given information.

Information:
{references}

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Structured_Start_Ignoramus_Prompt_Template = """To better answer the following question, let's think step by step.
Please generate your thoughts for the current step.

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Structured_Middle_RAG_Prompt_Template = """To better answer the following questions, let's think step by step.
Please generate your thoughts for the current step, and you may refer to the given information and your previous thoughts.

Information:
{references}

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Structured_Middle_Ignoramus_Prompt_Template = """To better answer the following questions, let's think step by step.
Please generate your thoughts for the current step, and you may refer to your previous thoughts.

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. Your thoughts should be concise but informative sentences.
2. You should only output the thoughts in one line (no code block)."""

Structured_Final_RAG_Prompt_Template = """To better answer the following question, let's think step by step.
Please help me generate the answer to the question based on the given information and your previous thoughts.

Information:
{references}

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

Structured_Final_Ignoramus_Prompt_Template = """To better answer the following question, let's think step by step.
Please help me generate the answer to the question based on your previous thoughts.

Previous Thoughts:
{previous_thoughts}

Question:
{question}

Requirements:
1. your answer should be as concise as possible, commonly in a few words.
2. if the answer is a date, please output it in YYYY-MM-DD format.
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions."""

Structured_Selection_RAG_Prompt_Template = """To better answer the following question, let's think step by step.
Here are multiple reasoning paths. Please select the most likely one to answer the question and output its number.

Reasoning Paths:
{reasoning_paths}

Question:
{question}

Requirements:
You should only output a single integer as the index of the selected path, without any other descriptions."""

Structured_Selection_Ignoramus_Prompt_Template = Structured_Selection_RAG_Prompt_Template

# ----- MaskSFT Training Prompt Templates -----

MaskSFT_Mask_Prompt_Template = """Please help me identify the entities or relationships in the following message.
Message:
{message}

Requirements:
1. You should output at least two entities or relationships, and each of them should be in one line.
2. Each of the entities or relationships should only be a part of a sentence, not the entire sentence.
3. The output entities or relationships must match exactly with a part of the sentence (a substring of a sentence).
4. You should only output entities or relationships, without any other explanations."""

MaskSFT_QA_Convert_Prompt_Template = """What should be filled in the [MASK] position in the following sentence?
Message:
{message}

You should only output the text that needs to be filled in [MASK], without any other explanations."""

# ----- SelfAskSFT Training Prompt Templates -----
SelfAskSFT_Ask_Prompt_Template = """Please ask a question based on the following message and provide the corresponding answer.
Message:
{message}

Requirements:
1. Your output should only include two lines, the first line is the question and the second line is the answer.
2. Your output should not include any other explanations, and prompy words like 'Question:' and 'Answer:'.
3. The answer to the question should be the subject, object or predicate, not Yes or No."""