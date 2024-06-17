from transformers import AutoTokenizer, AutoConfig

class PromptTemplate:
    placeholders = ['reference', 'question']
    base_system_prompt = "Answer the question based on the given document." \
                        "Only give me the answer and do not output any other words." \
                        "\nThe following are given documents.\n\n{reference}"
    base_user_prompt = "Question: {question}"

    def __init__(self,
                config,
                system_prompt = "",
                user_prompt = "",
                enable_chat = True
        ):

        self.config = config
        self.is_openai = config['framework'] == 'openai'
        if not self.is_openai:
            self.generator_path = config['generator_model_path']
            model_config = AutoConfig.from_pretrained(self.generator_path)
            model_name = model_config._name_or_path.lower()
            self.is_chat = False
            if 'chat' in model_name or 'instruct' in model_name:
                self.is_chat = True
                self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path)
        else:
            self.is_chat = True
            self.enable_chat = True

        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat

        self._check_placeholder()

    def _check_placeholder(self):
        # check placeholder in prompt
        for holder in self.placeholders:
            flag = False
            for prompt in [self.system_prompt, self.user_prompt]:
                if f'{holder}' in prompt:
                    print(f"Find `{holder}` in template")
                    flag = True
                    break
            if not flag and holder != 'reference':
                assert False

    def get_string(self,
                   question,
                   retrieval_result = None,
                   formatted_reference = None,
                   previous_gen = None,
                   **params
        ):

        if formatted_reference is None:
            if retrieval_result is not None:
                formatted_reference = self.format_reference(retrieval_result)
            else:
                formatted_reference = ""

        if previous_gen is None:
            formatted_previous_gen = None
        else:
            formatted_previous_gen = previous_gen

        input_params = {
            "question": question,
            "reference": formatted_reference,
            "previous_gen": formatted_previous_gen
        }
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role":"system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role":"user", "content": user_prompt})
            if self.is_openai:
                for item in input:
                    if item['role'] == 'system':
                        item['role'] == 'assistant'
            else:
                input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        if previous_gen is not None and self.is_openai is False:
            input += previous_gen

        return input


    def format_reference(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['contents']
            if "title" in doc_item.keys():
                title = doc_item['title']
                text = content
            else:
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1} (Title: {title}) {text}\n"

        return format_reference

DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {question}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below, as "
    "well as previous reasoning steps.\n"
    "Given the context and previous reasoning, return a question that can "
    "be answered from "
    "the context. This question can be the same as the original question, "
    "or this question can represent a subcomponent of the overall question."
    "It should not be irrelevant to the original question.\n"
    "If we cannot extract more information from the context, provide 'None' "
    "as the answer. "
    "Some examples are given below: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides names of the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None\n"
    "Next question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: Who was the winner of the 2020 Australian Open?\n"
    "Knowledge source context: Provides names of the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None.\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: None"
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open - includes biographical information for each winner\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: How many Grand Slam titles does Novak Djokovic have? "
    "\n\n"
    "Question: {question}\n"
    "Knowledge source context: {reference}\n"
    "Previous reasoning: {previous_gen}\n"
    "New question: "
)

DEFAULT_JUDGE_TMPL = (
    "The original question is as follows: {question}\n"
    "And we have an following answer: {answer_str}\n"
    "Please judge the answer is correct or not.\n"
    "If the answer is correct, please return '1', otherwise, please return '0'.\n"
    "Your judgement:"
)

DEFAULT_TREE_SUMMARIZE_TMPL = (
    "Context information from multiple sources is below.\n"
    "---------------------\n"
    "{reference}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge, "
    "answer the query.\n"
    # "Query: {question}\n"
    # "Answer: "
)
