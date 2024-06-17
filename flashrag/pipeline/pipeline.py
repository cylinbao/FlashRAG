from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
import numpy as np
import time
from tqdm import tqdm
import os
from flashrag.prompt import PromptTemplate

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template = None):
        self.config = config
        self.device = config['device']
        self.save_dir = config['save_dir']
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config['save_retrieval_cache']
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output('raw_pred',raw_pred)
            dataset.update_output('pred', processed_pred)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset

    def save_profile_time(self, profile_dict):
        file_name = "profile_time.txt"
        save_path = os.path.join(self.save_dir, file_name)
        with open(save_path, "w", encoding='utf-8') as f:
            for k,v in profile_dict.items():
                f.write(f"{k}: {v}\n")

    
class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template = None, verbose=False, no_retrieval=False):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if no_retrieval == False:
            self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        self.avg_gen_t = None
        self.avg_ret_t = None
        self.verbose = verbose

        # TODO: add rewriter module
        self.use_fid = config['use_fid']
        self.batch_size = config['retrieval_batch_size']
        if no_retrieval == False:
            self.retrieval_topk = config['retrieval_topk']
            self.retrieval_nprobe = config['retrieval_nprobe']

        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output('prompt', input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None, batch_size=None):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output('retrieval_query', input_query)
        dataset.update_output('retrieval_result', retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if 'llmlingua' in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output('prompt', input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output('refine_result', refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
        dataset.update_output('prompt', input_prompts)

        if self.use_fid:
            print('Use FiD generation')
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append(
                    [q + " " + doc for doc in docs]
                )

        _batch_size = self.batch_size if batch_size == None else batch_size 

        if _batch_size > 0:
            pred_answer_list = []
            for start_idx in tqdm(range(0, len(input_prompts), _batch_size), desc='Generation process: '):
                prompts_batch = input_prompts[start_idx:start_idx + _batch_size]
                pred_answers = self.generator.generate(prompts_batch)
                pred_answer_list.extend(pred_answers)
        else:
            pred_answer_list = self.generator.generate(input_prompts)

        if self.verbose:
            gen_t = np.array(self.generator.gen_t)
            # discard the first three runs for better measurement
            if gen_t.shape[0] > 3:
                gen_t = gen_t[3:]
            self.avg_gen_t = np.mean(gen_t)

            retrieval_t = self.retriever.retrieval_t[0]
            self.avg_ret_t = retrieval_t["emb"] + retrieval_t["retrieve"]

        # if self.verbose:
            print(f"Averaged generation time (ms): {self.avg_gen_t*1000:.3f}")
            print(f"Averaged retrieval time (ms): {self.avg_ret_t*1000:.3f}")
            profile_dict = {
                "generation_batch_size": _batch_size,
                "retrieval_batch_size": self.batch_size,
                "retrieval_topk": self.retrieval_topk,
                "retrieval_nprobe": self.retrieval_nprobe,
                "avg_gen_t_per_batch": self.avg_gen_t*1000,
                "avg_tot_retrival_t_per_batch": self.avg_ret_t*1000,
                "avg_emb_t_per_batch": retrieval_t["emb"]*1000,
                "avg_retrieve_t_per_batch": retrieval_t["retrieve"]*1000
            }
            self.save_profile_time(profile_dict)

        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset

class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template = None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)
        self.judger = get_judger(config)

        self.sequential_pipeline = SequentialPipeline(config, prompt_template)
        from flashrag.prompt import PromptTemplate
        self.zero_shot_templete = PromptTemplate(
            config = config,
            system_prompt =  "Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt = "Question: {question}"
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output('judge_result', judge_result)

        # split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class HyDEPipeline(BasicPipeline):
    def __init__(self, config, prompt_template = None, verbose=False, no_retrieval=False):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        if no_retrieval == False:
            self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        self.avg_gen_t = None
        self.avg_ret_t = None
        self.verbose = verbose

        # TODO: add rewriter module
        self.use_fid = config['use_fid']
        self.batch_size = config['retrieval_batch_size']
        if no_retrieval == False:
            self.retrieval_topk = config['retrieval_topk']
            self.retrieval_nprobe = config['retrieval_nprobe']

        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None
            
        from flashrag.prompt import PromptTemplate
        self.hyde_template = PromptTemplate(
            config = config,
            system_prompt =  """Please write a passage to answer the question\n
                              Try to include as many key details as possible.\n\n""",
            user_prompt = """Question: {question} \n\n
                             Passage: """
        )

    def query_transform(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.hyde_template.get_string(question=q) for q in dataset.question]
        dataset.update_output('hyde_prompt', input_prompts)

        hyde_gen_query = self.generator.generate(input_prompts)
        dataset.update_output("hyde_gen_query", hyde_gen_query)

        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None, batch_size=None):
        # input_query = dataset.question

        dataset = self.query_transform(dataset)

        input_query = dataset.hyde_gen_query

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output('retrieval_query', input_query)
        dataset.update_output('retrieval_result', retrieval_results)

        input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
        ]
        dataset.update_output('prompt', input_prompts)

        if self.use_fid:
            print('Use FiD generation')
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append(
                    [q + " " + doc for doc in docs]
                )

        _batch_size = self.batch_size if batch_size == None else batch_size 

        if _batch_size > 0:
            pred_answer_list = []
            for start_idx in tqdm(range(0, len(input_prompts), _batch_size), desc='Generation process: '):
                prompts_batch = input_prompts[start_idx:start_idx + _batch_size]
                pred_answers = self.generator.generate(prompts_batch)
                pred_answer_list.extend(pred_answers)
        else:
            pred_answer_list = self.generator.generate(input_prompts)

        if self.verbose:
            gen_t = np.array(self.generator.gen_t)
            # discard the first three runs for better measurement
            if gen_t.shape[0] > 3:
                gen_t = gen_t[3:]
            self.avg_gen_t = np.mean(gen_t)

            retrieval_t = self.retriever.retrieval_t[0]
            self.avg_ret_t = retrieval_t["emb"] + retrieval_t["retrieve"]

        # if self.verbose:
            print(f"Averaged generation time (ms): {self.avg_gen_t*1000:.3f}")
            print(f"Averaged retrieval time (ms): {self.avg_ret_t*1000:.3f}")
            profile_dict = {
                "generation_batch_size": _batch_size,
                "retrieval_batch_size": self.batch_size,
                "retrieval_topk": self.retrieval_topk,
                "retrieval_nprobe": self.retrieval_nprobe,
                "avg_gen_t_per_batch": self.avg_gen_t*1000,
                "avg_tot_retrival_t_per_batch": self.avg_ret_t*1000,
                "avg_emb_t_per_batch": retrieval_t["emb"]*1000,
                "avg_retrieve_t_per_batch": retrieval_t["retrieve"]*1000
            }
            self.save_profile_time(profile_dict)

        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset