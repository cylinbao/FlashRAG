from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse

def naive(args):
    # save_note = 'naive'
    # save_note = f'naive_batch_{args.batch_size}_topk_{args.retrieval_topk}_nprobe_{args.retrieval_nprobe}'
    save_note = f'topk_{args.retrieval_topk}_nprobe_{args.retrieval_nprobe}'
    config_dict = {
        'method_name': "ric",
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_batch_size': args.batch_size,
        'retrieval_topk': args.retrieval_topk,
        'retrieval_nprobe': args.retrieval_nprobe,
    }

    from flashrag.pipeline import SequentialPipeline
    # preparation
    # config = Config('my_config.yaml',config_dict)
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config, verbose=False)
    
    result = pipeline.run(test_data, batch_size=-1)

def zero_shot(args):
    save_note = 'zero-shot'
    config_dict = {
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_batch_size': args.batch_size,
        # 'retrieval_topk': args.retrieval_topk,
        # 'retrieval_nprobe':args.retrieval_nprobe,
    }

    # preparation
    # config = Config('my_config.yaml',config_dict)
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate
    templete = PromptTemplate(
        config = config,
        system_prompt =  "Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt = "Question: {question}"
    )
    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config, templete, no_retrieval=True)
    result = pipeline.naive_run(test_data)

def hyde(args):
    save_note = f'topk_{args.retrieval_topk}_nprobe_{args.retrieval_nprobe}'
    config_dict = {
        'method_name': "hyde",
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_batch_size': args.batch_size,
        'retrieval_topk': args.retrieval_topk,
        'retrieval_nprobe': args.retrieval_nprobe,
    }

    from flashrag.pipeline import HyDEPipeline
    # preparation
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = HyDEPipeline(config)
    
    result = pipeline.run(test_data, batch_size=-1)

def LlamaIndexIter(args):
    pipe_version = args.pipeline_version # "v1"
    save_note = f'topk_{args.retrieval_topk}_nprobe_{args.retrieval_nprobe}_{pipe_version}'
    config_dict = {
        'method_name': "llamaindex_inter",
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_batch_size': args.batch_size,
        'retrieval_topk': args.retrieval_topk,
        'retrieval_nprobe': args.retrieval_nprobe,
    }

    from flashrag.pipeline import LlamaIndexIterativePipeline
    from flashrag.prompt import PromptTemplate, DEFAULT_TREE_SUMMARIZE_TMPL

    # preparation
    config = Config(args.config_file, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pred_process_fun = lambda x: x.split("\n")[0]

    # prompt_template = PromptTemplate(
    #     config = config,
    #     system_prompt =  DEFAULT_TREE_SUMMARIZE_TMPL,
    #     user_prompt = (
    #         "Query: {question}\n"
    #         "Answer: "
    #     )
    # )
    # pipeline = LlamaIndexIterativePipeline(config, prompt_template=prompt_template)
    pipeline = LlamaIndexIterativePipeline(config, iter_num=3, pipeline_version=pipe_version)
    
    result = pipeline.run(test_data)

def aar(args):
    """
    Reference:
        Zichun Yu et al. "Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In"
        in ACL 2023.
        Official repo: https://github.com/OpenMatch/Augmentation-Adapted-Retriever
    """
    # two types of checkpoint: ance / contriever
    #retrieval_method = "AAR-contriever"  # AAR-ANCE
    # index path of this retriever
    retrieval_method = args.method_name
    if 'contriever' in retrieval_method:
        index_path = "aar-contriever_Flat.index"
    else:
        index_path = "aar-ance_Flat.index"

    model2path = {"AAR-contriever": "model/AAR-Contriever-KILT",
                            "AAR-ANCE": "model/AAR-ANCE"}
    model2pooling = {"AAR-contriever": "mean",
                    "AAR-ANCE": "cls"}
    save_note = retrieval_method
    config_dict = {
        'retrieval_method': retrieval_method,
        'model2path': model2path,
        'index_path': index_path,
        'model2pooling': model2pooling,
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }

    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pred_process_fun = lambda x: x.split("\n")[0]
    pipeline = SequentialPipeline(config)
    #result = pipeline.run(test_data, pred_process_fun=pred_process_fun)
    result = pipeline.run(test_data)

def llmlingua(args):
    """
    Reference:
        Huiqiang Jiang et al. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
        in EMNLP 2023
        Huiqiang Jiang et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression"
        in ICLR MEFoMo 2024.
        Official repo: https://github.com/microsoft/LLMLingua
    """
    refiner_name = "longllmlingua" #
    refiner_model_path = "model/llama-2-7b-hf"

    config_dict = {
        'refiner_name': refiner_name,
        'refiner_model_path': refiner_model_path,
        'llmlingua_config':{
            'rate': 0.55,
            'condition_in_question': 'after_condition',
            'reorder_context': 'sort',
            'dynamic_context_compression_ratio': 0.3,
            'condition_compare': True,
            'context_budget': "+100",
            'rank_method': 'longllmlingua'
        },
        'refiner_input_prompt_flag': False,
        'save_note':'longllmlingua',
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }

    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)

def recomp(args):
    """
    Reference:
        Fangyuan Xu et al. "RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation"
        in ICLR 2024.
        Official repo: https://github.com/carriex/recomp
    """
    # ###### Specified parameters ######
    refiner_name = "recomp-abstractive" # recomp-extractive
    model_dict = {'nq':"model/recomp_nq_abs",
                  'triviaqa':"model/recomp_tqa_abs",
                  'hotpotqa':'model/recomp_hotpotqa_abs'}

    refiner_model_path = model_dict.get(args.dataset_name, None)
    refiner_max_input_length = 1024
    refiner_max_output_length = 512
    # parameters for extractive compress
    refiner_topk = 5
    refiner_pooling_method = 'mean'
    refiner_encode_max_length = 256


    config_dict = {
        'refiner_name': refiner_name,
        'refiner_model_path': refiner_model_path,
        'refiner_max_input_length': refiner_max_input_length,
        'refiner_max_output_length': refiner_max_output_length,
        'refiner_topk': 5,
        'refiner_pooling_method': refiner_pooling_method,
        'refiner_encode_max_length': refiner_encode_max_length,
        'save_note': refiner_name,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }


    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)

def sc(args):
    """
    Reference:
        Yucheng Li et al. "Compressing Context to Enhance Inference Efficiency of Large Language Models"
        in EMNLP 2023.
        Official repo: https://github.com/liyucheng09/Selective_Context

    Note: 
        Need to install spacy:
            ```python -m spacy download en_core_web_sm```
        or 
            ```
            wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0.tar.gz  
            pip install en_core_web_sm-3.6.0.tar.gz
            ```
    """
    refiner_name = "selective-context"
    refiner_model_path = "model/gpt2"

    config_dict = {
        'refiner_name': refiner_name,
        'refiner_model_path': refiner_model_path,
        'sc_config':{
            'reduce_ratio': 0.5
        },
        'save_note': 'selective-context',
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }


    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data)



def retrobust(args):
    """
    Reference:
        Ori Yoran et al. "Making Retrieval-Augmented Language Models Robust to Irrelevant Context"
        in ICLR 2024.
        Official repo: https://github.com/oriyor/ret-robust
    """
    model_dict = {'nq':'model/llama-2-13b-peft-nq-retrobust',
                  '2wiki':'model/llama-2-13b-peft-2wikihop-retrobust'}
    if args.dataset_name in ['nq', 'triviaqa', 'popqa','web_questions']:
        lora_path = model_dict['nq']
    elif args.dataset_name in ['hotpotqa',"2wikimultihopqa"]:
        lora_path = model_dict['2wiki']
    else:
        print("Not use lora")
        lora_path = model_dict.get(args.dataset_name,None)
    config_dict = {'save_note':'Ret-Robust',
               'generator_model': 'llama2-13B',
               'generator_lora_path': lora_path,
               'generation_params':{"max_tokens":100},
               'gpu_id':args.gpu_id,
               'generator_max_input_len': 4096,
               'dataset_name':args.dataset_name}
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SelfAskPipeline
    from flashrag.utils import selfask_pred_parse
    pipeline = SelfAskPipeline(config, max_iter=5, single_hop=False)
    # use specify prediction parse function
    result = pipeline.run(test_data, pred_process_fun=selfask_pred_parse)

def sure(args):
    """
    Reference:
        Jaehyung Kim et al. "SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs"
        in ICLR 2024
        Official repo: https://github.com/bbuing9/ICLR24_SuRe
    """
    config_dict = {
            'save_note': 'SuRe',
            'gpu_id':args.gpu_id,
            'dataset_name':args.dataset_name
            }
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SuRePipeline
    pipeline = SuRePipeline(config)
    pred_process_fun = lambda x: x.split("\n")[0]
    result = pipeline.run(test_data)

def replug(args):
    """
    Reference:
        Weijia Shi et al. "REPLUG: Retrieval-Augmented Black-Box Language Models".
    """
    save_note = f'topk_{args.retrieval_topk}_nprobe_{args.retrieval_nprobe}'
    config_dict = {
        'framework': 'hf' ,
        'method_name': args.method_name,
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_batch_size': args.batch_size,
        'retrieval_topk': args.retrieval_topk,
        'retrieval_nprobe': args.retrieval_nprobe,
    }

    # preparation
    config = Config('base_config.yaml', config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    pred_process_fun = lambda x: x.split("\n")[0]

    from flashrag.pipeline import REPLUGPipeline
    pipeline = REPLUGPipeline(config)
    result = pipeline.run(test_data)

def skr(args):
    """
    Reference:
        Yile Wang et al. "Self-Knowledge Guided Retrieval Augmentation for Large Language Models"
        in EMNLP Findings 2023.
        Official repo: https://github.com/THUNLP-MT/SKR/

    Note:
        `skr-knn` need training data in inference stage to determain whether to retrieve. training data should in
        `.json` format in following format:
        format: 
            [
                {
                    "question": ... ,  // question
                    "judgement": "ir_better" / "ir_worse" / "same",  // judgement result, can be obtained by comparing 
                    ...
                },
                ...
            ]

    """
    judger_name = 'skr'
    model_path = 'model/sup-simcse-bert-base-uncased'
    training_data_path = './sample_data/skr_training.json'

    config_dict = {
        'judger_name': judger_name,
        'judger_model_path': model_path,
        'judger_training_data_path': training_data_path,
        'judger_topk': 5,
        'save_note': 'skr',
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name
    }


    # preparation
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ConditionalPipeline
    pipeline = ConditionalPipeline(config)
    result = pipeline.run(test_data)

def selfrag(args):
    """
    Reference:
        Akari Asai et al. " SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection"
        in ICLR 2024.
        Official repo: https://github.com/AkariAsai/self-rag
    """
    config_dict = {'generator_model':'selfrag-llama2-7B',
                'generator_model_path': 'model/selfrag_llama2_7b',
               'framework': 'vllm',
               'save_note':'self-rag',
               'gpu_id':args.gpu_id,
               'generation_params':{'max_new_tokens':100,'temperature':0.0,'top_p':1.0,'skip_special_tokens':False},
                'dataset_name':args.dataset_name}
    config = Config('my_config.yaml', config_dict)


    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SelfRAGPipeline
    pipeline = SelfRAGPipeline(config, threhsold=0.2, max_depth=2, beam_width=2,
                            w_rel=1.0, w_sup=1.0, w_use=1.0,
                            use_grounding=True, use_utility=True, use_seqscore=True, ignore_cont=True,
                                mode='adaptive_retrieval')
    result = pipeline.run(test_data, batch_size=256)

def flare(args):
    """
    Reference:
        Zhengbao Jiang et al. "Active Retrieval Augmented Generation"
        in EMNLP 2023.
        Official repo: https://github.com/bbuing9/ICLR24_SuRe

    """
    config_dict={
        'save_note':'flare', 
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_nprobe':args.retrieval_nprobe,
    }
    config = Config('my_config.yaml',config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import FLAREPipeline
    pipeline = FLAREPipeline(config)
    result = pipeline.run(test_data)

def iterretgen(args):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023. 
    """
    iter_num = 3

    save_note = f'topk_{args.retrieval_topk}_nprobe_{args.retrieval_nprobe}'
    config_dict = {
        'method_name': args.method_name,
        'save_note': save_note,
        'gpu_id':args.gpu_id,
        'dataset_name':args.dataset_name,
        'retrieval_batch_size': args.batch_size,
        'retrieval_topk': args.retrieval_topk,
        'retrieval_nprobe': args.retrieval_nprobe,
    }

    # preparation
    config = Config('base_config.yaml', config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IterativePipeline
    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Running exp")
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--method_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--dataset_name',type=str)
    parser.add_argument('--gpu_id', type=str)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--retrieval_topk', type=int)
    parser.add_argument('--retrieval_nprobe', type=int)
    parser.add_argument('--pipeline_version', type=str, default="v1")

    func_dict = {
        'AAR-contriever': aar,
        'AAR-ANCE': aar,
        'naive': naive,
        'zero-shot': zero_shot,
        'llmlingua': llmlingua,
        'recomp': recomp,
        'selective-context': sc,
        'ret-robust': retrobust,
        'sure': sure,
        'replug': replug,
        'skr': skr,
        'selfrag': selfrag,
        'flare': flare,
        'iterretgen': iterretgen,
        'hyde': hyde,
        'iter': LlamaIndexIter
    }

    args = parser.parse_args()

    func = func_dict[args.method_name]
    func(args)

