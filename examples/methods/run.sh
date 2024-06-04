GPU_ID="1"

python run_exp.py \
    --config_file base_config.yaml \
    --method_name naive \
    --gpu_id $GPU_ID \
    --dataset_name popqa \
    --split test \
    --batch_size 256 \
    --retrieval_topk 3 \
    --retrieval_nprobe 2048 \

# python run_exp.py \
#     --method_name naive \
#     --gpu_id $GPU_ID \
#     --dataset_name popqa \
#     --split test \
#     --retrieval_nprobe 1024

    # --dataset_name popqa \
    # --split test \
    # --split dev \
    # --model_path "/data/llama3/Meta-Llama-3-8B-Instruct-hf" \
    # --retriever_path "facebook/contriever-msmarco"

    # --method_name zero-shot \
    # --model_path "/data/llama2/Llama-2-7b-chat-hf" \