#!/usr/bin/env bash
RUN_NAME=$1
seed=42
OUTPUT_DIR=/root/autodl-tmp/HBGL/hogan_ztf/roberta_model/
CACHE_DIR=.cache
TRAIN_FILE=/root/autodl-tmp/HBGL/data/ztfData/train/train_data_clear_src_tgt.jsonl


MODEL_PATH=/root/.cache/huggingface/hub/models--hfl--chinese-roberta-wwm-ext-large/snapshots/a25cc9e05974bd9687e528edd516f2cfdb3f5db9
# MODEL_PATH=/root/.cache/huggingface/hub/models--hfl--chinese-bert-wwm-ext/snapshots/2a995a880017c60e4683869e817130d8af548486/


/root/autodl-tmp/HBGL/.conda/bin/python /root/autodl-tmp/HBGL/hogan_ztf/run_ztf.py\
    --train_file ${TRAIN_FILE} --output_dir ${OUTPUT_DIR}\
    --model_type bert --model_name_or_path  ${MODEL_PATH}\
    --fp16\
    --do_lower_case --max_source_seq_length 490 --max_target_seq_length 8\
    --per_gpu_train_batch_size 12 --gradient_accumulation_steps 2\
    --valid_file /root/autodl-tmp/HBGL/data/ztfData/eval/ori_eval_data_src_tgt.jsonl \
    --test_file /root/autodl-tmp/HBGL/data/ztfData/eval/ori_eval_data_src_tgt.jsonl \
    --add_vocab_file ./data/WebOfScience/label_map.pkl \
    --label_smoothing 0\
    --save_steps 550 \
    --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps -1 --cache_dir ${CACHE_DIR}\
    --soft_label --seed ${seed} \
    --label_cpt ./data/WebOfScience/wos.taxnomy --label_cpt_not_incr_mask_ratio --label_cpt_steps 300 --label_cpt_use_bce \
    --load_label_embedding_cache    ; /usr/bin/shutdown
