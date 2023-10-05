
CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/metaqa-2hop/ \
    --lm sbert \
    --relation_word_emb True \
    --name metaqa \
    --entity_dim 50 \
    --num_iter 2 \
    --num_expansion_ins 2 \
    --num_backup_ins 3 \
    --backup_depth 1 \
    --num_layers 3 \
    --context_coef 1.0 \
    --rf_ief \
    --pos_emb \
    --is_eval \
    --load_experiment mq2_h1.ckpt \
    # --load_experiment mq2_f1.ckpt \


