
CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/CWQ/ \
    --lm sbert \
    --relation_word_emb True \
    --name cwq \
    --num_iter 2 \
    --num_expansion_ins 3 \
    --num_backup_ins 3 \
    --backup_depth 2 \
    --num_layers 3 \
    --context_coef 1.0 \
    --rf_ief \
    --pos_emb \
    --is_eval \
    --load_experiment cwq_f1.ckpt \
    # --load_experiment cwq_f1.ckpt \