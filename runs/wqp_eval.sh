
CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/webqsp/ \
    --lm sbert \
    --relation_word_emb True \
    --name webqsp \
    --num_iter 2 \
    --num_expansion_ins 2 \
    --num_backup_ins 3 \
    --backup_depth 1 \
    --num_layers 2 \
    --context_coef 0.3 \
    --rf_ief \
    --is_eval \
    --load_experiment wqp_h1.ckpt \
    # --load_experiment wqp_f1.ckpt \
