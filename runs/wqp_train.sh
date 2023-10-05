
CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/webqsp/ \
    --lm sbert \
    --relation_word_emb True \
    --name webqsp \
    --num_epoch 100 \
    --eval_every 2 \
    --batch_size 8 \
    --decay_rate 0.99 \
    --linear_dropout 0.3 \
    --num_iter 2 \
    --num_expansion_ins 2 \
    --num_backup_ins 3 \
    --backup_depth 1 \
    --num_layers 2 \
    --context_coef 0.3 \
    --rf_ief \
    --experiment_name wqp_nutrea