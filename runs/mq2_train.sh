

CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/metaqa-2hop/ \
    --lm sbert \
    --relation_word_emb True \
    --name metaqa \
    --num_epoch 10 \
    --eval_every 1 \
    --batch_size 32 \
    --decay_rate 0.99 \
    --linear_dropout 0.2 \
    --num_iter 2 \
    --num_expansion_ins 2 \
    --num_backup_ins 3 \
    --backup_depth 1 \
    --num_layers 3 \
    --context_coef 1.0 \
    --rf_ief \
    --pos_emb \
    --experiment_name mq2_nutrea
