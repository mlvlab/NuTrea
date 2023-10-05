
CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/metaqa-3hop/ \
    --lm sbert \
    --relation_word_emb True \
    --name metaqa \
    --entity_dim 50 \
    --num_iter 2 \
    --num_expansion_ins 2 \
    --num_backup_ins 3 \
    --backup_depth 2 \
    --num_layers 3 \
    --context_coef 1.0 \
    --rf_ief \
    --pos_emb \
    --is_eval \
    --load_experiment mq3_f1.ckpt \
    # --load_experiment mq3_f1.ckpt \



# main.py ReaRev --model_name ExeGNN 
# --seed 15 --num_iter 2 --num_ins 2 
# --num_con 3 --depth 2 --agg max --optimizer radam 
# --num_gnn 3 --hscore_coef 1.0 --rf_ief --decay_rate 0.99
#  --num_epoch 20 --batch_size 32 --eval_every 2 
#  --data_folder ReaRev_data/metaqa-3hop/ --lm sbert 
#  --relation_word_emb True --name metaqa --entity_dim 50 
#  --linear_dropout 0.2 --rf_ief_bias 0.0 --version 5.0 
#  --experiment_name MQ3_eid2.11.15 --use_wandb --pos_emb