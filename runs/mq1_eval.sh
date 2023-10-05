
CUDA_VISIBLE_DEVICES=0 python src/main.py NuTrea \
    --model_name NuTrea \
    --data_folder data/metaqa-1hop/ \
    --lm sbert \
    --relation_word_emb True \
    --name metaqa \
    --entity_dim 50 \
    --num_iter 2 \
    --num_expansion_ins 2 \
    --num_backup_ins 3 \
    --backup_depth 1 \
    --num_layers 2 \
    --context_coef 1.0 \
    --rf_ief \
    --is_eval \
    --load_experiment mq1_h1.ckpt \
    # --load_experiment wqp_f1.ckpt \


# main.py ReaRev 
# --model_name ExeGNN --seed 24
#  --agg max --optimizer radam 
#  --hscore_coef 1.0 --num_iter 2 
#  --num_ins 2 --num_con 3 --depth 1 
#  --num_gnn 2 --rf_ief --decay_rate 0.99 
#  --num_epoch 10 --batch_size 32
#   --eval_every 1 
#   --data_folder ReaRev_data/metaqa-1hop/ 
#   --lm sbert --relation_word_emb True 
#   --name metaqa --entity_dim 50 
#   --linear_dropout 0.2 --rf_ief_bias 0.0
#    --version 5.0 --experiment_name MQ3_eid4.2.24
#     --use_wandb