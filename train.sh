deepspeed \
    --include localhost:4,5,6,7 \
    train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf train/genacc19-14.json \
    --instance_per_cycle 512 \
    --max_prepare_workers 4 \
    --prepare_batch_size_per_gpu 1 \
    --backward_per_head \
    --max_que 4096 \
    --max_oth 4096 \
    --max_top 4096 \
    --beta 3.0 \
    --margin 30
