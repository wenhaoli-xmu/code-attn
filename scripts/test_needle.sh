#!/bin/bash

# Define parameters
chat_template="llama-2"
s_len=128
e_len=4096
step=128

# Define configurations
configs=(
    "llama2-7b-chat.json"
    "llama2-7b-chat-maskout98.json"
    "llama2-7b-chat-maskout90.json")

# Loop through each configuration
for config in "${configs[@]}"; do
    echo "Running with config: $config"
    
    # Run the needle in haystack
    python test_needle/run_needle_in_haystack.py \
        --env_conf test_needle/$config \
        --chat_template $chat_template \
        --s_len $s_len \
        --e_len $e_len \
        --step $step

    # Visualize the result
    python test_needle/viz.py \
        --env_conf test_needle/$config
done