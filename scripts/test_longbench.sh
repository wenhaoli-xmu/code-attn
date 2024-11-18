test_scripts=(
    "llama2-7b-chat.json"
    "llama2-7b-chat-maskout98.json"
    "llama2-7b-chat-maskout90.json")

chat_template="llama-2"
model_max_length=4096

for test_script in "${test_scripts[@]}"
do
    echo "Running prediction for ${test_script}..."
    python test_longbench/pred.py --env_conf "test_longbench/${test_script}" --chat_template $chat_template --model_max_length $model_max_length

    echo "Evaluating model for ${test_script}..."
    python LongBench/eval.py --model "${test_script}"

    echo "Displaying results for ${test_script}..."
    cat "pred/${test_script}/result.json"

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done