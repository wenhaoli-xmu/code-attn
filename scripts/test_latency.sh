test_scripts=(
    "llama2-7b-chat.json"
    "llama2-7b-chat-maskout98.json")

prompt_length="[4096,8192,16384]"

for test_script in "${test_scripts[@]}"
do
    echo "Running latency test for ${test_script}..."
    python test_latency/test.py --env_conf "test_latency/${test_script}" --prompt_length $prompt_length

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done