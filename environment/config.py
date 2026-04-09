num_runners = 4
strategy_config = {
    "type": "cs",  # ClientServerExecutionStrategy
    "n_runners": num_runners,
    "server_port": 48000,
}
print(f"Distributed strategy configured for {num_runners} parallel runners.")

llm_proxy_config = {
    "port": 48001,
    "model_list": [
        # Route junior analyst requests to local Qwen2
        {
            "model_name": "Qwen/Qwen2-1.5B-Instruct",
            "litellm_params": {"model": "ollama/qwen2:1.5b"},
        },
        # Route senior strategist requests (PPO will update this dynamically)
        {
            "model_name": "senior_strategist_llm",
            "litellm_params": {"model": "ollama/llama3"},
        },
        # Route review board requests to Mixtral
        {
            "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "litellm_params": {"model": "ollama/mixtral"},
        },
    ],
}
