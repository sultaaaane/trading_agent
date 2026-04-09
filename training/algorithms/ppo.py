import agentlightning as agl

verl_config = {
    "algorithm": {"adv_estimator": "grpo"},
    "data": {
        "train_batch_size": 4,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
    },
    "actor_rollout_ref": {
        "rollout": {
            "n": 2,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.6,
        },
        "actor": {"ppo_mini_batch_size": 4, "optim": {"lr": 1e-6}},
        "model": {
            "path": "meta-llama/Llama-3-8B-Instruct",
            "enable_gradient_checkpointing": True,
        },
        "ref": {"fsdp_config": {"param_offload": True}},
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "total_epochs": 2,
        "logger": ["console", "wandb"],
        "project_name": "Trading-Agent-Training",
        "experiment_name": "PPO-Senior-Strategist",
        "total_training_steps": 10,
        "test_freq": 5,
        "save_freq": 5,
    },
}

ppo_algorithm = agl.VERL(verl_config)
