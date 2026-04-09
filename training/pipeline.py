import agentlightning as agl
import asyncio
import os
from environment.config import llm_proxy_config, num_runners, strategy_config
from monitoring.hooks import custom_hook
from agents.lit_agent import TradingIntelligenceAgent
from graph.workflow import trading_graph
from training.reward import get_weighted_reward, strategy_evaluator
from data.dataset import train_dataset, val_dataset
from training.algorithms.sft import sft_algorithm
from training.algorithms.ppo import ppo_algorithm
from training.adapter import custom_adapter
from training.algorithms.bandit import bandit_algorithm


def full_training_pipeline():
    print("--- CONFIGURING FULL TRAINING PIPELINE ---")

    # Shared components
    store = agl.InMemoryLightningStore()
    llm_proxy = agl.LLMProxy(
        port=llm_proxy_config["port"],
        model_list=llm_proxy_config["model_list"],
        store=store,
    )
    initial_resources = {
        "senior_strategist_llm": llm_proxy.as_resource(model="senior_strategist_llm")
    }
    tracer = agl.AgentOpsTracer()

    # Phase 1: Initial Data Gathering (baseline model)
    print("\n--- Phase 1: Initial Data Gathering ---")
    gather_trainer = agl.Trainer(
        n_runners=num_runners,
        strategy=strategy_config,
        store=store,
        tracer=tracer,
        llm_proxy=llm_proxy,
        initial_resources=initial_resources,
        hooks=[custom_hook],
    )
    agent_gather = TradingIntelligenceAgent(
        trading_graph, lambda s, c: get_weighted_reward(strategy_evaluator(s, c))
    )
    gather_trainer.dev(agent_gather, train_dataset[:10])
    succeeded_rollouts = asyncio.run(store.query_rollouts(status=["succeeded"]))
    has_successful_data = len(succeeded_rollouts) > 0

    # Phase 2: SFT on Junior Analysts
    if has_successful_data:
        print("\n--- Phase 2: SFT on Junior Analysts ---")
        sft_trainer = agl.Trainer(
            algorithm=sft_algorithm, store=store, llm_proxy=llm_proxy
        )
        sft_trainer.fit(agent_gather)
    else:
        print(
            "\n--- Phase 2: SFT on Junior Analysts (skipped: no successful rollouts) ---"
        )

    # Phase 3: PPO on Senior Strategist
    enable_ppo = os.environ.get("ENABLE_PPO_PHASE", "0") == "1"
    if enable_ppo:
        print("\n--- Phase 3: PPO on Senior Strategist ---")
        ppo_trainer = agl.Trainer(
            algorithm=ppo_algorithm,
            n_runners=num_runners,
            strategy=strategy_config,
            store=store,
            tracer=tracer,
            adapter=custom_adapter,
            hooks=[custom_hook],
        )
        agent_ppo = TradingIntelligenceAgent(
            trading_graph, lambda s, c: get_weighted_reward(strategy_evaluator(s, c))
        )
        ppo_trainer.fit(agent_ppo, train_dataset=train_dataset, val_dataset=val_dataset)
    else:
        print("\n--- Phase 3: PPO on Senior Strategist (skipped by default) ---")
        print(
            "Set ENABLE_PPO_PHASE=1 to enable PPO once Ray/VERL serialization is configured."
        )

    # Phase 4: Contextual Bandit on CIO
    if has_successful_data:
        print("\n--- Phase 4: Contextual Bandit on CIO ---")
        bandit_trainer = agl.Trainer(algorithm=bandit_algorithm, store=store)
        bandit_trainer.fit(agent_gather)
    else:
        print(
            "\n--- Phase 4: Contextual Bandit on CIO (skipped: no successful rollouts) ---"
        )

    print("\n--- Hierarchical Training Pipeline Complete ---")


full_training_pipeline()
