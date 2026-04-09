import agentlightning as agl
from typing import cast
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from data.dataset import TradingTask
from agents.prompts import prompts
from agents.tools import all_tools
from agents.runners import create_agent_runner
from graph.workflow import build_graph
from training.reward import get_weighted_reward


class TradingIntelligenceAgent(agl.LitAgent):
    def __init__(self, graph, reward_func):
        super().__init__()
        self.graph = graph
        self.reward_func = reward_func

    def rollout(
        self, task: TradingTask, resources: agl.NamedResources, rollout: agl.Rollout
    ) -> None:
        print(f"\n-- Starting Rollout {rollout.rollout_id} for Task: {task['id']} --")

        # Resolve the model-under-training resource from the rollout resources.
        llm_resource = cast(agl.LLM, resources.get("senior_strategist_llm"))
        if llm_resource is None:
            llm_resource = next(
                (
                    cast(agl.LLM, resource)
                    for resource in resources.values()
                    if isinstance(resource, agl.LLM)
                ),
                None,
            )
        if llm_resource is None:
            print(
                f"Rollout failed: No LLM resource found in resources: {list(resources.keys())}"
            )
            agl.emit_reward(0.0)
            return

        langchain_callback = self.trainer.tracer.get_langchain_handler()
        if hasattr(llm_resource, "get_base_url"):
            attempt_id = getattr(rollout, "attempt_id", "0")
            base_url = llm_resource.get_base_url(rollout.rollout_id, attempt_id)
        else:
            base_url = llm_resource.endpoint

        # Create a new ChatOpenAI instance pointed at the training server's endpoint.
        # We construct a fresh instance rather than using with_config, because
        # openai_api_base is a constructor parameter, not a runtime config key.
        llm_with_endpoint = ChatOpenAI(
            model="senior_strategist_llm",
            temperature=0.1,
            openai_api_base=base_url,
            openai_api_key=llm_resource.api_key or "dummy-key",
        )

        # Create fresh runners bound to the model-under-training
        refiner_trained = create_agent_runner(
            llm_with_endpoint, prompts["StrategyRefiner"], all_tools
        )
        designer_trained = create_agent_runner(
            llm_with_endpoint, prompts["ExecutionPlanDesigner"], all_tools
        )

        # Rebuild the graph with our trained runners swapped in.
        # We call build_graph() again and override the relevant runners before compiling.
        # This avoids relying on internal .copy() or .nodes mutation on compiled graphs,
        # which may not be supported in all LangGraph versions.
        workflow = build_graph(
            strategy_refiner_override=refiner_trained,
            execution_plan_override=designer_trained,
        )
        runnable = workflow.compile()

        # Execute the full workflow
        initial_state = {
            "trading_goal": task["goal"],
            "messages": [HumanMessage(content=task["goal"])],
            "turn_count": 0,
            "initial_theses": [],
        }
        config = {"callbacks": [langchain_callback]} if langchain_callback else {}

        try:
            final_state = runnable.invoke(initial_state, config=config)
            final_strategy = final_state.get("final_strategy")

            if final_strategy:
                reward_scores = self.reward_func(final_strategy, task["context"])
                final_reward = get_weighted_reward(reward_scores)
            else:
                final_reward = 0.0
        except Exception as e:
            print(f"Rollout failed: {e}")
            final_reward = 0.0

        agl.emit_reward(final_reward)
        print(
            f"-- Rollout {rollout.rollout_id} Finished. Reward: {final_reward:.2f} --"
        )

    async def rollout_async(
        self, task: TradingTask, resources: agl.NamedResources, rollout: agl.Rollout
    ) -> None:
        """Asynchronous rollout version calling the synchronous one."""
        return self.rollout(task, resources, rollout)
