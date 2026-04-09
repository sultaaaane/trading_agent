from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agents.state import AgentState
from agents.llms import (
    junior_analyst_llm,
    cio_llm,
    senior_strategist_llm,
    review_board_llm,
)
from agents.prompts import prompts
from agents.tools import all_tools
from agents.runners import create_agent_runner
from agents.nodes import (
    create_agent_node,
    tools_condition,
    route_after_tools,
    route_after_review,
)


def build_graph(
    strategy_refiner_override=None, execution_plan_override=None
) -> StateGraph:
    workflow = StateGraph(AgentState)

    # Create all agent runners, using overrides when provided (for training)
    agent_runners = {
        "TechnicalAnalyst": create_agent_runner(
            junior_analyst_llm, prompts["TechnicalAnalyst"], all_tools
        ),
        "FundamentalAnalyst": create_agent_runner(
            junior_analyst_llm, prompts["FundamentalAnalyst"], all_tools
        ),
        "SentimentAnalyst": create_agent_runner(
            junior_analyst_llm, prompts["SentimentAnalyst"], all_tools
        ),
        "CIO": create_agent_runner(cio_llm, prompts["CIO"], []),
        "StrategyRefiner": strategy_refiner_override
        or create_agent_runner(
            senior_strategist_llm, prompts["StrategyRefiner"], all_tools
        ),
        "ExecutionPlanDesigner": execution_plan_override
        or create_agent_runner(
            senior_strategist_llm, prompts["ExecutionPlanDesigner"], all_tools
        ),
        "RiskAnalyst": create_agent_runner(
            review_board_llm, prompts["RiskAnalyst"], []
        ),
        "ComplianceOfficer": create_agent_runner(
            review_board_llm, prompts["ComplianceOfficer"], []
        ),
        "PortfolioManager": create_agent_runner(
            review_board_llm, prompts["PortfolioManager"], []
        ),
    }

    # Add all nodes
    for name, runner in agent_runners.items():
        workflow.add_node(name, create_agent_node(name, runner))
    workflow.add_node("execute_tools", ToolNode(all_tools))

    # Junior Analysts run in PARALLEL from the start
    workflow.add_edge(START, "TechnicalAnalyst")
    workflow.add_edge(START, "FundamentalAnalyst")
    workflow.add_edge(START, "SentimentAnalyst")

    # ReAct edges for tool-using agents
    for name in [
        "TechnicalAnalyst",
        "FundamentalAnalyst",
        "SentimentAnalyst",
        "StrategyRefiner",
        "ExecutionPlanDesigner",
    ]:
        next_node = (
            "CIO"
            if "Analyst" in name
            else (
                "ExecutionPlanDesigner" if name == "StrategyRefiner" else "RiskAnalyst"
            )
        )
        workflow.add_conditional_edges(
            name, tools_condition, {"tools": "execute_tools", "end": next_node}
        )

    workflow.add_conditional_edges("execute_tools", route_after_tools)
    workflow.add_edge("CIO", "StrategyRefiner")
    workflow.add_edge("RiskAnalyst", "ComplianceOfficer")
    workflow.add_conditional_edges("ComplianceOfficer", route_after_review)
    workflow.add_edge("PortfolioManager", END)

    return workflow


trading_graph_builder = build_graph()
trading_graph = trading_graph_builder.compile()
print("LangGraph StateGraph compiled successfully.")
