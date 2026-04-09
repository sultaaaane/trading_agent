from langgraph.graph import END
from langchain_core.messages import HumanMessage
from agents.state import AgentState
import json

MAX_TURNS = 15  # Safety limit to prevent infinite loops


def create_agent_node(agent_name: str, agent_runner):
    """Creates a LangGraph node function for a given agent runner."""

    def agent_node(state: AgentState) -> dict:
        print(f"--- Node: {agent_name} (Turn {state['turn_count']}) ---")
        state["turn_count"] += 1
        result = agent_runner.invoke(state)

        # Parse structured JSON from review agents
        if agent_name in ["RiskAnalyst", "ComplianceOfficer"]:
            try:
                content = json.loads(result.content)
                if agent_name == "RiskAnalyst":
                    state["risk_review"] = content
                else:
                    state["compliance_review"] = content
            except (json.JSONDecodeError, TypeError):
                print(f"Error parsing JSON from {agent_name}")

        # Update messages and set the sender for ReAct routing
        return {"messages": [result], "sender": agent_name}

    return agent_node


def tools_condition(state: AgentState) -> str:
    """Checks for tool calls and the turn count."""
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    if state["turn_count"] >= MAX_TURNS:
        return "end"
    return "tools"


def route_after_tools(state: AgentState) -> str:
    """Routes back to the agent that initiated the tool call."""
    sender = state.get("sender")
    if not sender:
        return END
    return sender


def route_after_review(state: AgentState):
    """Intelligent router based on review severity."""
    risk = state.get("risk_review", {})
    compliance = state.get("compliance_review", {})

    risk_sev = risk.get("critique_severity", "MINOR")
    compliance_sev = compliance.get("critique_severity", "MINOR")

    if state["turn_count"] >= MAX_TURNS:
        return "PortfolioManager"

    # CRITICAL: the core thesis is flawed, go back to hypothesis refinement
    if risk_sev == "CRITICAL" or compliance_sev == "CRITICAL":
        state["messages"].append(
            HumanMessage(content="Critical feedback received. Rethink the core thesis.")
        )
        return "StrategyRefiner"

    # MAJOR: the execution plan needs significant work
    if risk_sev == "MAJOR" or compliance_sev == "MAJOR":
        state["messages"].append(
            HumanMessage(content="Major feedback received. Revise the execution plan.")
        )
        return "ExecutionPlanDesigner"

    # MINOR or APPROVE: proceed to final decision
    return "PortfolioManager"
