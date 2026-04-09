import pytest
from agents.state import AgentState
from agents.nodes import create_agent_node
from unittest.mock import MagicMock


def test_agent_state_initialization():
    state: AgentState = {
        "messages": [],
        "sender": "",
        "turn_count": 0,
        "initial_theses": [],
        "risk_review": None,
        "compliance_review": None,
        "final_strategy": None,
        "trading_goal": "Buy AAPL",
    }
    assert state["turn_count"] == 0
    assert len(state["messages"]) == 0


def test_create_agent_node_execution():
    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Test Response"
    mock_runner.invoke.return_value = mock_result

    node_fn = create_agent_node("TestAgent", mock_runner)

    state: AgentState = {
        "messages": [],
        "sender": "",
        "turn_count": 0,
        "initial_theses": [],
        "risk_review": None,
        "compliance_review": None,
        "final_strategy": None,
        "trading_goal": "Test Goal",
    }

    output = node_fn(state)

    assert state["turn_count"] == 1
    assert output["sender"] == "TestAgent"
    assert output["messages"][0].content == "Test Response"
