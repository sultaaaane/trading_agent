import pytest
from graph.workflow import build_graph
from langgraph.graph.state import CompiledStateGraph


def test_graph_build():
    """Test that the graph can be constructed and compiled without errors."""
    workflow = build_graph()
    assert isinstance(workflow.compile(), CompiledStateGraph)


def test_graph_nodes():
    """Test that all expected nodes are present in the graph."""
    workflow = build_graph()
    nodes = workflow.nodes.keys()
    expected_nodes = [
        "TechnicalAnalyst",
        "FundamentalAnalyst",
        "SentimentAnalyst",
        "CIO",
        "StrategyRefiner",
        "ExecutionPlanDesigner",
        "RiskAnalyst",
        "ComplianceOfficer",
        "PortfolioManager",
        "execute_tools",
    ]
    for node in expected_nodes:
        assert node in nodes
