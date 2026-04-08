from typing import List, TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage


# Output structure for each Junior Analyst's thesis
class AnalystThesis(TypedDict):
    thesis: str  # The core investment thesis
    supporting_evidence: List[str]  # Evidence gathered via tools
    agent_name: str  # Which analyst proposed it


# Output structure for the Senior Strategist's trading plan
class TradingStrategy(TypedDict):
    title: str
    entry_conditions: List[str]  # Specific conditions to enter the trade
    exit_conditions: List[str]  # Take-profit and stop-loss levels
    position_sizing: str  # How much capital to allocate
    risk_reward_ratio: float  # Expected risk/reward


# Structured feedback from reviewers
class ReviewDecision(TypedDict):
    decision: Literal["APPROVE", "REVISE"]
    critique_severity: Literal["CRITICAL", "MAJOR", "MINOR"]
    feedback: str


# The master state that flows through all nodes in our LangGraph
class AgentState(TypedDict):
    # Conversation history. The reducer tells LangGraph to append new messages.
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    trading_goal: str  # The initial investment question
    sender: str  # Tracks which agent last acted (crucial for ReAct loops)
    turn_count: int  # Safety counter to prevent infinite loops

    # Junior Analysts' output (accumulates from parallel runs)
    initial_theses: List[AnalystThesis]

    # CIO's selection
    selected_thesis: AnalystThesis
    cio_justification: str

    # Senior Strategist's output
    refined_thesis: str
    trading_strategy: TradingStrategy

    # Review Board's output
    risk_review: ReviewDecision
    compliance_review: ReviewDecision

    # Portfolio Manager's final decision
    final_strategy: TradingStrategy
    final_decision: Literal["GO", "NO-GO"]
    final_rationale: str

    # The final evaluation score from our reward function
    final_evaluation: dict
