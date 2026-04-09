import json
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from agents.state import TradingStrategy
from agents.llms import review_board_llm


# The scorecard for our LLM judge
class EvaluationOutput(BaseModel):
    risk_management: float = Field(
        description="Score 0-1 for quality of stop-losses and position sizing."
    )
    conviction: float = Field(
        description="Score 0-1 for strength and specificity of the thesis."
    )
    timing: float = Field(
        description="Score 0-1 for precision of entry/exit conditions."
    )
    groundedness: float = Field(
        description="Score 0-1 for how well the strategy is supported by evidence."
    )
    risk_reward: float = Field(
        description="Score 0-1 for the attractiveness of the risk/reward ratio."
    )
    clarity: float = Field(
        description="Score 0-1 for being specific, measurable, and actionable."
    )


def strategy_evaluator(strategy: TradingStrategy, context: str) -> dict:
    """LLM-as-a-Judge to score a strategy against multiple criteria."""
    evaluator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert panel of portfolio managers. Evaluate the "
                "following trading strategy on a scale of 0.0 to 1.0 for each criterion.",
            ),
            (
                "human",
                f"Market Context:\n\n{context}\n\n---\n\n"
                f"Strategy:\n\n{json.dumps(strategy, indent=2)}",
            ),
        ]
    )
    evaluator = review_board_llm.with_structured_output(EvaluationOutput)

    try:
        evaluation = evaluator.invoke(evaluator_prompt.format_messages())
        return evaluation.dict()
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            k: 0.1
            for k in [
                "risk_management",
                "conviction",
                "timing",
                "groundedness",
                "risk_reward",
                "clarity",
            ]
        }


def get_weighted_reward(scores: dict) -> float:
    """Combines scores into a single weighted reward."""
    weights = {
        "risk_management": 0.25,
        "conviction": 0.15,
        "timing": 0.15,
        "groundedness": 0.20,
        "risk_reward": 0.20,
        "clarity": 0.05,
    }
    return sum(scores.get(k, 0) * w for k, w in weights.items())
