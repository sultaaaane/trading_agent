import pandas as pd
import random
from typing import List, TypedDict


# A TypedDict provides a clean, structured way to represent each trading task.
class TradingTask(TypedDict):
    id: str  # A unique identifier for the task
    goal: str  # The investment question our agent must investigate
    context: str  # Market data, fundamentals, and news providing evidence
    expected_decision: str  # Ground truth ('buy', 'sell', or 'hold')


def load_and_prepare_dataset() -> tuple[List[TradingTask], List[TradingTask]]:
    """
    Constructs and splits a financial analysis dataset into training and validation sets.
    """
    print("Preparing financial analysis dataset...")

    # TODO Get data from financial data provider.
    
    
    # For the demo
    tickers = [
        "AAPL",
        "NVDA",
        "TSLA",
        "MSFT",
        "AMZN",
        "META",
        "GOOG",
        "JPM",
        "V",
        "UNH",
    ]
    scenarios = [
        {
            "context": "Revenue up 12% YoY, P/E ratio at 28x, strong iPhone demand in Asia...",
            "decision": "buy",
        },
        {
            "context": "Data center GPU demand exceeding supply, backlog growing 3x...",
            "decision": "buy",
        },
        {
            "context": "Regulatory headwinds in EU, slowing ad revenue growth, P/E at 35x...",
            "decision": "hold",
        },
    ]

    trading_tasks = []
    for i, scenario in enumerate(scenarios):
        ticker = random.choice(tickers)
        task = TradingTask(
            id=f"trade_{i:04d}",
            goal=f"Should we {scenario['decision']} {ticker} given current market conditions?",
            context=scenario["context"],
            expected_decision=scenario["decision"],
        )
        trading_tasks.append(task)

    # We perform a standard 80/20 split for training and validation sets.
    train_size = int(0.8 * len(trading_tasks))
    train_set = trading_tasks[:train_size]
    val_set = trading_tasks[train_size:]

    print(f"Dataset prepared. Total samples: {len(trading_tasks)}")
    print(f"Train: {len(train_set)} | Validation: {len(val_set)}")
    return train_set, val_set


train_dataset, val_dataset = load_and_prepare_dataset()
