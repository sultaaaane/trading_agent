import json
import agentlightning as agl
from typing import List
from agentlightning.adapter import TraceToMessages


class HierarchicalTraceAdapter(agl.TracerTraceToTriplet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_adapter = TraceToMessages()
        self.agent_match = ""

    def adapt_for_sft(self, source: List[agl.Span]) -> List[dict]:
        """Extracts junior analyst conversations for SFT."""
        junior_names = ["TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"]
        junior_spans = [s for s in source if s.attributes.get("name") in junior_names]
        if not junior_spans:
            return []
        return self.message_adapter.adapt(junior_spans)

    def adapt_for_ppo(self, source: List[agl.Span]) -> List[agl.Triplet]:
        """Extracts senior strategist data for PPO training."""
        senior_names = ["StrategyRefiner", "ExecutionPlanDesigner"]
        self.agent_match = "|".join(senior_names)
        return super().adapt(source)

    def adapt_for_bandit(self, source: List[agl.Span]) -> List[tuple]:
        """Extracts CIO decision data for contextual bandit."""
        final_reward = agl.find_final_reward(source)
        if final_reward is None:
            return []

        cio_span = next((s for s in source if s.attributes.get("name") == "CIO"), None)
        if not cio_span:
            return []

        # Reconstruct the available theses (context)
        junior_spans = [
            s
            for s in source
            if s.attributes.get("name")
            in ["TechnicalAnalyst", "FundamentalAnalyst", "SentimentAnalyst"]
        ]
        contexts = []
        for span in sorted(junior_spans, key=lambda s: s.start_time):
            try:
                output = span.attributes.get("output.messages")
                if output and isinstance(output, list):
                    content = json.loads(output[-1].get("content", "{}"))
                    contexts.append(content.get("thesis", ""))
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        if not contexts:
            return []

        # Extract the CIO's chosen action
        try:
            output = cio_span.attributes.get("output.messages")
            if output and isinstance(output, list):
                cio_output = json.loads(output[-1].get("content", "{}"))
                chosen = cio_output.get("selected_thesis_index")
                if chosen is not None and 0 <= chosen < len(contexts):
                    return [(contexts, chosen, final_reward)]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        return []


custom_adapter = HierarchicalTraceAdapter()
