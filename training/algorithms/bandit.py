from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
from agentlightning.algorithm import Algorithm
from training.adapter import HierarchicalTraceAdapter


class ContextualBanditRL(Algorithm):
    def __init__(self):
        super().__init__()
        self.policy = SGDClassifier(loss="log_loss", warm_start=True)
        self.vectorizer = HashingVectorizer(n_features=2**12)
        self.is_fitted = False
        self.adapter = HierarchicalTraceAdapter()

    async def run(self, train_dataset, val_dataset):
        print("--- Starting Contextual Bandit Training for CIO ---")
        store = self.get_store()
        completed = await store.query_rollouts(status=["succeeded"])

        if not completed:
            print("No completed rollouts. Skipping bandit training.")
            return

        training_samples = []
        for rollout in completed:
            spans = await store.query_spans(rollout.rollout_id)
            bandit_data = self.adapter.adapt_for_bandit(spans)
            training_samples.extend(bandit_data)

        if not training_samples:
            return

        print(f"Training bandit on {len(training_samples)} samples...")
        for contexts, chosen_action, final_reward in training_samples:
            X = self.vectorizer.fit_transform(contexts)
            y = np.zeros(len(contexts))
            y[chosen_action] = 1

            # Weight the chosen action by the reward
            sample_weight = np.full(
                len(contexts), (1 - final_reward) / max(len(contexts) - 1, 1)
            )
            sample_weight[chosen_action] = final_reward

            if self.is_fitted:
                self.policy.partial_fit(X, y, sample_weight=sample_weight)
            else:
                self.policy.fit(
                    X, y, sample_weight=sample_weight, classes=np.array([0, 1])
                )
                self.is_fitted = True

        print("CIO selection policy updated.")


bandit_algorithm = ContextualBanditRL()
