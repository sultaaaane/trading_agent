from agentlightning.algorithm import Algorithm
from datasets import Dataset as HuggingFaceDataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import multiprocessing
import time
import agentlightning as agl
from training.adapter import HierarchicalTraceAdapter


def unsloth_sft_trainer(dataset, base_model, output_dir):
    """SFT training function that runs in a separate process."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="messages",
        max_seq_length=4096,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


class SFTOnSuccess(Algorithm):
    def __init__(self, reward_threshold=0.8, base_model="Qwen/Qwen2-1.5B-Instruct"):
        super().__init__()
        self.reward_threshold = reward_threshold
        self.base_model = base_model
        self.adapter = HierarchicalTraceAdapter()

    async def run(self, train_dataset, val_dataset):
        print("--- Starting SFT Training for Junior Analysts ---")
        store = self.get_store()
        all_rollouts = await store.query_rollouts(status=["succeeded"])

        high_reward_traces = []
        for rollout in all_rollouts:
            spans = await store.query_spans(rollout.rollout_id)
            final_reward = agl.find_final_reward(spans)
            if final_reward and final_reward >= self.reward_threshold:
                high_reward_traces.append(spans)

        print(f"Found {len(high_reward_traces)} high-reward traces.")
        if high_reward_traces:
            sft_data = self.adapter.adapt_for_sft(sum(high_reward_traces, []))
            sft_dataset = HuggingFaceDataset.from_list(
                [{"messages": m["messages"]} for m in sft_data]
            )
            output_dir = f"./models/junior_analyst_sft_v{int(time.time())}"

            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(
                target=lambda: q.put(
                    unsloth_sft_trainer(sft_dataset, self.base_model, output_dir)
                )
            )
            p.start()
            p.join()

            # Update the LLMProxy to use the new model
            llm_proxy = self.get_llm_proxy()
            if llm_proxy:
                # Serve the new model and update routing
                await llm_proxy.replace_model(
                    self.base_model,
                    f"openai/{output_dir}",
                    api_base="http://localhost:8002/v1",
                )
                print("LLMProxy updated with fine-tuned junior analyst model.")


sft_algorithm = SFTOnSuccess()
