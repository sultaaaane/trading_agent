import os
import wandb
import agentlightning as agl


class WandbLoggingHook(agl.Hook):
    def __init__(self, project_name: str):
        self.run_initialized = False
        if os.environ.get("WANDB_API_KEY"):
            try:
                wandb.init(project=project_name, resume="allow")
                self.run_initialized = True
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")

    async def on_trace_end(self, *, rollout: agl.Rollout, tracer: agl.Tracer, **kwargs):
        _ = kwargs  # unused
        if not self.run_initialized:
            return
        final_reward = agl.find_final_reward(tracer.get_last_trace())
        if final_reward is not None:
            wandb.log({"live_reward": final_reward, "rollout_id": rollout.rollout_id})


custom_hook = WandbLoggingHook(project_name="Trading-Agent-Training")
