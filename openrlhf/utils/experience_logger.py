import os
import json
import torch
import ray
from typing import List, Dict, Any, Iterable

# Lazily import to avoid heavy deps at module import time
from openrlhf.trainer.ppo_utils.experience_maker import Experience


@ray.remote
class ExperienceDiskLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _save_pt(self, kind: str, step: int, records, mode="train"):
        """Save a list of per-item dicts with tensors/lists using torch.save.
        No additional batching; preserves original grouping.
        """
        fn = os.path.join(self.out_dir, f"{kind}_{mode}_step{step:06d}.pt")
        torch.save(records, fn)

    def log_rollouts(self, step: int, records, mode="train"):
        self._save_pt("rollouts", step, records, mode=mode)

    def log_experiences(self, step: int, records, mode="train"):
        self._save_pt("experiences", step, records, mode=mode)


# -------- Inspection utilities --------
def _dict_to_experience(d: Dict[str, Any]) -> Experience:
    """Convert a serialized dict record back into an Experience object.

    This assumes the dict was produced by Experience.to_serializable_dict().
    """
    return Experience(
        index=d.get("index"),
        sequences=d.get("sequences"),
        attention_mask=d.get("attention_mask"),
        action_mask=d.get("action_mask"),
        action_log_probs=d.get("action_log_probs"),
        base_action_log_probs=d.get("base_action_log_probs"),
        rollout_log_probs=d.get("rollout_log_probs"),
        values=d.get("values"),
        returns=d.get("returns"),
        advantages=d.get("advantages"),
        kl=d.get("kl"),
        prompts=d.get("prompts"),
        labels=d.get("labels"),
        rewards=d.get("rewards"),
        scores=d.get("scores"),
        sft_loss_mask=d.get("sft_loss_mask"),
        info=d.get("info"),
    )


def load_experience_file(path: str) -> List[Experience]:
    """Load a saved .pt dump and return a list of Experience objects.

    The file is expected to contain a list of dicts created by to_serializable_dict().
    """
    records = torch.load(path, map_location="cpu")
    if isinstance(records, list):
        if len(records) == 0:
            return []
        # If already Experience objects (future-proof), return as-is
        if isinstance(records[0], Experience):
            return records
        # Otherwise convert dicts to Experience
        return [_dict_to_experience(r) for r in records]
    raise ValueError(f"Unsupported file format in {path}: expected list, got {type(records)}")


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Iterate JSONL summary records (one dict per line)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
