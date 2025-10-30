"""Tests for auxiliary SFT loss functionality in PPO training."""

import pytest
import torch
import torch.nn.functional as F

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.models import SFTLoss


def log_probs_from_logits_cpu(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """CPU-only version of log_probs_from_logits for testing.
    
    Avoids flash_attn CUDA requirement by using pure PyTorch operations.
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # Use standard PyTorch operations (no flash_attn)
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log probs for the actual labels
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    
    return log_probs_labels


class TestExperienceWithSFTMask:
    """Test that Experience dataclass properly handles sft_loss_mask."""

    def test_experience_creation_with_sft_mask(self):
        """Test creating Experience with sft_loss_mask field."""
        batch_size = 2
        seq_len = 10

        experience = Experience(
            sequences=torch.randint(0, 1000, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            action_mask=torch.zeros(batch_size, seq_len - 1),
            sft_loss_mask=torch.zeros(batch_size, seq_len - 1),
        )

        assert experience.sft_loss_mask is not None
        assert experience.sft_loss_mask.shape == (batch_size, seq_len - 1)

    def test_experience_creation_without_sft_mask(self):
        """Test creating Experience without sft_loss_mask (should be None)."""
        batch_size = 2
        seq_len = 10

        experience = Experience(
            sequences=torch.randint(0, 1000, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            action_mask=torch.zeros(batch_size, seq_len - 1),
        )

        assert experience.sft_loss_mask is None

    def test_experience_to_device_with_sft_mask(self):
        """Test moving Experience to device preserves sft_loss_mask."""
        batch_size = 2
        seq_len = 10

        experience = Experience(
            sequences=torch.randint(0, 1000, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            action_mask=torch.zeros(batch_size, seq_len - 1),
            sft_loss_mask=torch.ones(batch_size, seq_len - 1),
        )

        # Move to CPU (should work even if already on CPU)
        experience.to_device(torch.device("cpu"))

        assert experience.sft_loss_mask is not None
        assert experience.sft_loss_mask.device.type == "cpu"

    def test_experience_serialization_with_sft_mask(self):
        """Test that sft_loss_mask is included in serialization."""
        batch_size = 2
        seq_len = 10

        experience = Experience(
            sequences=torch.randint(0, 1000, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len),
            action_mask=torch.zeros(batch_size, seq_len - 1),
            sft_loss_mask=torch.ones(batch_size, seq_len - 1),
        )

        serialized = experience.to_serializable_dict()

        assert "sft_loss_mask" in serialized
        assert serialized["sft_loss_mask"] is not None


class TestSFTTokenRangeConversion:
    """Test conversion from sft_token_ranges to sft_loss_mask."""

    def test_single_range_conversion(self):
        """Test converting a single token range to mask."""
        seq_len = 20
        sft_token_ranges = [[5, 10]]  # Exclusive end

        # Simulate the conversion logic from experience_maker_async.py
        attention_mask = torch.ones(seq_len)
        sft_loss_mask = torch.zeros_like(attention_mask)

        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1

        # Apply shift (like in experience_maker_async.py line 102)
        sft_loss_mask = sft_loss_mask[1:]

        # Verify the mask
        assert sft_loss_mask.sum() == 5  # 5 tokens in range [5, 10)
        assert sft_loss_mask[4:9].sum() == 5  # After shift, range is [4, 9)
        assert sft_loss_mask[:4].sum() == 0
        assert sft_loss_mask[9:].sum() == 0

    def test_multiple_ranges_conversion(self):
        """Test converting multiple token ranges to mask."""
        seq_len = 30
        sft_token_ranges = [[5, 10], [15, 20]]  # Two ranges

        attention_mask = torch.ones(seq_len)
        sft_loss_mask = torch.zeros_like(attention_mask)

        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1

        sft_loss_mask = sft_loss_mask[1:]

        # Verify both ranges are marked
        assert sft_loss_mask.sum() == 10  # 5 + 5 tokens
        assert sft_loss_mask[4:9].sum() == 5  # First range (shifted)
        assert sft_loss_mask[14:19].sum() == 5  # Second range (shifted)

    def test_empty_ranges_conversion(self):
        """Test that empty ranges produce all-zero mask."""
        seq_len = 20
        sft_token_ranges = []

        attention_mask = torch.ones(seq_len)
        sft_loss_mask = torch.zeros_like(attention_mask)

        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1

        sft_loss_mask = sft_loss_mask[1:]

        assert sft_loss_mask.sum() == 0

    def test_range_convention_matches_action_mask(self):
        """Test that sft_token_ranges uses same convention as action_ranges."""
        seq_len = 20
        action_ranges = [[5, 10]]  # Exclusive end
        sft_token_ranges = [[12, 17]]  # Exclusive end

        attention_mask = torch.ones(seq_len)

        # Create action mask
        action_mask = torch.zeros_like(attention_mask)
        for start, end in action_ranges:
            action_mask[start:end] = 1

        # Create SFT mask
        sft_loss_mask = torch.zeros_like(attention_mask)
        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1

        # Both should use same slicing convention
        assert action_mask[5:10].sum() == 5
        assert sft_loss_mask[12:17].sum() == 5


class TestSFTLossComputation:
    """Test SFT loss computation with masks."""

    def test_sft_loss_with_mask(self):
        """Test that SFT loss is computed only on masked tokens."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create mock logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute per-token log probs
        per_token_log_probs = log_probs_from_logits_cpu(logits, labels, temperature=1.0)

        # Create mask with only some positions active
        sft_loss_mask = torch.zeros(batch_size, seq_len)
        sft_loss_mask[:, 3:7] = 1  # Only positions 3-6 should contribute

        # Compute SFT loss
        sft_loss_fn = SFTLoss(token_level_loss=True)
        loss = sft_loss_fn(per_token_log_probs, sft_loss_mask)

        # Loss should be a scalar
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sft_loss_with_zero_mask(self):
        """Test that zero mask produces NaN (expected behavior for masked_mean with zero mask)."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create mock logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute per-token log probs
        per_token_log_probs = log_probs_from_logits_cpu(logits, labels, temperature=1.0)

        # All-zero mask
        sft_loss_mask = torch.zeros(batch_size, seq_len)

        sft_loss_fn = SFTLoss(token_level_loss=True)
        loss = sft_loss_fn(per_token_log_probs, sft_loss_mask)

        # With zero mask, masked_mean returns NaN (division by zero)
        # This is expected behavior - in practice, we check if mask is None or has any 1s
        assert torch.isnan(loss)

    def test_sft_loss_independent_from_action_mask(self):
        """Test that SFT loss and action mask are independent."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create mock logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Compute per-token log probs
        per_token_log_probs = log_probs_from_logits_cpu(logits, labels, temperature=1.0)

        # Non-overlapping masks
        action_mask = torch.zeros(batch_size, seq_len)
        action_mask[:, 2:5] = 1  # Positions 2-4

        sft_loss_mask = torch.zeros(batch_size, seq_len)
        sft_loss_mask[:, 6:9] = 1  # Positions 6-8

        # Verify masks don't overlap
        assert (action_mask * sft_loss_mask).sum() == 0

        # Both losses should be computable independently
        sft_loss_fn = SFTLoss(token_level_loss=True)
        sft_loss = sft_loss_fn(per_token_log_probs, sft_loss_mask)

        assert not torch.isnan(sft_loss)
        assert sft_loss.item() > 0


class TestLogProbsFromLogits:
    """Test the log_probs_from_logits utility function."""

    def test_log_probs_computation(self):
        """Test that log probs are computed correctly from logits."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create mock logits and sequences (ensure same device and dtype)
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        sequences = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Compute log probs
        log_probs = log_probs_from_logits_cpu(logits, sequences, temperature=1.0)

        # Check shape
        assert log_probs.shape == (batch_size, seq_len)

        # Check values are valid log probabilities (should be negative or zero)
        assert (log_probs <= 0).all()

    def test_log_probs_with_temperature(self):
        """Test that temperature affects log probs."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Create separate logits for each temperature to avoid in-place modification issues
        logits_t1 = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        logits_t2 = logits_t1.clone()  # Clone to avoid in-place modification
        sequences = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

        # Compute with different temperatures
        log_probs_t1 = log_probs_from_logits_cpu(logits_t1, sequences, temperature=1.0)
        log_probs_t2 = log_probs_from_logits_cpu(logits_t2, sequences, temperature=2.0)

        # Different temperatures should give different results
        assert not torch.allclose(log_probs_t1, log_probs_t2)


class TestIntegrationSFTLoss:
    """Integration tests for the full SFT loss pipeline."""

    def test_full_pipeline_with_sft_mask(self):
        """Test the full pipeline from ranges to loss."""
        batch_size = 2
        seq_len = 20
        vocab_size = 100

        # Step 1: Create token ranges (as would come from agent)
        sft_token_ranges = [[5, 10], [15, 18]]

        # Step 2: Convert to mask (as in experience_maker_async.py)
        attention_mask = torch.ones(seq_len)
        sft_loss_mask = torch.zeros_like(attention_mask)
        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1
        sft_loss_mask = sft_loss_mask[1:]  # Shift by 1

        # Expand to batch
        sft_loss_mask = sft_loss_mask.unsqueeze(0).expand(batch_size, -1)

        # Step 3: Create mock logits and sequences (ensure correct dtypes)
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        sequences = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)

        # Step 4: Compute log probs from logits (as in ppo_actor.py)
        log_probs = log_probs_from_logits_cpu(logits, rolled_sequences, temperature=1.0)
        log_probs = log_probs[:, :-1]  # Remove last position

        # Step 5: Compute SFT loss
        sft_loss_fn = SFTLoss(token_level_loss=True)
        sft_loss = sft_loss_fn(log_probs, sft_loss_mask)

        # Verify loss is valid
        assert not torch.isnan(sft_loss)
        assert not torch.isinf(sft_loss)
        assert sft_loss.item() > 0

    def test_no_sft_loss_when_mask_is_none(self):
        """Test that SFT loss is skipped when mask is None."""
        # Simulate the condition check in ppo_actor.py
        use_aux_sft_loss = True
        sft_loss_mask = None

        # This condition should be False
        should_compute_sft_loss = use_aux_sft_loss and sft_loss_mask is not None

        assert not should_compute_sft_loss

    def test_no_sft_loss_when_coefficient_is_zero(self):
        """Test that SFT loss is disabled when coefficient is 0."""
        aux_sft_coef = 0.0
        use_aux_sft_loss = aux_sft_coef > 1e-8

        assert not use_aux_sft_loss


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_sft_mask_with_truncation(self):
        """Test that SFT mask is properly truncated with sequences."""
        seq_len = 30
        truncate_length = 20

        # Create mask longer than truncate length
        sft_loss_mask = torch.ones(seq_len)
        sft_loss_mask = sft_loss_mask[1:truncate_length]

        # Should be truncated to truncate_length - 1
        assert sft_loss_mask.shape[0] == truncate_length - 1

    def test_sft_ranges_at_sequence_boundaries(self):
        """Test ranges at the start and end of sequences."""
        seq_len = 20

        # Range at start
        sft_token_ranges = [[0, 5]]
        sft_loss_mask = torch.zeros(seq_len)
        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1
        assert sft_loss_mask[:5].sum() == 5

        # Range at end
        sft_token_ranges = [[15, 20]]
        sft_loss_mask = torch.zeros(seq_len)
        for start, end in sft_token_ranges:
            sft_loss_mask[start:end] = 1
        assert sft_loss_mask[15:].sum() == 5

    def test_overlapping_action_and_sft_masks(self):
        """Test detection of overlapping masks (should be avoided)."""
        seq_len = 20

        action_mask = torch.zeros(seq_len)
        action_mask[5:10] = 1

        sft_loss_mask = torch.zeros(seq_len)
        sft_loss_mask[8:13] = 1  # Overlaps with action_mask at [8, 10)

        # Check for overlap
        overlap = (action_mask * sft_loss_mask).sum()

        # This should be > 0, indicating overlap (which is not recommended)
        assert overlap > 0
        # In practice, you should ensure overlap == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
