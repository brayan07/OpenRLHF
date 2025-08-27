import types
import pytest

from openrlhf.utils.utils import get_tokenizer


class StubModelConfig:
    def __init__(self):
        self.pad_token_id = None


class StubModel:
    def __init__(self):
        self.config = StubModelConfig()


class StubTokenizer:
    def __init__(self, eos_token="</s>", eos_token_id=2, pad_token=None, pad_token_id=None):
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.pad_token = pad_token
        self.pad_token_id = pad_token_id
        self.padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        # Define simple encoding behavior for tests
        if text == "<pad>":
            return [42]
        if text == "MULTI":
            return [10, 11]
        # default single id
        return [99]


@pytest.fixture()
def strategy_with_args():
    # minimal strategy stub holding .args
    class Strategy:
        def __init__(self, args):
            self.args = args

    return Strategy


@pytest.fixture()
def patch_auto_tokenizer(monkeypatch):
    # patch transformers.AutoTokenizer.from_pretrained to return our stub
    import transformers

    def _factory(*args, **kwargs):
        return StubTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(_factory))


def test_custom_pad_token_single_id(monkeypatch, strategy_with_args, patch_auto_tokenizer):
    # Arrange
    args = types.SimpleNamespace(pad_token_string="<pad>")
    strategy = strategy_with_args(args)
    model = StubModel()

    # Act
    tok = get_tokenizer("any-pretrain", model, padding_side="left", strategy=strategy, use_fast=True)

    # Assert
    assert tok.pad_token == "<pad>"
    assert tok.pad_token_id == 42
    assert model.config.pad_token_id == 42


def test_custom_pad_token_raises_if_multi_ids(monkeypatch, strategy_with_args, patch_auto_tokenizer):
    # Arrange
    args = types.SimpleNamespace(pad_token_string="MULTI")
    strategy = strategy_with_args(args)
    model = StubModel()

    # Act / Assert
    with pytest.raises(ValueError) as ei:
        _ = get_tokenizer("any-pretrain", model, strategy=strategy)
    assert "must map to exactly one token id" in str(ei.value)


def test_fallback_to_eos_when_no_custom_and_no_pad(monkeypatch, patch_auto_tokenizer):
    # Arrange: tokenizer initially has no pad set; our stub sets eos
    model = StubModel()

    # Act
    tok = get_tokenizer("any-pretrain", model, strategy=None)

    # Assert
    assert tok.pad_token == tok.eos_token
    assert tok.pad_token_id == tok.eos_token_id
    assert model.config.pad_token_id == tok.eos_token_id
