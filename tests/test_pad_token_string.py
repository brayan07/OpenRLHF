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
        # Minimal id<->token map for tests
        self._id_to_token = {
            42: "<pad>",
            10: "ID10",
            11: "ID11",
            99: "ID99",
        }

    def encode(self, text, add_special_tokens=False):
        # Define simple encoding behavior for tests
        if text == "<pad>":
            return [42]
        if text == "MULTI":
            return [10, 11]
        # default single id
        return [99]

    def convert_ids_to_tokens(self, idx):
        return self._id_to_token.get(idx, None)


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


def test_custom_pad_token_id_valid(monkeypatch, strategy_with_args, patch_auto_tokenizer):
    # Arrange: Provide only pad_token_id that maps to a valid string token
    args = types.SimpleNamespace(pad_token_id=42)
    strategy = strategy_with_args(args)
    model = StubModel()

    # Act
    tok = get_tokenizer("any-pretrain", model, strategy=strategy)

    # Assert
    assert tok.pad_token_id == 42
    assert tok.pad_token == "<pad>"
    assert model.config.pad_token_id == 42


def test_both_args_consistent(monkeypatch, strategy_with_args, patch_auto_tokenizer):
    # Arrange: Both string and id provided and consistent
    args = types.SimpleNamespace(pad_token_string="<pad>", pad_token_id=42)
    strategy = strategy_with_args(args)
    model = StubModel()

    # Act
    tok = get_tokenizer("any-pretrain", model, strategy=strategy)

    # Assert
    assert tok.pad_token == "<pad>"
    assert tok.pad_token_id == 42
    assert model.config.pad_token_id == 42


def test_both_args_inconsistent_raises(monkeypatch, strategy_with_args, patch_auto_tokenizer):
    # Arrange: String maps to 42, but id is 99 -> mismatch
    args = types.SimpleNamespace(pad_token_string="<pad>", pad_token_id=99)
    strategy = strategy_with_args(args)
    model = StubModel()

    # Act / Assert
    with pytest.raises(ValueError) as ei:
        _ = get_tokenizer("any-pretrain", model, strategy=strategy)
    assert "are inconsistent" in str(ei.value)


def test_pad_token_id_invalid_raises(monkeypatch, strategy_with_args, patch_auto_tokenizer):
    # Arrange: id 123 has no string mapping in stub convert_ids_to_tokens
    args = types.SimpleNamespace(pad_token_id=123)
    strategy = strategy_with_args(args)
    model = StubModel()

    # Act / Assert
    with pytest.raises(ValueError) as ei:
        _ = get_tokenizer("any-pretrain", model, strategy=strategy)
    assert "does not map to a valid string token" in str(ei.value)
