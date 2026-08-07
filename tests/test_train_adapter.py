"""Tests for the prefix-tuning input construction.

Uses a tiny stand-in model/tokenizer rather than downloading a real LM, so
the load-bearing tensor logic (concatenation order, attention mask, and the
label masking that keeps the LM loss off the prefix and off padding) is
verified offline and deterministically.
"""

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


class _FakeTokenized(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    """Pads to the longest sequence, mirroring transformers' padding=True."""

    pad_token = "<pad>"

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        import torch

        # One token id per character, truncated, then right-padded with 0.
        sequences = [[ord(c) % 50 + 1 for c in text][:max_length] for text in texts]
        longest = max(len(s) for s in sequences)
        input_ids, attention = [], []
        for sequence in sequences:
            pad_len = longest - len(sequence)
            input_ids.append(sequence + [0] * pad_len)
            attention.append([1] * len(sequence) + [0] * pad_len)
        return _FakeTokenized(
            input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attention)
        )


class _FakeModel:
    def __init__(self, model_dim: int = 6, vocab: int = 64, dtype=None) -> None:
        import torch

        self._embedding = torch.nn.Embedding(vocab, model_dim)
        if dtype is not None:
            self._embedding = self._embedding.to(dtype)

    def get_input_embeddings(self):
        return self._embedding


def _build(prefix_tokens: int = 3, texts=("abcd", "ab"), model_dtype=None):
    import torch

    from gnsm.training.train_adapter import build_prefixed_inputs

    model = _FakeModel(model_dim=6, dtype=model_dtype)
    tokenizer = _FakeTokenizer()
    prefix_embeds = torch.randn(len(texts), prefix_tokens, 6)  # always float32
    return build_prefixed_inputs(
        model, tokenizer, prefix_embeds, texts, max_target_tokens=16, device=torch.device("cpu")
    ), prefix_embeds


def test_float32_prefix_is_cast_to_a_bfloat16_lm() -> None:
    """Regression: a float32 adapter feeding a bf16-loaded LM must not blow up
    in torch.cat / the LM's matmuls (`mat1 and mat2 must have the same dtype`)."""

    import torch

    (inputs_embeds, _mask, _labels), _prefix = _build(model_dtype=torch.bfloat16)
    assert inputs_embeds.dtype == torch.bfloat16


def test_prefix_is_prepended_and_shapes_line_up() -> None:
    (inputs_embeds, attention_mask, labels), prefix_embeds = _build(prefix_tokens=3)
    batch, prefix_len, model_dim = prefix_embeds.shape
    # 2 sequences, longest text is 4 chars -> 3 prefix + 4 tokens.
    assert inputs_embeds.shape == (batch, prefix_len + 4, model_dim)
    assert attention_mask.shape == (batch, prefix_len + 4)
    assert labels.shape == (batch, prefix_len + 4)


def test_prefix_embeddings_are_placed_first_unmodified() -> None:
    import torch

    (inputs_embeds, _mask, _labels), prefix_embeds = _build(prefix_tokens=3)
    assert torch.allclose(inputs_embeds[:, :3, :], prefix_embeds)


def test_labels_mask_the_prefix_positions() -> None:
    (_embeds, _mask, labels), _prefix = _build(prefix_tokens=3)
    assert (labels[:, :3] == -100).all()


def test_labels_mask_padding_but_keep_real_tokens() -> None:
    # texts are "abcd" (4 real) and "ab" (2 real + 2 pad), prefix_tokens=3.
    (_embeds, _mask, labels), _prefix = _build(prefix_tokens=3, texts=("abcd", "ab"))
    row_full, row_padded = labels[0], labels[1]
    assert (row_full[3:] != -100).all()  # no padding in the longest row
    assert (row_padded[3:5] != -100).all()  # its 2 real tokens survive
    assert (row_padded[5:] == -100).all()  # its 2 pad positions are masked


def test_attention_mask_covers_prefix_and_real_tokens_only() -> None:
    (_embeds, attention_mask, _labels), _prefix = _build(prefix_tokens=3, texts=("abcd", "ab"))
    assert (attention_mask[:, :3] == 1).all()  # prefix always attended
    assert attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1, 0, 0]


def test_state_mode_real_is_identity() -> None:
    import torch

    from gnsm.training.train_adapter import apply_state_mode

    state = torch.randn(4, 8)
    assert torch.equal(apply_state_mode(state, "real"), state)


def test_state_mode_zero_removes_all_signal() -> None:
    import torch

    from gnsm.training.train_adapter import apply_state_mode

    state = torch.randn(4, 8)
    assert torch.count_nonzero(apply_state_mode(state, "zero")) == 0


def test_state_mode_shuffled_gives_every_example_a_different_state() -> None:
    """The control must actually break state<->example correspondence, or it
    silently measures the same thing as the treatment."""

    import torch

    from gnsm.training.train_adapter import apply_state_mode

    # Distinct rows so "did this row change?" is unambiguous.
    state = torch.arange(12, dtype=torch.float32).view(4, 3)
    shuffled = apply_state_mode(state, "shuffled")
    assert shuffled.shape == state.shape
    for row in range(state.shape[0]):
        assert not torch.equal(shuffled[row], state[row])
    # Same multiset of states -- only the pairing changed, not the distribution.
    assert torch.equal(shuffled.sum(dim=0), state.sum(dim=0))


def test_state_mode_shuffled_rejects_unshufflable_batch() -> None:
    import torch

    from gnsm.training.train_adapter import apply_state_mode

    with pytest.raises(ValueError, match="batch_size >= 2"):
        apply_state_mode(torch.randn(1, 8), "shuffled")


def test_state_mode_rejects_unknown_mode() -> None:
    import torch

    from gnsm.training.train_adapter import apply_state_mode

    with pytest.raises(ValueError, match="Unknown state_mode"):
        apply_state_mode(torch.randn(2, 8), "nonsense")


def test_every_position_is_either_masked_or_attended_consistently() -> None:
    (_embeds, attention_mask, labels), _prefix = _build(prefix_tokens=2, texts=("xyz", "x"))
    # Any position that isn't attended must also be excluded from the loss.
    unattended = attention_mask == 0
    assert (labels[unattended] == -100).all()
