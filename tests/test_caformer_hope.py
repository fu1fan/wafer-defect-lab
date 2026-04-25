import torch

from waferlab.models.caformer_hope import CAFormerHOPEClassifier


def _build_model(token_mode: str) -> CAFormerHOPEClassifier:
    return CAFormerHOPEClassifier(
        token_mode=token_mode,
        pretrained=False,
        token_dim=32,
        num_hope_blocks=1,
        cms_hidden_multiplier=1,
        selfmod_local_conv_window=None,
        dropout=0.0,
        drop_path_rate=0.0,
    )


def test_caformer_hope_token_shapes_teach_and_backward() -> None:
    model = _build_model("spatial")
    model.train()
    x = torch.randn(2, 1, 224, 224)
    tokens = model.forward_tokens(x)
    assert tokens.shape == (2, 49, 32)
    logits = model(x)
    assert logits.shape == (2, 9)
    teach = torch.randn_like(tokens)
    taught = model.forward_with_teach(x, teach_signal=teach, surprise_value=1.0)
    assert taught.shape == (2, 9)
    loss = taught.sum()
    loss.backward()
    assert model.fc.weight.grad is not None


def test_caformer_hope_pooled_shapes_teach_and_backward() -> None:
    model = _build_model("pooled")
    model.train()
    x = torch.randn(2, 1, 224, 224)
    tokens = model.forward_tokens(x)
    assert tokens.shape == (2, 1, 32)
    logits = model(x)
    assert logits.shape == (2, 9)
    teach = torch.randn_like(tokens)
    taught = model.forward_with_teach(x, teach_signal=teach, surprise_value=1.0)
    assert taught.shape == (2, 9)
    loss = taught.sum()
    loss.backward()
    assert model.fc.weight.grad is not None
