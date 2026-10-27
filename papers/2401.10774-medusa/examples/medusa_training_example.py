"""Teaching example for Medusa-1/2 training objectives (not official Medusa code).

This script is self-contained with PyTorch and tests label alignment + gradient flow.
Replace DummyLM with a causal Hugging Face model for real training, providing
output_hidden_states=True, logits and an input attention_mask.
Do not mistake these toy loss values for research training metrics.
"""
import torch
from torch import nn
from torch.nn import functional as F


class MedusaHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, lm_head: nn.Linear):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, vocab_size, bias=False)
        nn.init.zeros_(self.fc.weight)
        with torch.no_grad():
            self.out.weight.copy_(lm_head.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.fc(hidden)) + hidden)


def medusa_loss(hidden, heads, labels, decay=0.8):
    """Each k-th extra head predicts position t+k+1, not t+1."""
    total = hidden.new_zeros(())
    seq_len = hidden.size(1)
    for k, head in enumerate(heads, start=1):
        shift = k + 1
        if shift >= seq_len:
            continue
        target = labels[:, shift:].contiguous()
        valid = (target != -100).sum()
        if not valid.item():
            continue
        logits = head(hidden[:, :-shift, :])
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1),
            ignore_index=-100, reduction="sum"
        ) / valid
        total = total + (decay ** k) * ce
    return total


def training_loss(backbone, heads, input_ids, labels,
                  mode, lambda0=0.2, attention_mask=None):
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                  use_cache=False, output_hidden_states=True)
    if mode == "medusa1":
        with torch.no_grad():
            outputs = backbone(**kwargs)
        return medusa_loss(outputs.hidden_states[-1].detach(), heads, labels)
    if mode != "medusa2":
        raise ValueError("mode must be medusa1 or medusa2")
    outputs = backbone(**kwargs)
    logits = outputs.logits[:, :-1, :].contiguous()
    target = labels[:, 1:].contiguous()
    lm_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        ignore_index=-100
    )
    return lm_ce + lambda0 * medusa_loss(outputs.hidden_states[-1], heads, labels)


class DummyLM(nn.Module):
    """Toy backbone for gradient checks, not a causal Transformer."""
    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None,
                use_cache=False, output_hidden_states=True):
        h = self.emb(input_ids)
        return type("Out", (), {"logits": self.proj(h), "hidden_states": [h]})()

    def get_output_embeddings(self):
        return self.proj


if __name__ == "__main__":
    torch.manual_seed(1)
    model = DummyLM()
    heads = nn.ModuleList(
        [MedusaHead(16, 32, model.get_output_embeddings()) for _ in range(5)]
    )
    tokens = torch.randint(0, 32, (2, 12))
    labels = tokens.clone()
    labels[:, :3] = -100  # only selected targets participate

    # Stage 1: freeze the entire backbone, train only Medusa Heads.
    model.requires_grad_(False)
    loss1 = training_loss(model, heads, tokens, labels, "medusa1")
    loss1.backward()
    print("medusa1", loss1.detach().item(),
          "head grad", heads[0].fc.weight.grad is not None,
          "backbone grad", any(p.grad is not None for p in model.parameters()))

    # Stage 2: unfreeze selected backbone parameters (LoRA in a real run).
    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    loss2 = training_loss(model, heads, tokens, labels, "medusa2", lambda0=0.2)
    loss2.backward()
    print("medusa2", loss2.detach().item(),
          "backbone grad", any(p.grad is not None for p in model.parameters()))
