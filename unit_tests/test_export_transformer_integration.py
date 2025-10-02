from __future__ import annotations

import math
from pathlib import Path

import onnx
import pytest
import torch
import torch.nn as nn

import run


class MiniEncoder(nn.Module):
    def __init__(self, vocab_size: int = 32, embed_dim: int = 32, seq_len: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=64,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.register_buffer(
            "positional_encoding",
            torch.arange(seq_len, dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        embeddings = self.embed(input_ids) * math.sqrt(self.embed.embedding_dim)
        encoded = self.encoder(embeddings, src_key_padding_mask=~attention_mask.bool())
        return encoded


def _export_mini_encoder_to_onnx(path: Path) -> None:
    module = MiniEncoder()
    module.eval()

    input_ids = torch.randint(0, module.embed.num_embeddings, (1, 8), dtype=torch.int64)
    attention_mask = torch.ones((1, 8), dtype=torch.bool)

    torch.onnx.export(
        module,
        (input_ids, attention_mask),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["encoder_out"],
        opset_version=17,
        do_constant_folding=True,
    )


@pytest.mark.xfail(
    reason="Dynamic slice export constraints under investigation", strict=False
)
def test_transformer_encoder_export_roundtrip(tmp_path):
    onnx_path = tmp_path / "encoder.onnx"
    _export_mini_encoder_to_onnx(onnx_path)

    task = run.ModelTask(source=onnx_path, destination=tmp_path / "encoder.pt2")

    example_cfg = run.ExampleInputConfig(
        default_fill="zeros",
        overrides={
            "input_ids": {
                "shape": [1, 8],
                "warmup_shape": [1, 8],
                "dim_labels": ["batch", "sequence_length"],
            },
            "attention_mask": {
                "shape": [1, 8],
                "warmup_shape": [1, 8],
                "dim_labels": ["batch", "sequence_length"],
                "dtype": "bool",
                "fill": "ones",
            },
        },
        scales={"sequence_length": 8},
        enable_warmup=False,
    )
    config = run.RunnerConfig(inputs=[], example_inputs=example_cfg)
    run_context = run.RunContext(directory=tmp_path, verbosity=0)

    run._export_model(task, config, run_context)

    assert task.destination.exists()

    exported_program = torch.export.load(str(task.destination))
    runtime_args, _ = run._build_example_and_warmup_args(
        onnx.load(str(onnx_path)), config
    )
    output = exported_program.module()(*runtime_args)
    assert output.shape[-1] == 32
