#!/usr/bin/env python3
"""Capture Qwen3 attention-score-input Q/K vectors into the ADA E4 trace format.

The adapter deliberately keeps Transformers' built-in eager attention backend.
It replaces only the Qwen3 module-level eager fallback with a wrapper that
observes Q/K and immediately delegates to the original implementation. This
preserves the model's ordinary causal-mask construction.

E4 v1 restrictions enforced here:
- Qwen3 causal self-attention;
- no sliding-window attention;
- batch size 1 and no padding;
- prefill-only capture with use_cache=False;
- contiguous visible prefix [0, query_position + 1);
- post-Q/K-normalization, post-RoPE vectors consumed by the score dot product;
- little-endian IEEE-754 f32 tensor storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRACE_MAGIC = b"ADAQK01\x00"
TRACE_VERSION = 1
TENSOR_STAGE = "attention_score_input"

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_LAYERS = (0, 13, 27)
DEFAULT_QUERY_HEADS = (0, 5, 10, 15)
DEFAULT_POSITIONS = (63, 127, 255, 511)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str


@dataclass
class CapturedRecord:
    sample_id: str
    layer_index: int
    query_head_index: int
    kv_head_index: int
    query_position: int
    key_start_position: int
    head_dim: int
    key_count: int
    score_scale: float
    query: Any
    keys: Any


class CaptureSession:
    def __init__(
        self,
        layers: set[int],
        query_heads: tuple[int, ...],
        positions: tuple[int, ...],
    ) -> None:
        self.layers = layers
        self.query_heads = query_heads
        self.positions = positions
        self.current_sample_id: str | None = None
        self.expected_sequence_length: int | None = None
        self.source_dtype: str | None = None
        self.records: list[CapturedRecord] = []

    def begin_sample(self, sample_id: str, sequence_length: int) -> None:
        if self.current_sample_id is not None:
            raise RuntimeError("capture session already has an active sample")
        self.current_sample_id = sample_id
        self.expected_sequence_length = sequence_length

    def end_sample(self) -> None:
        self.current_sample_id = None
        self.expected_sequence_length = None


def parse_int_list(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated integer list"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} entries must be non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} entries must be unique")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    parser.add_argument("--query-heads", default=",".join(map(str, DEFAULT_QUERY_HEADS)))
    parser.add_argument("--positions", default=",".join(map(str, DEFAULT_POSITIONS)))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--capture-id", default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args()

    args.layers = parse_int_list(args.layers, "layers")
    args.query_heads = parse_int_list(args.query_heads, "query-heads")
    args.positions = parse_int_list(args.positions, "positions")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if max(args.positions) >= args.max_tokens:
        parser.error("largest --positions entry must be smaller than --max-tokens")
    if not args.revision or len(args.revision) < 12:
        parser.error("--revision must be an immutable model revision")
    if args.tokenizer_revision is not None and len(args.tokenizer_revision) < 12:
        parser.error("--tokenizer-revision must be immutable")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            sample_id = item.get("sample_id")
            text = item.get("text")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"line {line_number}: sample_id must be non-empty")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            if not isinstance(text, str) or not text:
                raise ValueError(f"line {line_number}: text must be non-empty")
            seen_ids.add(sample_id)
            samples.append(Sample(sample_id=sample_id, text=text))
    if not samples:
        raise ValueError("samples JSONL contains no records")
    return samples


def choose_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    return requested


def validate_model_config(
    config: Any,
    layers: tuple[int, ...],
    query_heads: tuple[int, ...],
) -> None:
    if getattr(config, "model_type", None) != "qwen3":
        raise RuntimeError(
            f"E4 Qwen3 adapter requires model_type=qwen3, got {getattr(config, 'model_type', None)!r}"
        )
    if bool(getattr(config, "use_sliding_window", False)) or getattr(
        config, "sliding_window", None
    ):
        raise RuntimeError("E4 v1 Qwen3 adapter rejects sliding-window attention")

    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    if any(layer >= num_layers for layer in layers):
        raise RuntimeError(f"selected layer outside range 0..{num_layers - 1}")
    if any(head >= num_heads for head in query_heads):
        raise RuntimeError(f"selected query head outside range 0..{num_heads - 1}")
    if num_heads % num_kv_heads != 0:
        raise RuntimeError("Qwen3 Q-head count is not divisible by KV-head count")


def install_qwen3_capture(
    torch: Any,
    qwen3_module: Any,
    session: CaptureSession,
) -> Any:
    original_eager = qwen3_module.eager_attention_forward

    def wrapper(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> Any:
        layer_index = int(module.layer_idx)
        if session.current_sample_id is not None and layer_index in session.layers:
            if query.ndim != 4 or key.ndim != 4:
                raise RuntimeError(
                    f"expected [B,H,T,D] Q/K, got {tuple(query.shape)} and {tuple(key.shape)}"
                )
            if query.shape[0] != 1 or key.shape[0] != 1:
                raise RuntimeError("E4 v1 capture requires batch size 1")
            if query.shape[2] != key.shape[2]:
                raise RuntimeError("E4 prefill requires equal Q and K sequence lengths")
            if session.expected_sequence_length != int(query.shape[2]):
                raise RuntimeError("attention sequence length differs from tokenizer input")

            query_dtype = str(query.dtype).removeprefix("torch.")
            key_dtype = str(key.dtype).removeprefix("torch.")
            if query_dtype != key_dtype:
                raise RuntimeError("Q and K source dtypes differ")
            if session.source_dtype is None:
                session.source_dtype = query_dtype
            elif session.source_dtype != query_dtype:
                raise RuntimeError("capture observed more than one Q/K source dtype")

            q_heads = int(query.shape[1])
            kv_heads = int(key.shape[1])
            if q_heads % kv_heads != 0:
                raise RuntimeError("Q/K head ratio is not integral")
            q_per_kv = q_heads // kv_heads

            scaling = float(kwargs.get("scaling", getattr(module, "scaling", math.nan)))
            if not math.isfinite(scaling) or scaling <= 0.0:
                raise RuntimeError("attention scaling must be finite and positive")

            for q_head in session.query_heads:
                kv_head = q_head // q_per_kv
                if kv_head >= kv_heads:
                    raise RuntimeError("Q-to-KV mapping falls outside captured K heads")
                for position in session.positions:
                    if position >= query.shape[2]:
                        raise RuntimeError(
                            f"sample {session.current_sample_id!r} has only {query.shape[2]} tokens; "
                            f"cannot capture position {position}"
                        )
                    visible_count = position + 1
                    q_cpu = (
                        query[0, q_head, position, :]
                        .detach()
                        .to(torch.float32)
                        .cpu()
                        .contiguous()
                    )
                    k_cpu = (
                        key[0, kv_head, :visible_count, :]
                        .detach()
                        .to(torch.float32)
                        .cpu()
                        .contiguous()
                    )
                    session.records.append(
                        CapturedRecord(
                            sample_id=session.current_sample_id,
                            layer_index=layer_index,
                            query_head_index=q_head,
                            kv_head_index=kv_head,
                            query_position=position,
                            key_start_position=0,
                            head_dim=int(query.shape[3]),
                            key_count=visible_count,
                            score_scale=scaling,
                            query=q_cpu,
                            keys=k_cpu,
                        )
                    )

        return original_eager(module, query, key, value, attention_mask, **kwargs)

    qwen3_module.eager_attention_forward = wrapper
    return original_eager


def push_u32(buffer: bytearray, value: int) -> None:
    if not 0 <= value <= 0xFFFF_FFFF:
        raise ValueError(f"u32 out of range: {value}")
    buffer += struct.pack("<I", value)


def push_u64(buffer: bytearray, value: int) -> None:
    if not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"u64 out of range: {value}")
    buffer += struct.pack("<Q", value)


def push_string(buffer: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    push_u32(buffer, len(encoded))
    buffer += encoded


def append_tensor_f32(buffer: bytearray, tensor: Any) -> None:
    for value in tensor.reshape(-1).tolist():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite Q/K value encountered during serialization")
        buffer += struct.pack("<f", numeric)


def serialize_trace(
    records: list[CapturedRecord],
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    capture_id: str,
    source_dtype: str,
) -> bytes:
    buffer = bytearray(TRACE_MAGIC)
    push_u32(buffer, TRACE_VERSION)
    for value in (
        model_id,
        model_revision,
        tokenizer_id,
        tokenizer_revision,
        capture_id,
        source_dtype,
        TENSOR_STAGE,
    ):
        push_string(buffer, value)
    push_u32(buffer, len(records))

    for record in records:
        push_string(buffer, record.sample_id)
        push_u32(buffer, record.layer_index)
        push_u32(buffer, record.query_head_index)
        push_u32(buffer, record.kv_head_index)
        push_u64(buffer, record.query_position)
        push_u64(buffer, record.key_start_position)
        push_u32(buffer, record.head_dim)
        push_u32(buffer, record.key_count)
        buffer += struct.pack("<d", record.score_scale)
        if tuple(record.query.shape) != (record.head_dim,):
            raise ValueError("captured Q shape does not match metadata")
        if tuple(record.keys.shape) != (record.key_count, record.head_dim):
            raise ValueError("captured K shape does not match metadata")
        append_tensor_f32(buffer, record.query)
        append_tensor_f32(buffer, record.keys)
    return bytes(buffer)


def main() -> int:
    args = parse_args()
    samples = load_samples(args.samples_jsonl)
    samples_sha = sha256_file(args.samples_jsonl)

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
    except ImportError as error:
        print(f"missing capture dependency: {error}", file=sys.stderr)
        return 2

    tokenizer_id = args.tokenizer_id or args.model_id
    tokenizer_revision = args.tokenizer_revision or args.revision
    device = choose_device(torch, args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=tokenizer_revision,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    validate_model_config(model.config, args.layers, args.query_heads)
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise RuntimeError("loaded model did not retain attn_implementation=eager")
    model.eval()
    model.to(device)

    session = CaptureSession(set(args.layers), args.query_heads, args.positions)
    original_eager = install_qwen3_capture(torch, modeling_qwen3, session)

    try:
        with torch.inference_mode():
            for sample in samples:
                encoded = tokenizer(
                    sample.text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=args.max_tokens,
                    padding=False,
                )
                input_ids = encoded["input_ids"]
                if input_ids.shape[0] != 1:
                    raise RuntimeError("E4 capture requires one sample per forward pass")
                sequence_length = int(input_ids.shape[1])
                if sequence_length <= max(args.positions):
                    raise RuntimeError(
                        f"sample {sample.sample_id!r} tokenizes to {sequence_length} tokens; "
                        f"need at least {max(args.positions) + 1}"
                    )
                if "attention_mask" in encoded and not bool(encoded["attention_mask"].all()):
                    raise RuntimeError("E4 v1 capture rejects padded samples")

                session.begin_sample(sample.sample_id, sequence_length)
                device_inputs = {
                    name: tensor.to(device) for name, tensor in encoded.items()
                }
                model(
                    **device_inputs,
                    use_cache=False,
                    output_attentions=False,
                )
                session.end_sample()
    finally:
        modeling_qwen3.eager_attention_forward = original_eager

    expected_records = (
        len(samples)
        * len(args.layers)
        * len(args.query_heads)
        * len(args.positions)
    )
    if len(session.records) != expected_records:
        raise RuntimeError(
            f"capture produced {len(session.records)} records; expected {expected_records}. "
            "Installed Transformers Qwen3 attention may not match the E4 adapter contract."
        )
    if session.source_dtype is None:
        raise RuntimeError("capture produced no Q/K tensors")

    selection = {
        "layers": list(args.layers),
        "query_heads": list(args.query_heads),
        "positions": list(args.positions),
        "max_tokens": args.max_tokens,
    }
    selection_bytes = json.dumps(
        selection,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    selection_sha = hashlib.sha256(selection_bytes).hexdigest()
    capture_id = args.capture_id or (
        f"qwen3-e4-{samples_sha[:12]}-{selection_sha[:12]}"
    )

    trace_bytes = serialize_trace(
        session.records,
        args.model_id,
        args.revision,
        tokenizer_id,
        tokenizer_revision,
        capture_id,
        session.source_dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(trace_bytes)
    trace_sha = hashlib.sha256(trace_bytes).hexdigest()

    metadata_path = args.metadata_json or args.output.with_suffix(
        args.output.suffix + ".json"
    )

    model_config_dict = model.config.to_dict()
    rope_parameters = model_config_dict.get("rope_parameters")
    if not isinstance(rope_parameters, dict):
        raise RuntimeError(
            "Qwen3 config does not expose serialized rope_parameters"
        )

    rope_theta = rope_parameters.get("rope_theta")
    if not isinstance(rope_theta, (int, float)):
        raise RuntimeError(
            "Qwen3 rope_parameters does not contain numeric rope_theta"
        )

    rope_theta = float(rope_theta)
    if not math.isfinite(rope_theta) or rope_theta <= 0.0:
        raise RuntimeError(
            "Qwen3 rope_theta must be finite and positive"
        )

    metadata = {
        "format": "ADAQK01\\0",
        "format_version": TRACE_VERSION,
        "tensor_stage": TENSOR_STAGE,
        "model_id": args.model_id,
        "model_revision": args.revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "capture_id": capture_id,
        "source_dtype": session.source_dtype,
        "samples_jsonl": os.fspath(args.samples_jsonl),
        "samples_sha256": samples_sha,
        "sample_count": len(samples),
        "selection": selection,
        "record_count": len(session.records),
        "trace_file": os.fspath(args.output),
        "trace_sha256": trace_sha,
        "device": device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_config": {
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "num_attention_heads": int(model.config.num_attention_heads),
            "num_key_value_heads": int(model.config.num_key_value_heads),
            "head_dim": int(model.config.head_dim),
            "rope_parameters": rope_parameters,
            "rope_theta": rope_theta,
            "use_sliding_window": bool(
                getattr(model.config, "use_sliding_window", False)
            ),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("capture_status=complete")
    print(f"model_id={args.model_id}")
    print(f"model_revision={args.revision}")
    print(f"tokenizer_revision={tokenizer_revision}")
    print(f"capture_id={capture_id}")
    print(f"source_dtype={session.source_dtype}")
    print(f"sample_count={len(samples)}")
    print(f"record_count={len(session.records)}")
    print(f"samples_sha256={samples_sha}")
    print(f"trace_sha256={trace_sha}")
    print(f"trace={args.output}")
    print(f"metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
