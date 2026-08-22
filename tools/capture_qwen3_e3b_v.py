#!/usr/bin/env python3
"""Capture natural Qwen3 pre-repeat GQA V tensors into ADAV01.

The capture observes Qwen3's value tensor at the input to the configured
attention implementation, before Transformers' eager `repeat_kv`.

For the frozen E3b protocol this means one full 512x128 V matrix per:
    sample x selected layer x physical KV head.

The capture does not alter attention semantics and delegates immediately to
the original eager attention implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRACE_MAGIC = b"ADAV01\x00"
TRACE_VERSION = 1

TENSOR_STAGE = (
    "attention_value_input_pre_repeat_kv"
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"

DEFAULT_MODEL_REVISION = (
    "c1899de289a04d12100db370d81485cdf75e47ca"
)

DEFAULT_LAYERS = (0, 13, 27)

EXPECTED_Q_HEADS = 16
EXPECTED_KV_HEADS = 8
EXPECTED_HEAD_DIM = 128


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str


@dataclass
class CapturedValueRecord:
    sample_id: str
    layer_index: int
    kv_head_index: int
    value_start_position: int
    value_count: int
    head_dim: int
    values: Any


class CaptureSession:
    def __init__(
        self,
        layers: set[int],
    ) -> None:
        self.layers = layers
        self.current_sample_id: str | None = None
        self.expected_sequence_length: int | None = None
        self.source_dtype: str | None = None

        self.records: list[
            CapturedValueRecord
        ] = []

        self.identities: set[
            tuple[str, int, int]
        ] = set()

    def begin_sample(
        self,
        sample_id: str,
        sequence_length: int,
    ) -> None:
        if self.current_sample_id is not None:
            raise RuntimeError(
                "capture session already has an active sample"
            )

        self.current_sample_id = sample_id
        self.expected_sequence_length = (
            sequence_length
        )

    def end_sample(self) -> None:
        self.current_sample_id = None
        self.expected_sequence_length = None


def parse_int_list(
    value: str,
    name: str,
) -> tuple[int, ...]:
    try:
        values = tuple(
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated integer list"
        ) from error

    if not values:
        raise argparse.ArgumentTypeError(
            f"{name} must not be empty"
        )

    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(
            f"{name} entries must be non-negative"
        )

    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            f"{name} entries must be unique"
        )

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--samples-jsonl",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )

    parser.add_argument(
        "--revision",
        default=DEFAULT_MODEL_REVISION,
    )

    parser.add_argument(
        "--tokenizer-id",
        default=None,
    )

    parser.add_argument(
        "--tokenizer-revision",
        default=None,
    )

    parser.add_argument(
        "--layers",
        default=",".join(
            map(str, DEFAULT_LAYERS)
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--capture-id",
        required=True,
    )

    parser.add_argument(
        "--metadata-json",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )

    args = parser.parse_args()

    args.layers = parse_int_list(
        args.layers,
        "layers",
    )

    if args.max_tokens <= 0:
        parser.error(
            "--max-tokens must be positive"
        )

    if (
        not args.revision
        or len(args.revision) < 12
    ):
        parser.error(
            "--revision must be immutable"
        )

    if (
        args.tokenizer_revision is not None
        and len(args.tokenizer_revision) < 12
    ):
        parser.error(
            "--tokenizer-revision must be immutable"
        )

    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_samples(
    path: Path,
) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                item = json.loads(
                    stripped
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(item, dict):
                raise ValueError(
                    f"line {line_number} must contain a JSON object"
                )

            sample_id = item.get(
                "sample_id"
            )

            text = item.get("text")

            if (
                not isinstance(
                    sample_id,
                    str,
                )
                or not sample_id
            ):
                raise ValueError(
                    f"line {line_number}: invalid sample_id"
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"duplicate sample_id: {sample_id}"
                )

            if (
                not isinstance(text, str)
                or not text
            ):
                raise ValueError(
                    f"line {line_number}: invalid text"
                )

            seen_ids.add(sample_id)

            samples.append(
                Sample(
                    sample_id=sample_id,
                    text=text,
                )
            )

    if not samples:
        raise ValueError(
            "samples JSONL contains no records"
        )

    return samples


def validate_model_config(
    config: Any,
    layers: tuple[int, ...],
) -> None:
    if (
        getattr(
            config,
            "model_type",
            None,
        )
        != "qwen3"
    ):
        raise RuntimeError(
            "E3b adapter requires model_type=qwen3"
        )

    if bool(
        getattr(
            config,
            "use_sliding_window",
            False,
        )
    ) or getattr(
        config,
        "sliding_window",
        None,
    ):
        raise RuntimeError(
            "E3b v1 rejects sliding-window attention"
        )

    num_layers = int(
        config.num_hidden_layers
    )

    num_heads = int(
        config.num_attention_heads
    )

    num_kv_heads = int(
        config.num_key_value_heads
    )

    head_dim = int(
        config.head_dim
    )

    if any(
        layer >= num_layers
        for layer in layers
    ):
        raise RuntimeError(
            "selected layer is outside model"
        )

    if num_heads != EXPECTED_Q_HEADS:
        raise RuntimeError(
            f"expected {EXPECTED_Q_HEADS} Q heads, "
            f"found {num_heads}"
        )

    if (
        num_kv_heads
        != EXPECTED_KV_HEADS
    ):
        raise RuntimeError(
            f"expected {EXPECTED_KV_HEADS} KV heads, "
            f"found {num_kv_heads}"
        )

    if head_dim != EXPECTED_HEAD_DIM:
        raise RuntimeError(
            f"expected head_dim={EXPECTED_HEAD_DIM}, "
            f"found {head_dim}"
        )

    if num_heads % num_kv_heads != 0:
        raise RuntimeError(
            "Q/KV head ratio is not integral"
        )


def install_qwen3_value_capture(
    torch: Any,
    qwen3_module: Any,
    session: CaptureSession,
) -> Any:
    original_eager = (
        qwen3_module.eager_attention_forward
    )

    def wrapper(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> Any:
        layer_index = int(
            module.layer_idx
        )

        sample_id = (
            session.current_sample_id
        )

        if (
            sample_id is not None
            and layer_index
            in session.layers
        ):
            if (
                query.ndim != 4
                or key.ndim != 4
                or value.ndim != 4
            ):
                raise RuntimeError(
                    "expected [B,H,T,D] Q/K/V"
                )

            if (
                query.shape[0] != 1
                or key.shape[0] != 1
                or value.shape[0] != 1
            ):
                raise RuntimeError(
                    "E3b v1 requires batch size 1"
                )

            if not (
                query.shape[2]
                == key.shape[2]
                == value.shape[2]
            ):
                raise RuntimeError(
                    "E3b prefill requires equal Q/K/V sequence lengths"
                )

            if (
                session.expected_sequence_length
                != int(value.shape[2])
            ):
                raise RuntimeError(
                    "attention sequence length differs from tokenizer input"
                )

            q_dtype = str(
                query.dtype
            ).removeprefix("torch.")

            k_dtype = str(
                key.dtype
            ).removeprefix("torch.")

            v_dtype = str(
                value.dtype
            ).removeprefix("torch.")

            if not (
                q_dtype
                == k_dtype
                == v_dtype
            ):
                raise RuntimeError(
                    "Q/K/V source dtypes differ"
                )

            if session.source_dtype is None:
                session.source_dtype = (
                    v_dtype
                )
            elif (
                session.source_dtype
                != v_dtype
            ):
                raise RuntimeError(
                    "capture observed multiple V dtypes"
                )

            q_heads = int(
                query.shape[1]
            )

            k_heads = int(
                key.shape[1]
            )

            v_heads = int(
                value.shape[1]
            )

            if (
                q_heads
                != EXPECTED_Q_HEADS
                or k_heads
                != EXPECTED_KV_HEADS
                or v_heads
                != EXPECTED_KV_HEADS
            ):
                raise RuntimeError(
                    "unexpected Q/K/V head count"
                )

            if q_heads % v_heads != 0:
                raise RuntimeError(
                    "Q/V head ratio is not integral"
                )

            if not (
                query.shape[3]
                == key.shape[3]
                == value.shape[3]
                == EXPECTED_HEAD_DIM
            ):
                raise RuntimeError(
                    "unexpected Q/K/V head dimension"
                )

            sequence_length = int(
                value.shape[2]
            )

            for kv_head in range(
                v_heads
            ):
                identity = (
                    sample_id,
                    layer_index,
                    kv_head,
                )

                if (
                    identity
                    in session.identities
                ):
                    raise RuntimeError(
                        f"duplicate V identity: {identity}"
                    )

                session.identities.add(
                    identity
                )

                values_cpu = (
                    value[
                        0,
                        kv_head,
                        :,
                        :,
                    ]
                    .detach()
                    .to(torch.float32)
                    .cpu()
                    .contiguous()
                )

                session.records.append(
                    CapturedValueRecord(
                        sample_id=sample_id,
                        layer_index=layer_index,
                        kv_head_index=kv_head,
                        value_start_position=0,
                        value_count=sequence_length,
                        head_dim=EXPECTED_HEAD_DIM,
                        values=values_cpu,
                    )
                )

        return original_eager(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )

    qwen3_module.eager_attention_forward = (
        wrapper
    )

    return original_eager


def push_u32(
    buffer: bytearray,
    value: int,
) -> None:
    if not 0 <= value <= 0xFFFF_FFFF:
        raise ValueError(
            f"u32 out of range: {value}"
        )

    buffer += struct.pack(
        "<I",
        value,
    )


def push_u64(
    buffer: bytearray,
    value: int,
) -> None:
    if not (
        0
        <= value
        <= 0xFFFF_FFFF_FFFF_FFFF
    ):
        raise ValueError(
            f"u64 out of range: {value}"
        )

    buffer += struct.pack(
        "<Q",
        value,
    )


def push_string(
    buffer: bytearray,
    value: str,
) -> None:
    encoded = value.encode(
        "utf-8"
    )

    push_u32(
        buffer,
        len(encoded),
    )

    buffer += encoded


def append_tensor_f32(
    buffer: bytearray,
    tensor: Any,
) -> None:
    values = (
        tensor
        .reshape(-1)
        .tolist()
    )

    for value in values:
        numeric = float(value)

        if not math.isfinite(
            numeric
        ):
            raise ValueError(
                "non-finite V value encountered during serialization"
            )

        buffer += struct.pack(
            "<f",
            numeric,
        )


def serialize_trace(
    records: list[CapturedValueRecord],
    model_id: str,
    model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    capture_id: str,
    source_dtype: str,
) -> bytes:
    buffer = bytearray(
        TRACE_MAGIC
    )

    push_u32(
        buffer,
        TRACE_VERSION,
    )

    for value in (
        model_id,
        model_revision,
        tokenizer_id,
        tokenizer_revision,
        capture_id,
        source_dtype,
        TENSOR_STAGE,
    ):
        push_string(
            buffer,
            value,
        )

    push_u32(
        buffer,
        len(records),
    )

    for record in records:
        push_string(
            buffer,
            record.sample_id,
        )

        push_u32(
            buffer,
            record.layer_index,
        )

        push_u32(
            buffer,
            record.kv_head_index,
        )

        push_u64(
            buffer,
            record.value_start_position,
        )

        push_u32(
            buffer,
            record.value_count,
        )

        push_u32(
            buffer,
            record.head_dim,
        )

        expected_shape = (
            record.value_count,
            record.head_dim,
        )

        if (
            tuple(
                record.values.shape
            )
            != expected_shape
        ):
            raise ValueError(
                "captured V shape does not match metadata"
            )

        append_tensor_f32(
            buffer,
            record.values,
        )

    return bytes(buffer)


def main() -> int:
    args = parse_args()

    samples = load_samples(
        args.samples_jsonl
    )

    samples_sha = sha256_file(
        args.samples_jsonl
    )

    try:
        import torch
        import transformers

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        import transformers.models.qwen3.modeling_qwen3 \
            as modeling_qwen3

    except ImportError as error:
        print(
            f"missing capture dependency: {error}"
        )
        return 2

    tokenizer_id = (
        args.tokenizer_id
        or args.model_id
    )

    tokenizer_revision = (
        args.tokenizer_revision
        or args.revision
    )

    load_kwargs = {
        "revision": args.revision,
        "trust_remote_code": False,
    }

    tokenizer_kwargs = {
        "revision":
            tokenizer_revision,
        "trust_remote_code": False,
    }

    if args.local_files_only:
        load_kwargs[
            "local_files_only"
        ] = True

        tokenizer_kwargs[
            "local_files_only"
        ] = True

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            tokenizer_id,
            **tokenizer_kwargs,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model_id,
            attn_implementation="eager",
            dtype=torch.bfloat16,
            **load_kwargs,
        )
    )

    validate_model_config(
        model.config,
        args.layers,
    )

    if (
        getattr(
            model.config,
            "_attn_implementation",
            None,
        )
        != "eager"
    ):
        raise RuntimeError(
            "loaded model did not retain eager attention"
        )

    model.eval()

    session = CaptureSession(
        set(args.layers)
    )

    original_eager = (
        install_qwen3_value_capture(
            torch,
            modeling_qwen3,
            session,
        )
    )

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

                input_ids = encoded[
                    "input_ids"
                ]

                if input_ids.shape[0] != 1:
                    raise RuntimeError(
                        "E3b capture requires one sample per forward pass"
                    )

                sequence_length = int(
                    input_ids.shape[1]
                )

                if (
                    sequence_length
                    != args.max_tokens
                ):
                    raise RuntimeError(
                        f"sample {sample.sample_id!r} "
                        f"tokenizes to {sequence_length}; "
                        f"expected exactly {args.max_tokens}"
                    )

                if (
                    "attention_mask"
                    in encoded
                    and not bool(
                        encoded[
                            "attention_mask"
                        ].all()
                    )
                ):
                    raise RuntimeError(
                        "E3b v1 rejects padded samples"
                    )

                session.begin_sample(
                    sample.sample_id,
                    sequence_length,
                )

                model(
                    **encoded,
                    use_cache=False,
                    output_attentions=False,
                )

                session.end_sample()

    finally:
        modeling_qwen3.eager_attention_forward = (
            original_eager
        )

    expected_records = (
        len(samples)
        * len(args.layers)
        * EXPECTED_KV_HEADS
    )

    if (
        len(session.records)
        != expected_records
    ):
        raise RuntimeError(
            f"capture produced {len(session.records)} "
            f"records; expected {expected_records}"
        )

    if (
        len(session.identities)
        != expected_records
    ):
        raise RuntimeError(
            "capture identity set is incomplete"
        )

    if session.source_dtype is None:
        raise RuntimeError(
            "capture produced no V tensors"
        )

    sample_rank = {
        sample.sample_id: index
        for index, sample
        in enumerate(samples)
    }

    session.records.sort(
        key=lambda record: (
            sample_rank[
                record.sample_id
            ],
            record.layer_index,
            record.kv_head_index,
        )
    )

    trace_bytes = serialize_trace(
        session.records,
        args.model_id,
        args.revision,
        tokenizer_id,
        tokenizer_revision,
        args.capture_id,
        session.source_dtype,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_bytes(
        trace_bytes
    )

    trace_sha = hashlib.sha256(
        trace_bytes
    ).hexdigest()

    metadata = {
        "format": "ADAV01\\0",
        "format_version":
            TRACE_VERSION,
        "tensor_stage":
            TENSOR_STAGE,
        "model_id":
            args.model_id,
        "model_revision":
            args.revision,
        "tokenizer_id":
            tokenizer_id,
        "tokenizer_revision":
            tokenizer_revision,
        "capture_id":
            args.capture_id,
        "source_dtype":
            session.source_dtype,
        "samples_sha256":
            samples_sha,
        "sample_count":
            len(samples),
        "record_count":
            len(session.records),
        "selection": {
            "layers":
                list(args.layers),
            "kv_heads":
                list(
                    range(
                        EXPECTED_KV_HEADS
                    )
                ),
            "sequence_length":
                args.max_tokens,
        },
        "model_config": {
            "num_attention_heads":
                int(
                    model.config
                    .num_attention_heads
                ),
            "num_key_value_heads":
                int(
                    model.config
                    .num_key_value_heads
                ),
            "q_per_kv":
                int(
                    model.config
                    .num_attention_heads
                )
                // int(
                    model.config
                    .num_key_value_heads
                ),
            "head_dim":
                int(
                    model.config
                    .head_dim
                ),
        },
        "storage_dtype":
            "f32-le",
        "trace_sha256":
            trace_sha,
        "torch_version":
            torch.__version__,
        "transformers_version":
            transformers.__version__,
    }

    metadata_bytes = (
        json.dumps(
            metadata,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    args.metadata_json.write_bytes(
        metadata_bytes
    )

    print(
        "capture_status=complete"
    )

    print(
        "format=ADAV01"
    )

    print(
        f"tensor_stage={TENSOR_STAGE}"
    )

    print(
        f"sample_count={len(samples)}"
    )

    print(
        f"record_count={len(session.records)}"
    )

    print(
        f"source_dtype={session.source_dtype}"
    )

    print(
        "q_heads="
        f"{int(model.config.num_attention_heads)}"
    )

    print(
        "kv_heads="
        f"{int(model.config.num_key_value_heads)}"
    )

    print(
        "q_per_kv="
        f"{int(model.config.num_attention_heads) // int(model.config.num_key_value_heads)}"
    )

    print(
        "head_dim="
        f"{int(model.config.head_dim)}"
    )

    print(
        f"sequence_length={args.max_tokens}"
    )

    print(
        f"samples_sha256={samples_sha}"
    )

    print(
        f"trace_sha256={trace_sha}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
