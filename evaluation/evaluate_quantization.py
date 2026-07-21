import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from app.prompts.generation_prompts import (
    build_user_prompt,
    get_system_prompt,
)
from app.retrieval.service import (
    retrieval_service,
)
from evaluation.evaluate_prompts import (
    calculate_topic_coverage,
    load_benchmark,
)

from torchao.quantization import (
    Int8WeightOnlyConfig,
    quantize_,
)

import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIRECTORY
    / "quantization_ablation.json"
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPT_VERSION = "evidence_grounded"

MAX_NEW_TOKENS = 100


def get_memory_mb() -> float:
    """
    Return current process resident memory in megabytes.
    """
    process = psutil.Process(os.getpid())

    return (
        process.memory_info().rss
        / (1024 ** 2)
    )

def get_serialized_model_size_mb(
    model,
) -> float:
    """
    Measure model storage by serializing its state dictionary.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".pt",
        delete=False,
    ) as temporary_file:
        temporary_path = temporary_file.name

    try:
        torch.save(
            model.state_dict(),
            temporary_path,
        )

        size_bytes = os.path.getsize(
            temporary_path
        )

        return size_bytes / (1024 ** 2)

    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

def format_context(
    retrieved_results: list[dict],
) -> str:
    """
    Format retrieved chunks for the generation prompt.
    """
    context_parts = []

    for result in retrieved_results:
        source = result["source"]
        text = result["text"]

        context_parts.append(
            f"Source: [{source}]\n{text}"
        )

    return "\n\n".join(context_parts)


def load_model(
    mode: str,
) -> tuple:
    """
    Load either the FP32 or INT8 model on CPU.
    """
    if mode not in {
        "fp32",
        "int8_weights_only",
    }:
        raise ValueError(
            "mode must be fp32 or int8_weights_only."
        )

    tokenizer_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    tokenizer_load_seconds = (
        time.perf_counter()
        - tokenizer_start
    )

    model_start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map=None,
    )

    model.to("cpu")
    model.eval()

    base_model_load_seconds = (
        time.perf_counter()
        - model_start
    )

    quantization_seconds = 0.0

    if mode == "int8_weights_only":
        quantization_start = time.perf_counter()

        quantize_(
            model,
            Int8WeightOnlyConfig(),
        )

        quantization_seconds = (
            time.perf_counter()
            - quantization_start
        )

        model.eval()
    
    total_load_seconds = (
        tokenizer_load_seconds
        + base_model_load_seconds
        + quantization_seconds
    )

    return (
        tokenizer,
        model,
        {
            "tokenizer_load_seconds": (
                tokenizer_load_seconds
            ),
            "base_model_load_seconds": (
                base_model_load_seconds
            ),
            "quantization_seconds": (
                quantization_seconds
            ),
            "total_load_seconds": (
                total_load_seconds
            ),
        },
    )


def generate_answer(
    tokenizer,
    model,
    query: str,
    retrieved_results: list[dict],
) -> tuple[str, float, int]:
    """
    Generate one deterministic answer and return its latency
    and generated-token count.
    """
    context = format_context(
        retrieved_results
    )

    messages = [
        {
            "role": "system",
            "content": get_system_prompt(
                PROMPT_VERSION
            ),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                query=query,
                context=context,
            ),
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    )

    input_token_count = int(
        inputs["input_ids"].shape[1]
    )

    start = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    latency = time.perf_counter() - start

    generated_ids = output_ids[
        0,
        input_token_count:,
    ]

    generated_token_count = int(
        generated_ids.shape[0]
    )

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return (
        answer,
        latency,
        generated_token_count,
    )


def run_worker(
    mode: str,
) -> dict:
    """
    Run one model configuration in an isolated process.
    """
    memory_before_load_mb = get_memory_mb()

    tokenizer, model, load_metrics = (
        load_model(mode)
    )

    serialized_model_size_mb = (
        get_serialized_model_size_mb(
            model
        )
    )

    memory_after_load_mb = get_memory_mb()

    examples = [
        example
        for example in load_benchmark()
        if not example["should_abstain"]
    ]

    latencies = []
    generated_tokens = []
    topic_coverage_scores = []
    answer_lengths = []
    details = []

    for index, example in enumerate(examples):
        question = example["question"]

        retrieved_results = (
            retrieval_service.search(
                query=question,
                top_k=3,
            )
        )

        answer, latency, token_count = (
            generate_answer(
                tokenizer=tokenizer,
                model=model,
                query=question,
                retrieved_results=(
                    retrieved_results
                ),
            )
        )

        topic_coverage = (
            calculate_topic_coverage(
                answer=answer,
                expected_topics=example[
                    "expected_topics"
                ],
            )
        )

        tokens_per_second = (
            token_count / latency
            if latency > 0
            else 0.0
        )

        latencies.append(latency)
        generated_tokens.append(
            token_count
        )
        topic_coverage_scores.append(
            topic_coverage
        )
        answer_lengths.append(
            len(answer.split())
        )

        details.append(
            {
                "question": question,
                "answer": answer,
                "topic_coverage": (
                    topic_coverage
                ),
                "latency_seconds": latency,
                "generated_tokens": (
                    token_count
                ),
                "tokens_per_second": (
                    tokens_per_second
                ),
            }
        )

        print(
            f"{mode} | "
            f"example={index + 1}/{len(examples)} | "
            f"latency={latency:.2f}s | "
            f"tokens={token_count} | "
            f"tokens_per_second="
            f"{tokens_per_second:.2f}",
            flush=True,
        )

    total_latency = sum(latencies)
    total_generated_tokens = sum(
        generated_tokens
    )

    result = {
        "mode": mode,
        "model": MODEL_NAME,
        "device": "cpu",
        "dtype": (
            "float32"
            if mode == "fp32"
            else "int8_weights_only"
        ),
        "serialized_model_size_mb": (
            serialized_model_size_mb
        ),
        "examples": len(examples),
        **load_metrics,
        "memory_before_load_mb": (
            memory_before_load_mb
        ),
        "memory_after_load_mb": (
            memory_after_load_mb
        ),
        "model_memory_delta_mb": (
            memory_after_load_mb
            - memory_before_load_mb
        ),
        "average_latency_seconds": (
            total_latency / len(examples)
            if examples
            else 0.0
        ),
        "total_latency_seconds": (
            total_latency
        ),
        "average_tokens_per_second": (
            total_generated_tokens
            / total_latency
            if total_latency > 0
            else 0.0
        ),
        "mean_topic_coverage": (
            sum(topic_coverage_scores)
            / len(topic_coverage_scores)
            if topic_coverage_scores
            else 0.0
        ),
        "average_response_words": (
            sum(answer_lengths)
            / len(answer_lengths)
            if answer_lengths
            else 0.0
        ),
        "examples_detail": details,
    }

    worker_path = (
        RESULTS_DIRECTORY
        / f"quantization_{mode}_worker.json"
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with worker_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return result


def run_parent() -> None:
    """
    Run each configuration in a separate process so memory
    measurements do not contaminate one another.
    """
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    modes = [
        "fp32",
        "int8_weights_only",
    ]

    results = []

    for mode in modes:
        print()
        print("=" * 70)
        print(f"Running mode: {mode}")
        print("=" * 70)

        worker_path = (
            RESULTS_DIRECTORY
            / f"quantization_{mode}_worker.json"
        )

        if worker_path.exists():
            worker_path.unlink()

        command = [
            sys.executable,
            "-m",
            "evaluation.evaluate_quantization",
            "--worker",
            mode,
        ]

        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        with worker_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            results.append(
                json.load(file)
            )

    output = {
        "experiment": (
            "quantization_ablation"
        ),
        "model": MODEL_NAME,
        "device": "cpu",
        "prompt_version": PROMPT_VERSION,
        "max_new_tokens": MAX_NEW_TOKENS,
        "results": results,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("QUANTIZATION RESULTS")
    print("=" * 70)

    for result in results:
        print(
            f"mode={result['mode']:16} | "
            f"load={result['total_load_seconds']:.2f}s | "
            f"serialized_size="
            f"{result['serialized_model_size_mb']:.1f}MB | "
            f"latency="
            f"{result['average_latency_seconds']:.2f}s | "
            f"tokens_per_second="
            f"{result['average_tokens_per_second']:.2f} | "
            f"topic_coverage="
            f"{result['mean_topic_coverage']:.3f} | "
            f"words="
            f"{result['average_response_words']:.1f}"
        )

    print()
    print(f"Saved results to: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        choices=[
            "fp32",
            "int8_weights_only",
        ],
    )

    arguments = parser.parse_args()

    if arguments.worker:
        run_worker(arguments.worker)
    else:
        run_parent()


if __name__ == "__main__":
    main()