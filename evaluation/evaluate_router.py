import json
from collections import Counter, defaultdict
from pathlib import Path

from app.orchestration.router import Intent, route_request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "router_benchmark.jsonl"
)


def load_benchmark() -> list[dict]:
    examples = []

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            examples.append(example)

    return examples


def calculate_precision_recall(
    expected_labels: list[str],
    predicted_labels: list[str],
    intents: list[str],
) -> dict[str, dict[str, float]]:
    metrics = {}

    for intent in intents:
        true_positive = sum(
            expected == intent and predicted == intent
            for expected, predicted in zip(
                expected_labels,
                predicted_labels,
            )
        )

        false_positive = sum(
            expected != intent and predicted == intent
            for expected, predicted in zip(
                expected_labels,
                predicted_labels,
            )
        )

        false_negative = sum(
            expected == intent and predicted != intent
            for expected, predicted in zip(
                expected_labels,
                predicted_labels,
            )
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )

        f1_denominator = precision + recall

        f1 = (
            2 * precision * recall / f1_denominator
            if f1_denominator
            else 0.0
        )

        metrics[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


def evaluate_router(
    verbose: bool = True,
) -> dict:
    benchmark = load_benchmark()

    expected_labels = []
    predicted_labels = []

    intent_correct = 0
    tools_correct = 0

    per_intent_total = Counter()
    per_intent_correct = Counter()

    confusion_matrix = defaultdict(Counter)
    failures = []

    for example in benchmark:
        question = example["question"]
        expected_intent = example["expected_intent"]
        expected_tools = sorted(
            example["expected_tools"]
        )

        decision = route_request(question)

        predicted_intent = decision.intent.value
        predicted_tools = sorted(decision.tools)

        expected_labels.append(expected_intent)
        predicted_labels.append(predicted_intent)

        per_intent_total[expected_intent] += 1

        confusion_matrix[
            expected_intent
        ][predicted_intent] += 1

        is_intent_correct = (
            predicted_intent == expected_intent
        )

        is_tools_correct = (
            predicted_tools == expected_tools
        )

        if is_intent_correct:
            intent_correct += 1

            per_intent_correct[
                expected_intent
            ] += 1

        if is_tools_correct:
            tools_correct += 1

        if (
            not is_intent_correct
            or not is_tools_correct
        ):
            failures.append(
                {
                    "question": question,
                    "expected_intent": (
                        expected_intent
                    ),
                    "predicted_intent": (
                        predicted_intent
                    ),
                    "expected_tools": (
                        expected_tools
                    ),
                    "predicted_tools": (
                        predicted_tools
                    ),
                    "reason": decision.reason,
                }
            )

    total = len(benchmark)

    intent_accuracy = (
        intent_correct / total
        if total
        else 0.0
    )

    tool_accuracy = (
        tools_correct / total
        if total
        else 0.0
    )

    intents = [
        intent.value
        for intent in Intent
    ]

    class_metrics = calculate_precision_recall(
        expected_labels=expected_labels,
        predicted_labels=predicted_labels,
        intents=intents,
    )

    per_intent_results = {}

    for intent in intents:
        correct = per_intent_correct[intent]
        intent_total = per_intent_total[intent]

        accuracy = (
            correct / intent_total
            if intent_total
            else 0.0
        )

        per_intent_results[intent] = {
            "examples": intent_total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": (
                class_metrics[intent]["precision"]
            ),
            "recall": (
                class_metrics[intent]["recall"]
            ),
            "f1": class_metrics[intent]["f1"],
        }

    confusion_matrix_dict = {
        expected_intent: {
            predicted_intent: confusion_matrix[
                expected_intent
            ][predicted_intent]
            for predicted_intent in intents
        }
        for expected_intent in intents
    }

    metrics = {
        "examples": total,
        "intent_correct": intent_correct,
        "intent_accuracy": intent_accuracy,
        "tool_selection_correct": tools_correct,
        "tool_selection_accuracy": tool_accuracy,
        "per_intent": per_intent_results,
        "confusion_matrix": (
            confusion_matrix_dict
        ),
        "failure_count": len(failures),
        "failures": failures,
    }

    if verbose:
        print("=" * 70)
        print("CRUXAI ROUTER EVALUATION")
        print("=" * 70)
        print(f"Examples: {total}")
        print(
            "Intent accuracy:",
            f"{intent_accuracy:.3f}",
        )
        print(
            "Tool-selection accuracy:",
            f"{tool_accuracy:.3f}",
        )

        print("\nPER-INTENT RESULTS")
        print("-" * 70)

        for intent in intents:
            result = per_intent_results[intent]

            print(
                f"{intent:30} "
                f"accuracy="
                f"{result['accuracy']:.3f} "
                f"precision="
                f"{result['precision']:.3f} "
                f"recall="
                f"{result['recall']:.3f} "
                f"f1={result['f1']:.3f}"
            )

        print("\nCONFUSION MATRIX")
        print("-" * 70)

        header = "Expected \\ Predicted"
        print(f"{header:30}", end="")

        for intent in intents:
            print(
                f"{intent[:8]:>10}",
                end="",
            )

        print()

        for expected_intent in intents:
            print(
                f"{expected_intent:30}",
                end="",
            )

            for predicted_intent in intents:
                count = confusion_matrix[
                    expected_intent
                ][predicted_intent]

                print(
                    f"{count:>10}",
                    end="",
                )

            print()

        print("\nFAILURES")
        print("-" * 70)

        if not failures:
            print("No routing failures.")
        else:
            for failure in failures:
                print(
                    f"Question: "
                    f"{failure['question']!r}"
                )
                print(
                    "Expected intent: "
                    f"{failure['expected_intent']}"
                )
                print(
                    "Predicted intent: "
                    f"{failure['predicted_intent']}"
                )
                print(
                    "Expected tools: "
                    f"{failure['expected_tools']}"
                )
                print(
                    "Predicted tools: "
                    f"{failure['predicted_tools']}"
                )
                print(
                    "Router reason: "
                    f"{failure['reason']}"
                )
                print("-" * 70)

    return metrics           
            
if __name__ == "__main__":
    evaluate_router()