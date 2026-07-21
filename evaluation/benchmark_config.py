from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Configuration for one CruxAI benchmark run.
    """

    experiment_name: str = "stage5_baseline"

    retrieval_method: str = "dense"
    reranking_enabled: bool = False

    chunk_size: int = 500
    chunk_overlap_paragraphs: int = 2
    top_k: int = 3

    prompt_version: str = "evidence_grounded"

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization: str = "none"
    max_new_tokens: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE_CONFIG = BenchmarkConfig()