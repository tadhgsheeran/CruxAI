import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.prompts.generation_prompts import (
    build_user_prompt,
    get_system_prompt,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class GenerationService:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def _load_model(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        print(f"Loading generation model: {MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model.eval()

    def generate_answer(
        self,
        query: str,
        retrieved_results: list[dict],
        max_new_tokens: int = 250,
        prompt_version: str = "evidence_grounded",
        append_missing_citations: bool = True,
    ) -> str:
        self._load_model()
        
        context_sections = []

        for index, result in enumerate(retrieved_results, start=1):
            context_sections.append(
                f"[Source {index}: {result['source']}]\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_sections)

        system_message = get_system_prompt(
            prompt_version
        )

        user_message = build_user_prompt(
            query=query,
            context=context,
        )

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[
            0,
            model_inputs["input_ids"].shape[1]:,
        ]

        answer = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        source_citations = []

        for result in retrieved_results:
            citation = f"[{result['source']}]"

            if citation not in source_citations:
                source_citations.append(citation)

        citations_text = " ".join(source_citations)

        if (
            append_missing_citations
            and citations_text
            and not any(
                citation in answer
                for citation in source_citations
            )
        ):
            answer = (
                f"{answer}\n\n"
                f"Retrieved sources: {citations_text}"
            )

        return answer

generation_service = GenerationService()