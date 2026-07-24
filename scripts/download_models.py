from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Downloading tokenizer: {MODEL_NAME}")
AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Downloading model: {MODEL_NAME}")
AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
)

print("Generation model cached successfully.")
