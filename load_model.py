from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B")
