# tuning/merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

base_model_id = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="auto", device_map="auto")

peft_model_id = "./lora_finetuned_phi2"
peft_config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(base_model, peft_model_id)

model = model.merge_and_unload()

save_dir = "./merged_phi2_lora"
model.save_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.save_pretrained(save_dir)