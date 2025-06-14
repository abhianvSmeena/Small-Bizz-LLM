# ==========================================================
# STEP 1: Install dependencies
# ==========================================================

!pip install transformers datasets peft trl accelerate bitsandbytes sentencepiece #

# ==========================================================
# STEP 2: Load model & tokenizer
# ==========================================================

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

model_id = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set pad_token to eos_token (important for CausalLMs like Phi-2)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", 
    torch_dtype="auto"
)

# ==========================================================
# STEP 3: Upload your dataset (restaurant_faq.json)
# ==========================================================

from google.colab import files
uploaded = files.upload()  # Upload your JSON file here

import json

with open("restaurant_faq.json") as f:
    data = json.load(f)

# Prepare dataset
train_data = [
    {
        "prompt": f"### Question: {item['question']}\n### Answer:",
        "response": item["answer"]
    }
    for item in data
]

# Convert to HF Dataset
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": [x['prompt'] for x in train_data],
    "response": [x['response'] for x in train_data],
})

# ==========================================================
# STEP 4: Apply LoRA using PEFT
# ==========================================================

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# ==========================================================
# STEP 5: Tokenize dataset
# ==========================================================

def tokenize_function(example):
    inputs = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(example["response"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# ==========================================================
# STEP 6: Trainer setup (with proper data collator)
# ==========================================================

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import os

os.environ["WANDB_DISABLED"] = "true"  # disable wandb

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Helps prevent OOM on Colab
    learning_rate=5e-5,
    bf16=False,
    fp16=False,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# Proper collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    label_names=["labels"] 
)


# ==========================================================
# STEP 7: Train model!
# ==========================================================

trainer.train()

# ==========================================================
# STEP 8: Save fine-tuned LoRA adapter
# ==========================================================

model.save_pretrained("./lora_finetuned_phi2")
tokenizer.save_pretrained("./lora_finetuned_phi2")

# Zip and download
!zip -r lora_finetuned_phi2.zip lora_finetuned_phi2
files.download("lora_finetuned_phi2.zip")