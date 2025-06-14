# Small-Bizz-LLM
# 🦙 LLM Fine-Tuning Project — Custom Restaurant FAQ Bot using Phi-2, PEFT (LoRA) and Huggingface

---

## 🔥 Project Overview

This project demonstrates end-to-end fine-tuning of Microsoft's Phi-2 large language model (2.7B parameters) for a domain-specific task — building a **Restaurant FAQ Chatbot**.

The entire pipeline is built with **free resources only** (Google Colab + Huggingface ecosystem), and uses **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA** adapters to train large models on limited hardware.

The project showcases:

- ✅ Advanced LLM fine-tuning
- ✅ PEFT / LoRA based lightweight training
- ✅ Huggingface Transformers & Datasets
- ✅ TRL SFTTrainer (Supervised Fine-Tuning)
- ✅ Model inference after fine-tuning

---

## 🚀 Workflow Summary

### **Step 1 — Install Dependencies**

We install all required open-source libraries:
- transformers
- datasets
- peft
- trl
- accelerate
- bitsandbytes

### **Step 2 — Load Pre-trained Phi-2 Model**

- Downloaded Phi-2 model and tokenizer from Huggingface hub.
- Configured tokenizer to handle pad tokens properly.
- Loaded model on Google Colab with device_map="auto" to fit GPU resources.

### **Step 3 — Upload Custom Dataset**

- Uploaded custom restaurant FAQs dataset.
- Each sample contains a question-answer pair.

### **Step 4 — Prepare Data**

- Converted dataset into supervised instruction format.
- Simple prompt-response style input:  
  `"### Question: <question>\n### Answer: <answer>"`

### **Step 5 — Apply LoRA via PEFT**

- Defined LoRA configuration targeting key attention layers (`q_proj`, `v_proj`).
- Applied LoRA adapters to freeze most model weights and fine-tune small adapter layers.
- Allowed efficient training on small GPUs (Colab T4).

### **Step 6 — Initialize Trainer**

- Used `SFTTrainer` from Huggingface TRL to handle:
  - Dataset tokenization
  - Data collator
  - Loss computation
  - PEFT compatibility (solves the common Huggingface `Trainer` issues)

### **Step 7 — Fine-tune the Model**

- Trained the model for multiple epochs on the dataset.
- Only LoRA adapter weights updated (highly efficient resource usage).
- Training completed comfortably on free-tier GPU.

### **Step 8 — Save Fine-tuned Model**

- Saved both fine-tuned LoRA adapter and tokenizer.
- Exported model files for later inference.

### **Step 9 — Inference (Test the Model)**

- Loaded the base model + LoRA adapter for inference.
- Prompted the model with custom questions to verify accurate restaurant-specific responses.

---

## 🔧 Why LoRA (PEFT) was used?

- Large models like Phi-2 have billions of parameters.
- Full fine-tuning is expensive and impractical on free GPUs.
- PEFT (LoRA) allows training only a small subset of trainable parameters (~0.1% of total).
- Gives nearly full fine-tuning performance at a fraction of compute cost.

---

## 💡 Key Takeaways

- Demonstrates **real-world scalable fine-tuning** for LLMs using minimal resources.
- Uses **open-source only** stack.
- Follows modern industry best practices (PEFT, SFTTrainer).
- Highly efficient pipeline even for students, startups or small research teams.
- Project easily adaptable to other domains beyond Restaurant FAQ.

---

## ✅ Tech Stack

- Google Colab (Free Tier GPU)
- Huggingface Transformers
- Huggingface Datasets
- PEFT (Parameter Efficient Fine-Tuning)
- TRL (Supervised Fine-Tuning Trainer)
- Microsoft Phi-2 Model
- Python

---

## 📊 Resource Requirements

- ✅ Runs fully on free Google Colab
- ✅ ~20-40 minutes training time on T4 GPU
- ✅ Dataset can be as small as 50-100 samples for domain adaptation

---

## 🚩 Final Remarks

This project demonstrates full understanding of:
- Large Language Models (LLMs)
- Resource-efficient fine-tuning
- PEFT architecture
- Huggingface complete ecosystem integration
- End-to-end ML product deployment mindset

---

⭐ **This repository is part of my ML project portfolio for placement showcase.**

