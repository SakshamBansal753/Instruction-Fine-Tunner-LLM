# 🚀 Instruction Finetunner

**Hello young guns, I am Saksham Bansal 👋**  
Welcome to my project **Instruction Finetunner** — a hands-on implementation of instruction-based dataset preparation and fine-tuning pipeline for language models.

---

## 📌 Overview

This project demonstrates how to:

- Load and preprocess instruction-based datasets  
- Format prompts for instruction tuning  
- Tokenize text using GPT-style tokenization  
- Build a custom PyTorch dataset  
- Prepare data loaders with custom batching  

It is inspired by practical LLM training workflows and is a great starting point for anyone interested in **fine-tuning language models**.

---

## 📂 Dataset

The dataset is automatically downloaded from:

```
https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json
```

Each entry contains:

- **instruction** → Task description  
- **input** → Optional input context  
- **output** → Expected response  

---

## ⚙️ How It Works

### 1. Data Loading
- Downloads JSON dataset if not present locally  
- Loads and parses instruction-response pairs  

---

### 2. Prompt Formatting

Each example is converted into structured format:

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
...

### Input:
...

### Response:
...
```

---

### 3. Train / Validation / Test Split

- **85%** → Training  
- **10%** → Testing  
- **5%** → Validation  

---

### 4. Tokenization

Uses `tiktoken` with GPT-2 encoding:

- Converts full prompt-response into token IDs  
- Prepares sequences for training  

---

### 5. Custom Dataset (PyTorch)

```python
class InstructionDataset(Dataset)
```

- Stores tokenized sequences  
- Returns encoded text for each sample  

---

### 6. Custom Collate Function

Handles:
- Padding sequences  
- Creating uniform batch sizes  
- Preparing tensors for training  

---

## 🧠 Tech Stack

- Python 🐍  
- PyTorch 🔥  
- tiktoken (GPT tokenizer)  
- JSON dataset handling  

---

## 📦 Installation

```bash


pip install torch tiktoken
```

---

## ▶️ Usage

Run the notebook:

```bash
jupyter notebook "Instruction Finetunner.ipynb"
```

Or adapt the code into your training pipeline.

---

## 🎯 Goals of This Project

- Understand instruction tuning pipeline  
- Learn dataset formatting for LLMs  
- Build foundation for fine-tuning GPT-style models  
- Practice PyTorch data handling  

---

## 🚀 Future Improvements

- Add model fine-tuning (GPT / LLaMA / etc.)  
- Implement loss masking for responses only  
- Add evaluation metrics  
- Integrate HuggingFace Transformers  

---

## 🤝 Contributing

Feel free to fork this repo and improve it. Contributions are always welcome!

---

## 📬 Connect

If you found this helpful or want to collaborate, feel free to reach out 🚀

---

🔥 *Keep building, keep learning — young guns!*
