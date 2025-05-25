# Fine-Tuning Large Language Models (LLMs) Repository

This repository contains implementations for fine-tuning large language models, specifically focusing on **Llama 2** using **QLoRA** (Quantized Low-Rank Adaptation) technique. The code demonstrates how to efficiently fine-tune large models with limited computational resources.

## 🚀 Overview

Fine-tuning large language models like Llama 2 traditionally requires substantial computational resources. This repository showcases how to overcome these limitations using:

- **QLoRA (Quantized Low-Rank Adaptation)**: Combines quantization with LoRA for memory-efficient training
- **4-bit Precision Training**: Drastically reduces VRAM usage
- **Parameter-Efficient Fine-Tuning (PEFT)**: Updates only a small subset of model parameters

## 🛠️ Features

- **Memory Efficient**: Fine-tune 7B+ parameter models on consumer GPUs (15GB VRAM)
- **QLoRA Implementation**: 4-bit quantization with LoRA adapters
- **Customizable Parameters**: Easy configuration of training hyperparameters
- **Hugging Face Integration**: Seamless model saving and sharing
- **TensorBoard Logging**: Training monitoring and visualization

## 📋 Requirements

### Hardware Requirements
- GPU with at least 15GB VRAM (Google Colab T4/V100 compatible)
- CUDA-compatible GPU for optimal performance

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd fine-tuning-llms
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Fine-Tuning

```python
python fine_tune_llama_2.py
```

### Training Process

1. **Dataset Loading**: Loads pre-formatted instruction dataset
2. **Model Quantization**: Applies 4-bit quantization using bitsandbytes
3. **LoRA Configuration**: Sets up low-rank adapters
4. **Training**: Fine-tunes using SFTTrainer
5. **Model Merging**: Combines LoRA weights with base model
6. **Hub Upload**: Pushes final model to Hugging Face Hub

## 🎯 Usage Examples

### Basic Inference

```python
from transformers import pipeline

# Load your fine-tuned model
pipe = pipeline(
    task="text-generation",
    model="your-username/Llama-2-7b-chat-finetune",
    max_length=200
)

# Generate response
prompt = "What is a large language model?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

### Custom Training Configuration

```python
# Modify hyperparameters in fine_tune_llama_2.py
learning_rate = 1e-4          # Adjust learning rate
num_train_epochs = 3          # Train for more epochs
lora_r = 32                   # Change LoRA rank
```

## 📊 Monitoring Training

The script includes TensorBoard integration for monitoring training progress:

```bash
tensorboard --logdir results/runs
```

Key metrics to monitor:

* Training loss
* Learning rate schedule
* Gradient norms

## 🎨 Customization Options

### 1. **Dataset Customization**

Replace the dataset with your own instruction-tuning data:

```python
dataset_name = "your-dataset-name"
```

Ensure your dataset follows the Llama 2 chat template:

```
<s>[INST] {instruction} [/INST] {response}</s>
```

### 2. **Model Selection**

Switch to different base models:

```python
model_name = "meta-llama/Llama-2-13b-chat-hf"  # For 13B model
```

### 3. **Training Configuration**

Adjust training parameters based on your hardware:

```python
per_device_train_batch_size = 2    # Reduce for less VRAM
gradient_accumulation_steps = 2     # Increase to maintain effective batch size
```

## 💡 Tips and Best Practices

### Memory Optimization

* Use `gradient_checkpointing = True` to trade compute for memory
* Reduce `per_device_train_batch_size` if running out of VRAM
* Enable `group_by_length = True` for efficiency

### Training Stability

* Monitor gradient norms (should stay below `max_grad_norm`)
* Use cosine learning rate schedule for better convergence
* Start with lower learning rates for stability

### Quality Improvements

* Use larger datasets for better performance
* Increase `lora_r` for more model capacity
* Train for multiple epochs with learning rate decay

## 🔍 New Fine-Tuning: DeepSeek-R1-Distill-Llama-8B on Medical CoT Dataset

This repository now also includes fine-tuning of **DeepSeek R1 Distill Llama 8B** using the **Unsloth** and **SFTTrainer** pipeline on a **Medical Chain-of-Thought (CoT)** dataset.

### ✅ Key Highlights:

* **Model**: `unsloth/DeepSeek-R1-Distill-Llama-8B`
* **Quantization**: 4-bit loading
* **Prompting Style**: Structured Medical Instruction with Step-by-Step reasoning
* **Use-case**: Medical Question Answering
* **Tools**: Unsloth for fast inference, WandB for monitoring, Hugging Face Hub for model storage

### 🧠 Prompt Format:

```text
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{question}

### Response:
<think>{model-generated answer}
```

### 🧪 Sample Inference:

```python
from unsloth import FastLanguageModel

question = "A 61-year-old woman with involuntary urine loss while sneezing undergoes a Q-tip test..."

inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

FastLanguageModel.for_inference(model)
outputs = model.generate(**inputs, max_new_tokens=1200)
print(tokenizer.decode(outputs[0]))
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   * Reduce batch size
   * Enable gradient checkpointing
   * Use smaller model variant

2. **Poor Model Performance**

   * Increase training epochs
   * Use larger/better dataset
   * Adjust LoRA parameters

3. **Training Instability**

   * Lower learning rate
   * Increase warmup steps
   * Check gradient clipping

## 📚 References and Resources

* [QLoRA Paper](https://arxiv.org/abs/2305.14314)
* [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
* [LoRA Paper](https://arxiv.org/abs/2106.09685)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers)
* [PEFT Library](https://github.com/huggingface/peft)
* [Unsloth for Fast Fine-Tuning](https://github.com/unslothai/unsloth)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

* Hugging Face for the transformers and datasets libraries
* Meta AI for the Llama 2 models
* DeepSeek for the R1 Distilled Llama 8B
* The open-source community for QLoRA, LoRA, and Unsloth
