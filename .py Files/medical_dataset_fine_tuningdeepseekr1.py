# -*- coding: utf-8 -*-
"""Medical_Dataset_Fine_TuningDeepSeekR1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/Harris-giki/FineTuningLLMs/blob/main/Medical_Dataset_Fine_TuningDeepSeekR1.ipynb

##Installing Relevant Packages
"""

# Commented out IPython magic to ensure Python compatibility.
# #installing important dependancies
# %%capture
# 
# !pip install unsloth # install unsloth; unsloth: Efficient fine-tuning and inference for LLMs
# !pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git # Also get the latest version Unsloth!

# modules for fine tuning
from unsloth import FastLanguageModel #module to optimize inference & fine-tuning
import torch # Import PyTorch; Deep learning framework used for training
from trl import SFTTrainer # Transformer Reinforcement Learning from HuggingFace which allows for supervised fine-tuning of the model (SFT)
from unsloth import is_bfloat16_supported # Checks if the hardware supports bfloat16 precision
# Hugging Face modules
from huggingface_hub import login # Lets you login to API
from transformers import TrainingArguments # Defines training hyperparameters
from datasets import load_dataset # Lets you load fine-tuning datasets
# Import weights and biases
import wandb

"""##Setting Manual Prompting for Token/API inputs"""

import os
import wandb
from huggingface_hub import login

# Prompt user for Hugging Face Token if not already set
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")  # Check if set as env variable
if not hugging_face_token:
    hugging_face_token = input("🔑 Enter your Hugging Face Token: ").strip()

# Login to Hugging Face
login(hugging_face_token)

# Prompt user for W&B Token if not already set
wnb_token = os.getenv("WANDB_API_KEY")  # Check if set as env variable
if not wnb_token:
    wnb_token = input("🔑 Enter your Weights & Biases API Key: ").strip()

# Login to W&B
wandb.login(key=wnb_token)

# Initialize W&B Run
run = wandb.init(
    project="Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset",
    job_type="training",
    anonymous="allow"
)

"""## Loading DeepSeek R1 and the Tokenizer"""

max_seq_length = 2048 # Define the maximum sequence length a model can handle (i.e. how many tokens can be processed at once)
dtype = None # Set to default
load_in_4bit = True # Enables 4 bit quantization — a memory saving optimization

# Load the DeepSeek R1 model and tokenizer using unsloth — imported using: from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",  # Load the pre-trained DeepSeek R1 model (8B parameter version)
    max_seq_length=max_seq_length, # Ensure the model can process up to 2048 tokens at once
    dtype=dtype, # Use the default data type (e.g., FP16 or BF16 depending on hardware support)
    load_in_4bit=load_in_4bit, # Load the model in 4-bit quantization to save memory
    token=hugging_face_token, # Use hugging face token
)

"""**Testing DeepSeek R1 on a medical use-case before fine-tuning**

The prompt_style variable is crucial in fine-tuning because it defines how the model is prompted during training. It provides a structured format for interacting with the model, ensuring that responses are logical, consistent, and aligned with the task at hand.
"""

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>{}"""

"""**Running inference on the model**

In this step, we test the DeepSeek R1 model by providing a medical question and generating a response.
The process involves the following steps:

Define a test question related to a medical case.

*   Define a test question related to a medical case.
*   Format the question using the structured prompt (prompt_style) to ensure the model follows a logical reasoning process.
*   Tokenize the input and move it to the GPU (cuda) for faster inference.
*   Generate a response using the model, specifying key parameters like max_new_tokens=1200 (limits response length).
*   Decode the output tokens back into text to obtain the final readable answer.
"""

# Creating a test medical question for inference
question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or
              sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Enable optimized inference mode for Unsloth models (improves speed and efficiency)
FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!

# Format the question using the structured prompt (`prompt_style`) and tokenize it
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")  # Convert input to PyTorch tensor & move to GPU

# Generate a response using the model
outputs = model.generate(
    input_ids=inputs.input_ids, # Tokenized input question
    attention_mask=inputs.attention_mask, # Attention mask to handle padding
    max_new_tokens=1200, # Limit response length to 1200 tokens (to prevent excessive output)
    use_cache=True, # Enable caching for faster inference
)
# Decode the generated output tokens into human-readable text
response = tokenizer.batch_decode(outputs)

# Extract and print only the relevant response part (after "### Response:")
print(response[0].split("### Response:")[1])

"""**Before starting fine-tuning — why are we fine-tuning in the first place?**

Even without fine-tuning, our model successfully generated a chain of thought and provided reasoning before delivering the final answer. The reasoning process is encapsulated within the <think> </think> tags. So, why do we still need fine-tuning? The reasoning process, while detailed, was long-winded and not concise. Additionally, we want the final answer to be consistent in a certain style.

##Fine-tuning step by step

### **Step 1 — Update the system prompt**

slightly change the prompt style for processing the dataset by adding the third placeholder for the complex chain of thought column.

***A complex chain of thoughts (CoT) ***is an advanced form of step-by-step reasoning. It differs from simple reasoning by:

Breaking problems into multiple intermediate steps rather than jumping to conclusions.
Explaining the rationale behind each step explicitly.
Handling multi-step logical connections, ensuring a structured and explainable response.

📌 Example Without CoT (Shallow Thinking)
* Question: "What are the possible diagnoses for a 55-year-old patient with chest pain?"
* Response: "Heart attack, angina, or GERD." (This is direct but lacks reasoning.)

📌 Example With Complex CoT (Deep Reasoning)
* Question: "What are the possible diagnoses for a 55-year-old patient with chest pain?"
* Complex Chain of Thought Response:
1. First, I will categorize the symptoms as cardiac or non-cardiac.
2. If the pain worsens with exertion and radiates to the left arm, I will consider angina or a heart attack.
3. If the pain occurs after meals and is relieved by antacids, it may be GERD.
4. I will also check for additional symptoms such as sweating, nausea, or shortness of breath to refine my diagnosis."
"""

# Updated training prompt style to add </think> tag
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

"""### **Step 2 — Download the fine-tuning dataset and format it for fine-tuning**

We will use the Medical O1 Reasoninng SFT found here on Hugging Face. From the authors: This dataset is used to fine-tune HuatuoGPT-o1, a medical LLM designed for advanced medical reasoning. This dataset is constructed using GPT-4o, which searches for solutions to verifiable medical problems and validates them through a medical verifier.
"""

# Download the dataset using Hugging Face — function imported using from datasets import load_dataset
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en", split = "train[0:500]",trust_remote_code=True) # Keep only first 500 rows
dataset

# Show an entry from the dataset
dataset[1]

# We need to format the dataset to fit our prompt training style
EOS_TOKEN = tokenizer.eos_token  # Define EOS_TOKEN which tells the model when to stop generating text during training
EOS_TOKEN

"""The function below is designed to process and format dataset examples for training a language model. It takes raw medical AI dataset examples and converts them into structured prompts using a predefined template (train_prompt_style).

Note:

1. The function iterates through the dataset in parallel, processing each row.
The zip() function ensures that for each row, we have:
* input (Question)
* cot (Complex Chain of Thought reasoning)
* output (Final Response)

2. The End of Sequence (EOS_TOKEN) is added at the end to mark the completion of each example.
This helps the model learn where one example ends and another begins during training.

"""

# Define formatting prompt function
def formatting_prompts_func(examples):  # Takes a batch of dataset examples as input
    inputs = examples["Question"]       # Extracts the medical question from the dataset
    cots = examples["Complex_CoT"]      # Extracts the chain-of-thought reasoning (logical step-by-step explanation)
    outputs = examples["Response"]      # Extracts the final model-generated response (answer)

    texts = []  # Initializes an empty list to store the formatted prompts

    # Iterate over the dataset, formatting each question, reasoning step, and response
    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN  # Insert values into prompt template & append EOS token
        texts.append(text)  # Add the formatted text to the list

    return {
        "text": texts,  # Return the newly formatted dataset with a "text" column containing structured prompts
    }

"""**.map()** below is used to apply the formatting_prompts_func function to each batch of examples in the dataset.

**The batched=True** argument means that multiple rows of the dataset will be processed at once, improving efficiency.
The output is stored in dataset_finetune, which is the new, formatted version of the dataset.
"""

# Update dataset formatting
dataset_finetune = dataset.map(formatting_prompts_func, batched = True)
dataset_finetune["text"][0]

"""### **Step 3 — Setting up the model using LoRA**

**Target Modules:**

The "target modules" refer to specific parts of the transformer architecture where LoRA adapters are applied to reduce the complexity of training while retaining performance. In the original transformer architecture (like the one from the Attention is All You Need paper), these modules are not explicitly named, but the concepts underlying them are present. Here's a breakdown:

"q_proj", "k_proj", "v_proj", "o_proj":

These are the projections used in the self-attention mechanism.
q_proj: The query projection, which takes the input and transforms it into a query vector.

* k_proj: The key projection, which transforms the input into a key vector.
* v_proj: The value projection, which transforms the input into a value vector.
* o_proj: The output projection, which is applied after the attention mechanism to get the final output.
* "gate_proj", "up_proj", "down_proj": These are typically found in the feed-forward network (FFN) part of the transformer. gate_proj: This is used in the gated mechanism inside the feed-forward network. up_proj and down_proj: These usually correspond to the upward and downward projections in the feed-forward layers of the transformer, transforming the data through various layers.

Note:

rslora refers to Rank-Stabilized LoRA, a technique to dynamically adjust the rank during fine-tuning to improve efficiency and performance.
"""

# Apply LoRA (Low-Rank Adaptation) fine-tuning to the model
model_lora = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank: Determines the size of the trainable adapters (higher = more parameters, lower = more efficiency)
    target_modules=[  # List of transformer layers where LoRA adapters will be applied
        "q_proj",   # Query projection in the self-attention mechanism
        "k_proj",   # Key projection in the self-attention mechanism
        "v_proj",   # Value projection in the self-attention mechanism
        "o_proj",   # Output projection from the attention layer
        "gate_proj",  # Used in feed-forward layers (MLP)
        "up_proj",    # Part of the transformer’s feed-forward network (FFN)
        "down_proj",  # Another part of the transformer’s FFN
    ],
    lora_alpha=16,  # Scaling factor for LoRA updates (higher values allow more influence from LoRA layers)
    lora_dropout=0,  # Dropout rate for LoRA layers (0 means no dropout, full retention of information)
    bias="none",  # Specifies whether LoRA layers should learn bias terms (setting to "none" saves memory)
    use_gradient_checkpointing="unsloth",  # Saves memory by recomputing activations instead of storing them (recommended for long-context fine-tuning)
    random_state=3407,  # Sets a seed for reproducibility, ensuring the same fine-tuning behavior across runs
    use_rslora=False,  # Whether to use Rank-Stabilized LoRA (disabled here, meaning fixed-rank LoRA is used)
    loftq_config=None,  # Low-bit Fine-Tuning Quantization (LoFTQ) is disabled in this configuration
)

"""### SFT Block

Initializing SFTTrainer, a supervised fine-tuning trainer from trl (Transformer Reinforcement Learning), to fine-tune our model efficiently on a dataset.
"""

# Initialize the fine-tuning trainer — Imported using from trl import SFTTrainer
trainer = SFTTrainer(
    model=model_lora,  # The model to be fine-tuned
    tokenizer=tokenizer,  # Tokenizer to process text inputs
    train_dataset=dataset_finetune,  # Dataset used for training
    dataset_text_field="text",  # Specifies which field in the dataset contains training text
    max_seq_length=max_seq_length,  # Defines the maximum sequence length for inputs
    dataset_num_proc=2,  # Uses 2 CPU threads to speed up data preprocessing

    # Define training arguments
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Number of examples processed per device (GPU) at a time
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps before updating weights
        num_train_epochs=1, # Full fine-tuning run
        warmup_steps=5,  # Gradually increases learning rate for the first 5 steps
        max_steps=60,  # Limits training to 60 steps (useful for debugging; increase for full fine-tuning)
        learning_rate=2e-4,  # Learning rate for weight updates (tuned for LoRA fine-tuning)
        fp16=not is_bfloat16_supported(),  # Use FP16 (if BF16 is not supported) to speed up training
        bf16=is_bfloat16_supported(),  # Use BF16 if supported (better numerical stability on newer GPUs)
        logging_steps=10,  # Logs training progress every 10 steps
        optim="adamw_8bit",  # Uses memory-efficient AdamW optimizer in 8-bit mode
        weight_decay=0.01,  # Regularization to prevent overfitting
        lr_scheduler_type="linear",  # Uses a linear learning rate schedule
        seed=3407,  # Sets a fixed seed for reproducibility
        output_dir="outputs",  # Directory where fine-tuned model checkpoints will be saved
    ),
)

"""##Model Training"""

# Start the fine-tuning process
trainer_stats = trainer.train()

# Save the fine-tuned model
wandb.finish()

"""##Run model inference after fine-tuning"""

question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing
              but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
              what would cystometry most likely reveal about her residual volume and detrusor contractions?"""

# Load the inference model using FastLanguageModel (Unsloth optimizes for speed)
FastLanguageModel.for_inference(model_lora)  # Unsloth has 2x faster inference!

# Tokenize the input question with a specific prompt format and move it to the GPU
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# Generate a response using LoRA fine-tuned model with specific parameters
outputs = model_lora.generate(
    input_ids=inputs.input_ids,          # Tokenized input IDs
    attention_mask=inputs.attention_mask, # Attention mask for padding handling
    max_new_tokens=1200,                  # Maximum length for generated response
    use_cache=True,                        # Enable cache for efficient generation
)

# Decode the generated response from tokenized format to readable text
response = tokenizer.batch_decode(outputs)
# Extract and print only the model's response part after "### Response:"
print(response[0].split("### Response:")[1])