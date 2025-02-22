from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load JSON dataset
def load_custom_data():
    with open("data.json", "r") as f:
        data = json.load(f)
    return [{"input": d["question"], "output": d["answer"]} for d in data]

dataset = load_custom_data()

# Use a pre-trained model (GPT-2 for small scale)
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ Fix: Set padding token
tokenizer.pad_token = tokenizer.eos_token  

# Tokenize dataset and add labels
def tokenize_function(data):
    encoded = tokenizer(data["input"] + " " + data["output"], truncation=True, padding="max_length", max_length=256)
    encoded["labels"] = encoded["input_ids"].copy()  # ✅ Fix: Assign labels
    return encoded

# Convert dataset into Hugging Face Dataset format
dataset = Dataset.from_dict({key: [tokenize_function(d)[key] for d in dataset] for key in ["input_ids", "attention_mask", "labels"]})

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch"
)

# Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./my_llm")
tokenizer.save_pretrained("./my_llm")
