import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# ✅ Load cleaned dataset
df = pd.read_csv("data/processed/balanced_cleaned_emails.csv")

# ✅ Extract phishing emails only (label == 1)
phishing_df = df[df['label'] == 1]

# ✅ Combine subject and body (if not already combined)
phishing_texts = phishing_df['text'].astype(str).tolist()
phishing_texts = [text.strip() for text in phishing_texts if len(text.strip()) > 10]

# ✅ Save to plain text file for GPT-2 training
with open("data/processed/phishing_train.txt", "w", encoding="utf-8") as f:
    for line in phishing_texts:
        f.write(line + "\n")

# ✅ Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ✅ Prepare dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/processed/phishing_train.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ✅ Define training arguments
training_args = TrainingArguments(
    output_dir="./generator/gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=10,
)

# ✅ Trainer setup and training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()

# ✅ Save final model and tokenizer for testing later
model.save_pretrained("generator/final_gpt2")
tokenizer.save_pretrained("generator/final_gpt2")

print("✅ Final model saved to generator/final_gpt2")
