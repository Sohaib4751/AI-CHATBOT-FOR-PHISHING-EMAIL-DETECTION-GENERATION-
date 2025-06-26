import tkinter as tk
from tkinter import messagebox, scrolledtext
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification
import random
import re

# Load models and tokenizers
gpt2_path = "generator/final_gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_path, local_files_only=True)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_path, local_files_only=True)
gpt2_model.eval()

bert_path = "detector/bert_detector_model"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)
bert_model = BertForSequenceClassification.from_pretrained(bert_path, local_files_only=True)
bert_model.eval()

# Prompts and senders
random_prompts = [
    "Reset your credentials urgently",
    "Your package delivery failed",
    "Security alert: unusual login detected",
    "Verify your email account now",
    "You've won a reward! Claim now"
]

fake_senders = [
    "Apple Support", "Google Security", "PayPal Help Center", "Bank of America", "Amazon Support"
]

fake_subjects = [
    "Immediate Action Required", "Suspicious Login Attempt", "Your Account Has Been Locked",
    "Payment Failed", "Claim Your Prize Now"
]

def clean_output(text):
    return re.sub(r"http[s]?://\S*(cnn\.com|video/partners/email|/video/us).*", "[malicious link]", text)

def generate_email(prompt):
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, pad_token_id=gpt2_tokenizer.eos_token_id)
    return clean_output(gpt2_tokenizer.decode(output[0], skip_special_tokens=True))

def classify_email(email_text):
    inputs = bert_tokenizer(email_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    phishing_conf = probs[0][1].item()
    legit_conf = probs[0][0].item()
    label = "Phishing" if phishing_conf > 0.6 else "Legitimate"
    return label, phishing_conf, legit_conf

# GUI Setup
window = tk.Tk()
window.title("AI Phishing Email Generator and Detector")
window.geometry("700x600")

# Email Detection
tk.Label(window, text="Email to Detect:").pack()
detect_text = scrolledtext.ScrolledText(window, height=5)
detect_text.pack(fill="x")

def on_detect():
    email = detect_text.get("1.0", tk.END).strip()
    label, phishing, legit = classify_email(email)
    result_output.delete("1.0", tk.END)
    result_output.insert(tk.END, f"Prediction: {label}\nPhishing Confidence: {phishing:.2f}\nLegitimate Confidence: {legit:.2f}")

tk.Button(window, text="Detect Email", command=on_detect).pack(pady=5)

# Custom Generator
tk.Label(window, text="Custom Email Generator:").pack()

tk.Label(window, text="Sender:").pack()
sender_entry = tk.Entry(window)
sender_entry.pack(fill="x")

tk.Label(window, text="Subject:").pack()
subject_entry = tk.Entry(window)
subject_entry.pack(fill="x")

tk.Label(window, text="Concern/Message Type:").pack()
concern_entry = tk.Entry(window)
concern_entry.pack(fill="x")

def on_generate_custom():
    prompt = f"From: {sender_entry.get()}\nSubject: {subject_entry.get()}\n{concern_entry.get()}"
    email = generate_email(prompt)
    result_output.delete("1.0", tk.END)
    result_output.insert(tk.END, email)

tk.Button(window, text="Generate Custom Email", command=on_generate_custom).pack(pady=5)

# Random Generator

def on_generate_random():
    sender = random.choice(fake_senders)
    subject = random.choice(fake_subjects)
    prompt = random.choice(random_prompts)
    email = generate_email(f"From: {sender}\nSubject: {subject}\n{prompt}")
    result_output.delete("1.0", tk.END)
    result_output.insert(tk.END, email)

tk.Button(window, text="Generate Random Email", command=on_generate_random).pack(pady=5)

# Output Area
tk.Label(window, text="Output:").pack()
result_output = scrolledtext.ScrolledText(window, height=10)
result_output.pack(fill="both", expand=True)

# Run
window.mainloop()
