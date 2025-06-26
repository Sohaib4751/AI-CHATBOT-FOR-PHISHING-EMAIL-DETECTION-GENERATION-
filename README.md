# 🛡️ AI Chatbot for Phishing Email Detection and Generation

An academic project demonstrating the development of an AI-powered chatbot capable of **generating simulated phishing emails** using GPT-2 and **detecting phishing emails** using BERT. This tool is intended for **educational, training, and awareness** purposes only.

---

## 📌 Features

- 🔒 **Phishing Detection** – BERT-based classifier trained on a curated dataset of phishing and legitimate emails.
- ✉️ **Phishing Email Generation** – Fine-tuned GPT-2 model capable of generating realistic phishing-style messages.
- 💬 **Chatbot Interface** – Terminal-based menu interface combining both generator and detector functionality.
- 🔗 **Malicious Link Simulation** – Automatically replaces known domains like `cnn.com` with `[malicious link]` in generated emails.
- 🧪 **Random & Custom Prompt Generation** – Users can provide custom input or let the model auto-generate prompts.

---

## 🛠️ Environment Setup

### 1. Create Virtual Environment
```bash
python -m venv chatbot_env
```

### 2. Activate Environment
- **Windows PowerShell**:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\chatbot_env\Scripts\Activate
```

### 3. Install Dependencies
```bash
pip install torch transformers flask scikit-learn pandas jupyterlab
```

---

## 📂 Project Structure

```
AI_Phishing_Chatbot/
├── chatbot_env/              # Virtual environment
├── data/
│   ├── raw/                  # Raw dataset files
│   └── processed/            # Cleaned CSVs
├── generator/
│   ├── final_gpt2/           # Fine-tuned GPT-2 model
│   └── train_generator.py
├── detector/
│   ├── bert_detector_model/  # Trained BERT model
│   └── train_detector.py
├── chatbot/
│   └── chatbot_interface.py  # Chatbot logic
├── utils/
│   └── preprocess.py         # Helper preprocessing scripts
├── test_generator.py         # GPT-2 email test script
├── app.py                    # Main entry point
└── phishing_train.txt        # Text data used to fine-tune GPT-2
```

---

## 📊 Datasets Used

- **Phishing Email Dataset** (Kaggle)  
  Includes labeled email samples from Enron, CEAS, Nazario, SpamAssassin, etc.

- **Balanced Dataset**:  
  - 9,182 phishing emails  
  - 9,182 legitimate emails  
  Used to fine-tune BERT classifier for binary classification.

---

## 🤖 Models

### 🔹 GPT-2 (Phishing Generator)
- Fine-tuned using phishing email text data.
- Trained over **3 epochs**, **batch size = 2**.
- Final loss: `~0.49`.

### 🔹 BERT (Email Detector)
- Model: `bert-base-uncased`
- Trained on 2000 emails (1600 train, 400 test).
- Epochs: 3
- Final loss: `~0.066`

---

## 💬 Chatbot Options

1. **Generate Phishing Email (User Prompt)**  
   Enter:
   - Sender Name (e.g., `PayPal Help Center`)
   - Subject (e.g., `Account Locked`)
   - Type (e.g., `Password Reset`)  
   GPT-2 generates a phishing-style message accordingly.

2. **Generate Phishing Email (Random Prompt)**  
   Automatically uses randomized inputs for sender, subject, and message type.

3. **Detect Email (Phishing/Legitimate)**  
   User inputs full email text.  
   BERT classifier provides label + confidence scores.

4. **Exit**  
   Ends chatbot session.

---

## 🚨 Example Outputs

### Phishing Detection
```
📨 Email: Your PayPal account has been locked. Click here to unlock it now.
🧠 Prediction: Phishing
🔐 Phishing Confidence: 1.00
📨 Legitimate Confidence: 0.00
```

### Phishing Generation
```
Prompt: Urgent action required
Generated: Dear user, we noticed unusual activity on your account...
Please verify your credentials by visiting [malicious link]...
```

---

## ⚠️ Disclaimer

This tool is intended **strictly for educational and awareness purposes**. Do not use generated emails for any malicious activity. Misuse of this code may lead to ethical, academic, or legal consequences.

---

## 🧠 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Kaggle Datasets](https://www.kaggle.com/)
- VS Code and JupyterLab for development
