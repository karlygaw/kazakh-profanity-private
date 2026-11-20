# Kazakh Profanity Detection in Product Reviews 🇰🇿

This repository contains a fine-tuned sequence classification model designed to read Kazakh product reviews or comments and predict whether they contain profanity.

## 🚀 Live Demo & Model Access

* **Interactive Demo (Gradio):** [KazProSpace](https://huggingface.co/spaces/karlygaw/KazProSpace)
* **Trained Model & Tokenizer:** [karlygaw/kazakh-profanity-xlmr](https://huggingface.co/karlygaw/kazakh-profanity-xlmr/)

---

## 📌 Task Definition

The input is a single comment or sentence, and the output is a binary label alongside a probability score:

* `0` = No Kazakh profanity (Safe)
* `1` = Kazakh profanity present (Toxic / Obscene)

---

## 📊 Dataset & Labeling

### Source and Context

The model was trained using a curated version of the [**Kaz-Cassandra**](https://huggingface.co/datasets/issai/kazsandra) dataset. While the original dataset covers general**** web text, it contains conversational and opinion-based language highly representative of Kazakh product reviews and user comments.

### Labeling Rules

The dataset was strictly labeled according to the following criteria:

* **Profanity (`1`):** Contains explicit swear words, heavily obscene slang, or toxic insults.
* **Safe (`0`):** Standard text, positive reviews, and **negative reviews without swear words**. A review stating "Бұл тауар өте нашар, сынған" (This product is very bad, broken) is labeled `0` because negative sentiment is not automatically profane.
* **Mixed-Language:** Comments containing Russian profanity mixed with Kazakh text (a common phenomenon in Kazakh internet slang) are labeled `1`.
* **Masked/Misspelled Words:** Obfuscated profanity (e.g., `p***da`, `ш#&с`) or common spelling variations were actively labeled as `1`.

### Data Splits & Preprocessing

To prevent data leakage, duplicates were removed, and near-duplicates were kept in the same split. **PII (Personally Identifiable Information)** was masked out of the training data and repository to protect reviewer privacy. Links, emails, phone numbers, and social media handles are replaced with special tokens (`<URL>`, `<EMAIL>`, `<PHONE>`, `<USER>`).

* **Total Samples:** 6,728
* **Train Set:** 4,714 samples (class distribution: ~66.6% safe, ~33.4% profane)
* **Validation Set:** 1,015 samples
* **Test Set:** 999 samples
* **Random Seed:** `42`

**Labeled Examples:**

1. `Өте жақсы қосымша, маған ұнады` -> `0` (Positive, Safe)
2. `Тауар уақытында келмеді, сапасы нашар!` -> `0` (Negative, but Safe)
3. `piiiiii$$$$daaa, қайтадан қате шықты` -> `1` (Profanity present, masked)

---

## 🧠 Model Choice & Architecture

I chose to fine-tune **XLM-Roberta (XLM-R)** for this classification task.

* **Why this model?** XLM-R is a highly effective multilingual model. Since Kazakh online reviews frequently feature a mix of Kazakh and Russian (code-switching), a robust multilingual tokenizer is required.
* **Input Handling:** Text is tokenized using the pretrained XLM-R SentencePiece tokenizer, which handles unknown words via subword tokenization. Punctuation is retained as it often correlates with aggressive sentiment.
* **Long Comments:** Inputs are truncated to a maximum length of `128` tokens. Product reviews rarely exceed this length, and profanity usually appears early in the text.

---

## 📈 Evaluation & Results

The model was evaluated once on the held-out test set after selecting hyperparameters using the validation data.

### Test Metrics

* **Test Set Size:** 999 examples
* **Precision (Class 1):** 0.945
* **Recall (Class 1):** 0.928
* **F1-Score (Class 1):** 0.936

### Confusion Matrix

<img width="499" height="388" alt="image" src="https://github.com/user-attachments/assets/45b34186-029e-4828-9385-e37dec0981c0" />

### Business Context: False Positives vs. False Negatives

* **False Positive (FP):** A clean review is flagged as profane and hidden from the store. This frustrates normal users and suppresses valid product feedback.
* **False Negative (FN):** A profane review slips through and is displayed publicly. This damages the platform's reputation.
* *Conclusion:* For product moderation, minimizing False Positives (high precision) is often prioritized so legitimate customer complaints aren't unfairly censored.

### Error Analysis & Limitations

Here are 3 challenging examples where the model made mistakes based on the validation data:

1. **"ШЕШЕН СЫГИЫНДЕР ! ЖАНА ЖУМЫСТАН КЕЛIП... НАМАЗ ОКЫП ТУРГАНДА..."**

* *Actual: 1 (Profane), Predicted: 0 (Safe).*
* **Reason:** The text contains a heavily misspelled Kazakh swear word ("СЫГИЫНДЕР"), but the rest of the long comment is filled with religious and everyday words ("НАМАЗ", "ЖУМЫСТАН"). The safe context completely overwhelmed the obfuscated profanity, causing a False Negative.

2. **"Сыныптастан келдым нехуя екен приложения"**

* *Actual: 1 (Profane), Predicted: 0 (Safe).*
* **Reason:** Code-switching and grammar issues. A Russian swear word ("нехуя") is embedded directly into a Kazakh sentence. Because of the informal spelling and lack of punctuation, the multilingual tokenizer likely split the word in a way that didn't trigger the profanity classification.

3. **"Каиф оин ғой каиф"**
<img width="959" height="529" alt="image" src="https://github.com/user-attachments/assets/b8f81c59-1a8e-40bf-b805-4ff151d82226" />

* *Actual: 0 (Safe), Predicted: 1 (Profane).*
* **Reason:** The user is expressing a positive emotion using street slang ("Каиф" / Kaif). While not a swear word, such slang frequently appears in toxic or highly informal internet datasets, leading the model to over-index on the slang and trigger a False Positive.
* **Suggested Improvement:** Augment the training data with more diverse masked/misspelled profanity (especially code-switched Russian-Kazakh slang) and ensure safe street slang (like "кайф", "сила") is well-represented in the positive class to reduce False Positives.

---

## 💻 Reproducibility & Setup

### 1. Dependencies & Configuration

Ensure you have the following installed:

```bash
pip install -q transformers==5.22.1 torch huggingface_hub

```

* **Random Seed used for training:** `42`
* **Training Settings:** Epochs: 3, Batch Size: 32, Learning Rate: 2e-5

### 2. Project Pipeline (Notebooks)

To reproduce the entire pipeline from scratch, the repository is split into 3 core Jupyter Notebooks. Run them sequentially in Google Colab or your local environment:

1. **`01_Data.ipynb`**: Handles dataset loading, cleaning, PII masking, and creating the train/validation/test splits.
2. **`02_Training.ipynb`**: Contains the code for fine-tuning the XLM-Roberta model using the prepared data.
3. **`03_Inference.ipynb`**: Handles test set evaluation (confusion matrix, metrics) and includes the inference pipeline for testing new strings.

### 3. Predicting on Custom Text (Inference)

You can run predictions directly using `03_Inference.ipynb`, or load the saved model and tokenizer directly from Hugging Face anywhere using the snippet below. The code automatically handles the necessary PII masking.

```python
import torch
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "karlygaw/kazakh-profanity-xlmr"
device = "cuda" if torch.cuda.is_available() else "cpu"

def mask_pii(text):
    """Applies the same personal-data masking used during training."""
    text = re.sub(r"https?://\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+\.\S+", "<EMAIL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    return re.sub(r"\+?\d[\d\-\s\(\)]{8,}\d", "<PHONE>", text)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()

def predict(text):
    masked_text = mask_pii(str(text)).strip()
    inputs = tokenizer(masked_text, truncation=True, max_length=128, return_tensors="pt").to(device)
    
    with torch.no_grad():
        prob = torch.softmax(model(**inputs).logits, dim=-1)[0, 1].item()
        
    return {
        "label": 1 if prob >= 0.7 else 0, 
        "probability": round(prob, 3)
    }

# Test it
review = "piiiiii$$$$daaa, қайтадан қате шықты"
print(predict(review)) 
# Output: {'label': 1, 'probability': 0.798}


```

## 🤝 Acknowledgements

* **AI Tools:** AI tools Gemini and ChatGPT were used to assist in generating boilerplate PyTorch inference code, and debugging regular expressions for PII masking. The data curation, architectural decisions, hyperparameter tuning, and error analysis are my own original contributions.
* **Dataset:** Original raw data sourced from [**Kaz-Cassandra**](https://huggingface.co/datasets/issai/kazsandra).
