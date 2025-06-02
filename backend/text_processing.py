import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Вместо "cointegrated/rubert-tiny-sentiment-balanced" указываем локальный путь
local_path = "models/rubert-tiny-sentiment-balanced"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForSequenceClassification.from_pretrained(local_path)

if torch.cuda.is_available():
    model.cuda()


def get_sentiment(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
            model.device
        )

        logits = model(**inputs).logits
        proba = torch.sigmoid(logits).cpu().numpy()[0]
        label = model.config.id2label[proba.argmax()]

    return label, proba
