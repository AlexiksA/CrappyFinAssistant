from transformers import pipeline
from sentiment_digest import _fetch_page

classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

res = classifier(["This stock is awesome"])

print(res)
