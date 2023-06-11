# 翻译示例

from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

results = translator("Ce cours est produit par Hugging Face.")

print(results)