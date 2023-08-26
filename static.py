import json

file_path = "D:/code/prompt-ABSA/dataset/data/laptops16/test.json"

with open(file_path, "r", encoding='utf-8') as f:
    category, _, data = json.load(f)
sentence = len(data)

positive, neutral, negative = 0, 0, 0
for elem in data:
    for i in elem[1:]:
        if i['polarity'] == 'positive':
            positive += 1
        elif i['polarity'] == 'neutral':
            neutral += 1
        elif i['polarity'] == 'negative':
            negative += 1
print(f"category:{category}, sentence:{sentence}, positive:{positive}, neutral:{neutral}, negative:{negative}")
