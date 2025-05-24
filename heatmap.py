import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка датасета
df = pd.read_csv("dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Эмбеддинги
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# Вычисление важности координат
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
clf = RandomForestClassifier()
clf.fit(embeddings, y)

importance = clf.feature_importances_
top_n = 10
top_dims = np.argsort(importance)[-top_n:][::-1]

# Присвоим имена первым 3 осям, остальные — по индексам
named_dims = {
    top_dims[0]: "эмоциональность",
    top_dims[1]: "вопросительность",
    top_dims[2]: "научность"
}
for i in range(3, top_n):
    named_dims[top_dims[i]] = f"dim_{top_dims[i]}"

# Построение таблицы
heatmap_data = []
for emb in embeddings:
    row = [emb[dim] for dim in top_dims]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, columns=[named_dims[d] for d in top_dims])
heatmap_df["label"] = labels
heatmap_df["text"] = texts

# Визуализация
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df.drop(columns=["label", "text"]), cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap по смысловым координатам эмбеддингов")
plt.xlabel("Координата")
plt.ylabel("Фраза")
plt.tight_layout()
plt.show()
