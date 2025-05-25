import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import numpy as np
from classifier import MeaningClassifier
from capsule_embedding import CapsuleEmbedding


# === Загрузка датасета ===
df = pd.read_csv("dataset.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()
sublabels = df["sublabel"].tolist()

# === Обучение модели ===
model = MeaningClassifier()
embeddings = model.fit(texts, labels, sublabels)
capsules = [CapsuleEmbedding(e).get_capsule_matrix() for e in embeddings]  # (384, 2)
capsule_layer_1 = [cap[:, 1] for cap in capsules]  # Извлекаем 2-й слой (например, tanh)


# === PCA-визуализация ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(embeddings)

color_map = {
    "emotion": "red",
    "fact": "blue",
    "question": "green"
}

plt.figure(figsize=(10, 8))
for i in range(len(X_pca)):
    # Соединяем линией точки из orig и capsule

    x, y = X_pca[i]
    color = color_map[labels[i]]
    plt.scatter(x, y, color=color)
    plt.annotate(labels[i], (x + 0.01, y + 0.01), fontsize=7)
plt.title("PCA по эмбеддингам")
plt.grid(True)
plt.show()

# === PCA сравнение оригинала и capsule_layer_1 ===
X_pca_orig = PCA(n_components=2).fit_transform(embeddings)
X_pca_caps = PCA(n_components=2).fit_transform(capsule_layer_1)

diffs = np.linalg.norm(X_pca_orig - X_pca_caps, axis=1)
print("Среднее смещение:", np.mean(diffs))


fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Рисуем точки
for ax, X_pca, title in zip(
    axs,
    [X_pca_orig, X_pca_caps],
    ["Обычные эмбеддинги", "Капсулы (слой Z=1, tanh)"]
):
    for i in range(len(X_pca)):
        x, y = X_pca[i]
        color = color_map[labels[i]]
        ax.scatter(x, y, color=color)
        ax.annotate(labels[i], (x + 0.01, y + 0.01), fontsize=7)
    ax.set_title(title)
    ax.grid(True)

# Добавляем соединяющие линии только один раз
for (x1, y1), (x2, y2) in zip(X_pca_orig, X_pca_caps):
    axs[1].plot([x1, x2], [y1, y2], color="gray", linestyle="--", alpha=0.3)

plt.suptitle("Сравнение PCA: embedding vs capsule-layer[1]", fontsize=14)
plt.tight_layout()
plt.show()



# === Предсказание нового текста ===
test_text = "I can't stand how rude people are."
label, sublabel = model.predict(test_text)

print(f"\n Текст: \"{test_text}\"")
print(" Объяснение:")
print(f"   Основная категория: {label.upper()}")
print(f"   Подкатегория: {sublabel.capitalize()}")

description_map = {
    ("emotion", "positive"): "Выражает положительную эмоцию: радость, воодушевление, восхищение.",
    ("emotion", "negative"): "Сильная негативная эмоция: раздражение, гнев, грусть.",
    ("emotion", "unknown"): "Обнаружена эмоция, но тональность не распознана.",
    ("question", "scientific"): "Научный вопрос — стремление понять устройство мира.",
    ("question", "personal"): "Личное обращение или социальное взаимодействие.",
    ("question", "unknown"): "Обнаружен вопрос, но тип неизвестен.",
    ("fact", "scientific"): "Факт научного характера, описывающий явление или знание.",
    ("fact", "common"): "Общепринятый житейский факт или наблюдение.",
    ("fact", "unknown"): "Обнаружен факт, но подкатегория не определена."
}

explanation = description_map.get((label, sublabel), "Описание отсутствует.")
print(f"   Интерпретация: {explanation}")

#  Heatmap осмысленных координат

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Анализ важности координат
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
clf = RandomForestClassifier()
clf.fit(embeddings, y)

importance = clf.feature_importances_
top_n = 10
top_dims = np.argsort(importance)[-top_n:][::-1]

# Имена координат
named_dims = {
    top_dims[0]: "эмоциональность",
    top_dims[1]: "вопросительность",
    top_dims[2]: "научность"
}
for i in range(3, top_n):
    named_dims[top_dims[i]] = f"dim_{top_dims[i]}"

# Построение таблицы активаций
heatmap_data = []
for emb in embeddings:
    row = [emb[dim] for dim in top_dims]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, columns=[named_dims[d] for d in top_dims])
heatmap_df["label"] = labels
heatmap_df["text"] = texts

# Сортировка по метке и установка текста как индекс
heatmap_df.sort_values("label", inplace=True)
heatmap_df.set_index("text", inplace=True)


plt.figure(figsize=(14, 10))
sns.heatmap(
    heatmap_df.drop(columns=["label"]),
    cmap="coolwarm",
    linewidths=0.5,
    yticklabels=True
)
plt.title("Heatmap по смысловым координатам (с группировкой и подписями)")
plt.xlabel("Координата")
plt.ylabel("Фраза")
plt.tight_layout()
plt.savefig("heatmap_full.png", dpi=300)

plt.show()



input("\nНажмите Enter для выхода...")
