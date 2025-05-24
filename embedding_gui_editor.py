# embedding_gui_editor.py
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from classifier import MeaningClassifier

# === Загрузка модели ===
st.title(" Embedding GUI Editor")
input_text = st.text_area("Введите фразу:", "I love how peaceful it feels in the mountains.")

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
classifier = MeaningClassifier()

# Загрузка и обучение на датасете 
@st.cache_resource
def train_classifier():
    import pandas as pd
    df = pd.read_csv("dataset.csv")
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    sublabels = df["sublabel"].tolist()
    classifier.fit(texts, labels, sublabels)
    return classifier

# Получение топ-координат по категориям
@st.cache_data
def get_coord_map():
    import pandas as pd
    df = pd.read_csv("dataset.csv")
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    return classifier.get_top_coordinates_by_class(texts, labels, top_n=20)


coord_map = get_coord_map()


classifier = train_classifier()

# === Получение эмбеддинга ===
embedding = model.encode([input_text])[0]
original_embedding = embedding.copy()

# Интерфейс маскировк
st.subheader("Отключение / модификация осей")
selected_dims = st.multiselect("Выберите координаты для изменения (0–383):", options=list(range(len(embedding))), default=[])

mod_type = st.radio("Режим модификации:", ["Обнулить", "Ослабить (x0.5)", "Инвертировать (x-1)"])
if selected_dims:
    st.markdown("### Выбранные координаты и их предполагаемые значения:")
    for dim in selected_dims:
        tagged = []
        for cat, coords in coord_map.items():
            for d, score in coords:
                if d == dim:
                    tagged.append(f"{cat} ({score:.3f})")
        tag_str = ", ".join(tagged) if tagged else " Неопределено"
        st.write(f"• dim_{dim} → {tag_str}")


# === Применение маски ===
for dim in selected_dims:
    if mod_type == "Обнулить":
        embedding[dim] = 0
    elif mod_type == "Ослабить (x0.5)":
        embedding[dim] *= 0.5
    elif mod_type == "Инвертировать (x-1)":
        embedding[dim] *= -1

# === Визуализация изменений ===
st.subheader("Сравнение оригинала и модифицированного")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(original_embedding, label="Original", alpha=0.6)
ax.plot(embedding, label="Modified", alpha=0.6)
ax.set_title("Embedding Vector Comparison")
ax.legend()
st.pyplot(fig)
st.sidebar.subheader(" Координаты по категориям")

for category, dims in coord_map.items():
    st.sidebar.markdown(f"**{category.upper()}**")
    for dim, score in dims:
        st.sidebar.write(f"• dim_{dim} — важность: {score:.3f}")


# === Предсказание смысла ===
st.subheader("Предсказанный смысл")
label, sublabel = classifier.predict(input_text)

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

st.markdown(f"**Категория:** `{label.upper()}`\n\n**Подкатегория:** `{sublabel}`\n\n**Интерпретация:** {explanation}")

# === Подсказка ===
st.info("💡 Изменяйте embedding вручную и наблюдайте, как меняется смысл! Это прямой редактор мышления модели.")
# === Автоматический подбор осей под нужный смысл ===
st.subheader("Автоматический подбор координат")

desired_sublabel = st.selectbox("Желаемая подкатегория:", [
    "positive", "negative", "scientific", "personal", "common", "unknown"
])

auto_run = st.button(" Попробовать изменить embedding")

if auto_run:
    from copy import deepcopy

    def try_modification(embedding, dims, mod_fn):
        new_emb = deepcopy(embedding)
        for dim in dims:
            new_emb[dim] = mod_fn(new_emb[dim])
        return new_emb

    mod_fns = {
        "Обнулить": lambda x: 0,
        "Ослабить (x0.5)": lambda x: x * 0.5,
        "Инвертировать (x-1)": lambda x: -x
    }

    results = []

    for mod_name, fn in mod_fns.items():
        for category, dims in coord_map.items():
            dim_ids = [d for d, _ in dims]
            modified_emb = try_modification(original_embedding, dim_ids, fn)
            pred_label, pred_sublabel = classifier.predict_by_embedding(modified_emb)

            if pred_sublabel == desired_sublabel:
                results.append((category, mod_name, dim_ids, pred_label, pred_sublabel))
                break  # Успешный результат найден — достаточно одного

    if results:
        st.success(" Найдено изменение!")
        for cat, method, changed_dims, pred_label, pred_sublabel in results:
            st.markdown(f"""
**Метод:** `{method}`  
**Категория координат:** `{cat}`  
**Изменённые dim:** `{changed_dims[:5]}...`  
**Новый смысл:** `{pred_label.upper()} -> {pred_sublabel}`
""")
    else:
        st.warning(" Не удалось найти подходящее изменение. Попробуйте другую подкатегорию.")
