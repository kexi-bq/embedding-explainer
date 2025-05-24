from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class MeaningClassifier:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.label_encoder = LabelEncoder()
        self.sublabel_encoders = {}
        self.label_model = RandomForestClassifier()
        self.sublabel_models = {}

    def predict_by_embedding(self, embedding):
        import numpy as np

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        label_idx = self.label_model.predict(embedding)[0]
        label = self.label_encoder.inverse_transform([label_idx])[0]

        if label in self.sublabel_models:
            sublabel_model = self.sublabel_models[label]
            sublabel_encoder = self.sublabel_encoders[label]
            sublabel_idx = sublabel_model.predict(embedding)[0]
            sublabel = sublabel_encoder.inverse_transform([sublabel_idx])[0]
        else:
            sublabel = "unknown"

        return label, sublabel


    def fit(self, texts, labels, sublabels):
        # Эмбеддинги
        self.texts = texts
        embeddings = self.model.encode(texts)

        # Основной label
        y_label = self.label_encoder.fit_transform(labels)
        self.label_model.fit(embeddings, y_label)

        # Подмодели sublabel
        label_groups = {}
        for text, label, sub in zip(texts, labels, sublabels):
            label_groups.setdefault(label, []).append((text, sub))

        for label, group in label_groups.items():
            group_texts, group_sublabels = zip(*group)
            group_embeddings = self.model.encode(group_texts)

            encoder = LabelEncoder()
            y_sub = encoder.fit_transform(group_sublabels)
            clf = RandomForestClassifier()
            clf.fit(group_embeddings, y_sub)

            self.sublabel_encoders[label] = encoder
            self.sublabel_models[label] = clf

        return embeddings


    def get_top_coordinates_by_class(self, texts, labels, top_n=5):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        embeddings = self.model.encode(texts)
        label_set = list(set(labels))
        result = {}

        for target_label in label_set:
            y_bin = np.array([1 if l == target_label else 0 for l in labels])
            clf = RandomForestClassifier()
            clf.fit(embeddings, y_bin)
            importances = clf.feature_importances_
            top_dims = np.argsort(importances)[-top_n:][::-1]
            result[target_label] = [(dim, importances[dim]) for dim in top_dims]

        return result


    def predict(self, text):
        embedding = self.model.encode([text])
        label_idx = self.label_model.predict(embedding)[0]
        label = self.label_encoder.inverse_transform([label_idx])[0]

        if label in self.sublabel_models:
            sublabel_model = self.sublabel_models[label]
            sublabel_encoder = self.sublabel_encoders[label]
            sublabel_idx = sublabel_model.predict(embedding)[0]
            sublabel = sublabel_encoder.inverse_transform([sublabel_idx])[0]
        else:
            sublabel = "unknown"

        return label, sublabel
