# === capsule_embedding.py ===
# Добавим 2 слоя по оси Z к каждому dim (прототип)

import numpy as np

class CapsuleEmbedding:
    def __init__(self, base_embedding, z_layers=2):
        self.base = np.array(base_embedding)
        self.z_layers = z_layers
        self.capsules = self.expand_capsules()

    def expand_capsules(self):
        # На каждый dim добавим z_layers "внутренних аспектов"
        return np.stack([self.generate_layers(val) for val in self.base])

    def generate_layers(self, base_val):
        # Простейшее: создаём слой из base_val с модификаторами
        return np.array([
            base_val * 1.0,           # исходное значение (смысловая сила)
            np.tanh(base_val)         # преобразованное значение (например, эмоциональный фильтр)
        ])

    def get_capsule_matrix(self):
        # Вернуть (384, z_layers) матрицу
        return self.capsules

    def to_flat_embedding(self):
        # Схлопываем вектор обратно в (384,) путём усреднения или другого правила
        return self.capsules.mean(axis=1)
