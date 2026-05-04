import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "rf_model.pkl"
FULL_DIM = 1536   # mean(768) + std(768) — формат эмбеддингов из notebook 02
HALF_DIM = FULL_DIM // 2


def load_model() -> dict:
    """Загружает модель. Возвращает dict с ключами 'model', 'feature_config', и др."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_embeddings(npy_path: Path) -> np.ndarray:
    """Загружает .npy файл. Ожидается вектор [1536] (mean+std) или [n_clips, 768]."""
    emb = np.load(npy_path)
    if emb.ndim == 1 and emb.shape[0] != FULL_DIM:
        raise ValueError(
            f"Ожидается вектор размером {FULL_DIM} (mean+std эмбеддинги), "
            f"получено {emb.shape[0]}. Убедитесь, что файлы созданы через notebook 02."
        )
    return emb


def predict(emb: np.ndarray, meta: dict) -> dict:
    """
    emb: [1536] — предвычисленный mean+std вектор одного видео.
    meta: словарь из load_model() с ключом 'model' (sklearn Pipeline).
    """
    pipeline = meta["model"]
    feature_config = meta.get("feature_config", "std_only")

    # Воспроизводим apply_features из notebook 01
    if feature_config == "std_only":
        features = emb[HALF_DIM:].reshape(1, -1)   # вторая половина = std
    elif feature_config == "mean_only":
        features = emb[:HALF_DIM].reshape(1, -1)
    else:
        features = emb.reshape(1, -1)               # mean_std — полный вектор

    proba = pipeline.predict_proba(features)[0]
    classes = pipeline.classes_
    # В датасете: deceptive=0 (ложь), truthful=1 (правда)
    lie_proba = float(proba[classes == 0]) if 0 in classes else float(proba[0])

    mean_part = emb[:HALF_DIM]
    std_part = emb[HALF_DIM:]

    return {
        "lie_probability": lie_proba,
        "truth_probability": 1.0 - lie_proba,
        "prediction": "Ложь" if lie_proba > 0.5 else "Правда",
        "mean_embedding": mean_part,
        "std_embedding": std_part,
    }
