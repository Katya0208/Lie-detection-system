import zipfile
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import load_model, load_embeddings, predict
from visualizer import plot_gauge, plot_embedding_profile, plot_std_distribution, plot_summary

st.set_page_config(
    page_title="Детекция лжи по мимике",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Система детекции недостоверного поведения")
st.caption("Анализ мимики лица на основе MARLIN-эмбеддингов | ВКР Пиняева Е., МАИ 2026")


@st.cache_resource(show_spinner="Загрузка модели...")
def get_model():
    return load_model()


meta = get_model()
pipeline_info = meta.get("model")

with st.sidebar:
    st.header("О системе")
    st.info(
        "Приложение принимает ZIP-архив с файлами эмбеддингов (`.npy`).\n\n"
        "Каждый `.npy` файл — агрегированные эмбеддинги одного видео, "
        "shape: `[1536]` (mean + std по клипам).\n\n"
        "Классификатор использует **std-агрегацию** — "
        "лучшую стратегию по результатам экспериментов (BA = 0.668)."
    )
    st.markdown("---")
    st.markdown(f"**Конфигурация модели:** `{meta.get('feature_config', '—')}`")
    st.markdown(f"**Классификатор:** `{meta.get('classifier', '—')}`")
    st.markdown(f"**Balanced Accuracy:** `{meta.get('balanced_accuracy', '—')}`")
    st.markdown(f"**AUC:** `{meta.get('auc', '—')}`")
    st.markdown(f"**Обучено на:** `{meta.get('n_train_samples', '—')}` видео")
    st.markdown("---")
    st.markdown("**Порог классификации:** 0.5")

uploaded_file = st.file_uploader(
    "Загрузите ZIP-архив с .npy файлами эмбеддингов",
    type=["zip"],
)

if uploaded_file is None:
    st.markdown(
        """
        ### Как подготовить архив

        1. Запустите notebook `02_marlin_embeddings_extraction.ipynb` для получения эмбеддингов
        2. Каждое видео сохраняется как `<название>.npy` — вектор `[1536]` (mean + std по клипам)
        3. Упакуйте все `.npy` файлы в `.zip`-архив и загрузите выше
        """
    )
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    with zipfile.ZipFile(uploaded_file) as zf:
        npy_names = [
            f for f in zf.namelist()
            if f.endswith(".npy")
            and not f.endswith("/")
            and not Path(f).name.startswith("._")
            and "__MACOSX" not in f
        ]
        if not npy_names:
            st.error("В архиве не найдено .npy файлов.")
            st.stop()
        zf.extractall(tmpdir)

    st.success(f"Загружено файлов: **{len(npy_names)}**")

    results = []
    errors = []
    progress_bar = st.progress(0, text="Обработка файлов...")

    for i, npy_name in enumerate(npy_names):
        npy_path = tmpdir / npy_name
        try:
            emb = load_embeddings(npy_path)
            result = predict(emb, meta)
            p = Path(npy_name)
            result["name"] = str(p.parent / p.stem) if p.parent != Path(".") else p.stem
            results.append(result)
        except Exception as e:
            errors.append(f"`{npy_name}`: {e}")
        progress_bar.progress((i + 1) / len(npy_names), text=f"Обработка: {Path(npy_name).stem}")

    progress_bar.empty()

    if errors:
        with st.expander(f"Ошибки при обработке ({len(errors)})", expanded=False):
            for err in errors:
                st.warning(err)

    if not results:
        st.error("Ни один файл не удалось обработать.")
        st.stop()

    # --- Summary ---
    st.header("Сводные результаты")

    col1, col2, col3 = st.columns(3)
    n_lie = sum(1 for r in results if r["lie_probability"] > 0.5)
    col1.metric("Всего видео", len(results))
    col2.metric("Классифицировано как ложь", n_lie)
    col3.metric("Классифицировано как правда", len(results) - n_lie)

    if len(results) > 1:
        st.plotly_chart(plot_summary(results), use_container_width=True)

    # --- Per-video details ---
    st.header("Детальный анализ")

    for result in sorted(results, key=lambda r: r["lie_probability"], reverse=True):
        label = (
            f"{'🔴' if result['lie_probability'] > 0.5 else '🟢'} "
            f"{result['name']} — {result['prediction']} ({result['lie_probability']:.1%})"
        )
        with st.expander(label, expanded=len(results) == 1):
            name_key = result["name"].replace("/", "_")
            col_gauge, col_dist = st.columns([1, 2])
            with col_gauge:
                st.plotly_chart(plot_gauge(result["lie_probability"]), use_container_width=True, key=f"gauge_{name_key}")
            with col_dist:
                st.plotly_chart(
                    plot_std_distribution(result["std_embedding"]),
                    use_container_width=True,
                    key=f"dist_{name_key}",
                )
            st.plotly_chart(
                plot_embedding_profile(result["mean_embedding"], result["std_embedding"]),
                use_container_width=True,
                key=f"profile_{name_key}",
            )
