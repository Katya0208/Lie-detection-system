import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_gauge(lie_probability: float) -> go.Figure:
    if lie_probability < 0.4:
        color = "#2ecc71"
    elif lie_probability < 0.6:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(lie_probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": "Вероятность лжи", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#eafaf1"},
                {"range": [40, 60], "color": "#fef9e7"},
                {"range": [60, 100], "color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def plot_embedding_profile(mean_emb: np.ndarray, std_emb: np.ndarray, top_n: int = 40) -> go.Figure:
    """Топ-N измерений по std — именно они используются классификатором."""
    top_idx = np.argsort(std_emb)[-top_n:][::-1]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Вариативность мимики (std) — топ-{top_n} измерений",
            f"Средняя активность (mean) — те же измерения",
        ),
        vertical_spacing=0.28,
    )

    fig.add_trace(go.Bar(
        x=[f"{i}" for i in top_idx],
        y=std_emb[top_idx],
        marker_color="#9b59b6",
        name="std",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[f"{i}" for i in top_idx],
        y=mean_emb[top_idx],
        marker_color="#3498db",
        name="mean",
    ), row=2, col=1)

    fig.update_layout(
        height=480,
        showlegend=False,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    fig.update_xaxes(title_text="Индекс измерения", tickangle=-45)
    fig.update_yaxes(title_text="std", row=1, col=1)
    fig.update_yaxes(title_text="mean", row=2, col=1)
    return fig


def plot_std_distribution(std_emb: np.ndarray) -> go.Figure:
    """Гистограмма распределения std по всем 768 измерениям."""
    fig = go.Figure(go.Histogram(
        x=std_emb,
        nbinsx=50,
        marker_color="#9b59b6",
        opacity=0.8,
    ))
    fig.add_vline(
        x=float(std_emb.mean()),
        line_dash="dash",
        line_color="black",
        annotation_text=f"среднее: {std_emb.mean():.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Распределение вариативности по измерениям эмбеддинга",
        xaxis_title="Стандартное отклонение",
        yaxis_title="Количество измерений",
        height=300,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def plot_summary(results: list[dict]) -> go.Figure:
    names = [r["name"] for r in results]
    probs = [r["lie_probability"] for r in results]
    colors = ["#e74c3c" if p > 0.5 else "#2ecc71" for p in probs]
    labels = [f"{r['prediction']} ({p:.1%})" for r, p in zip(results, probs)]

    fig = go.Figure(go.Bar(
        x=probs,
        y=names,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="outside",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", line_width=1.5)
    fig.update_layout(
        title="Сводные результаты по всем видео",
        xaxis=dict(title="Вероятность лжи", range=[0, 1.15], tickformat=".0%"),
        yaxis_title="",
        height=max(300, 60 * len(results)),
        margin=dict(t=50, b=40, l=160, r=20),
    )
    return fig
