# Детекция недостоверного поведения по мимике лица с использованием MARLIN

Выпускная квалификационная работа бакалавра — Пиняева Екатерина, МАИ, 2026

## О работе

Исследование применимости видеотрансформерной архитектуры **MARLIN** (Masked Autoencoder for Facial video Representation LearnINg) к задаче бинарной классификации правдивого и ложного поведения по мимике лица.

**Датасет:** Real-Life Trial Dataset — 121 видеозапись судебных заседаний, 54 субъекта  
**Лучший результат:** MARLIN vit_base + Random Forest + std_only → **Balanced Accuracy = 0.668**  
**Схема оценки:** 5-fold StratifiedGroupKFold с разделением по субъектам

---

## Структура репозитория

```
├── notebooks/
│   ├── 01_marlin_classification.ipynb     # Основной эксперимент: полный перебор конфигураций
│   ├── 02_marlin_embeddings_extraction.ipynb  # Извлечение эмбеддингов через MARLIN
│   ├── 03_lora_adaptation.ipynb           # LoRA-адаптация MARLIN vit_base (Kaggle)
│   ├── 04_dataset_overview.ipynb          # Анализ датасета и предобработка
│   ├── 05_classification_results.ipynb    # Визуализация результатов классификации
│   └── 06_embedding_analysis.ipynb        # Анализ признакового пространства (t-SNE)
│
├── results/
│   ├── marlin_vit_base_results.csv        # Все конфигурации MARLIN vit_base (72 комбинации)
│   ├── marlin_vit_large_results.csv       # Результаты MARLIN vit_large
│   ├── videomae_base_results.csv          # Результаты VideoMAE-base
│   ├── general_encoders_results.csv       # VideoMAE-large, ViViT, TimeSformer
│   └── all_encoders_comparison.csv        # Сводная таблица всех энкодеров
│
└── figures/
    ├── fig_01_class_distribution.png      # Распределение клипов по субъектам
    ├── fig_02_aggregation_comparison.png  # Сравнение стратегий агрегирования
    ├── fig_05_fold_stability.png          # Разброс BA по 5 фолдам (box plot)
    ├── fig_06_split_comparison.png        # Наивное vs корректное разделение данных
    ├── fig_07_marlin_progress.png         # Сравнение конфигураций MARLIN
    ├── fig_08_loss_curves.png             # Кривые обучения MLP-классификатора
    ├── fig_face_vs_general.png            # Лицевые vs общие видеоэнкодеры
    └── fig_all_models_full.png            # Сводное сравнение всех подходов
```

---

## Ключевые результаты

| Модель | Balanced Accuracy | AUC | F1-macro |
|--------|:-----------------:|:---:|:--------:|
| **MARLIN vit_base + RF + std_only** | **0.668** | 0.628 | 0.625 |
| MARLIN vit_base + MLP | 0.662 | 0.643 | 0.617 |
| VideoMAE-large + LR | 0.636 | 0.685 | 0.604 |
| MARLIN vit_large + SVM | 0.620 | 0.676 | 0.583 |
| MARLIN vit_base + LoRA | 0.584 | 0.521 | 0.493 |
| ViViT + SVM | 0.572 | 0.535 | 0.512 |
| TimeSformer + LR | 0.550 | 0.434 | 0.503 |
| MARLIN vit_small + RF | 0.487 | 0.544 | 0.487 |

## Основные выводы

- **Лицевая специализация важна:** MARLIN vit_base (0.668) превосходит лучший общецелевой энкодер VideoMAE-large (0.636)
- **Динамика информативнее среднего:** стратегия `std_only` (стандартное отклонение эмбеддингов по клипам) даёт BA = 0.622 против 0.478 у `mean_only`
- **Корректное разделение критично:** наивный random split завышает результат до 0.736 (+7 п.п.) из-за утечки по идентичности субъектов
- **LoRA на малых данных не работает:** при ~88 обучающих примерах на фолд градиентная адаптация нестабильна (std = 0.177)

## Воспроизводимость

Все эксперименты воспроизводимы: `random_state=42` зафиксирован везде. Эмбеддинги предварительно вычислены и сохранены — ноутбук `01` запускается без GPU поверх готовых `.npy` файлов.
