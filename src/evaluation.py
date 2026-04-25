"""
Метрики качества детекции против ground truth (`is_bot`, `bot_cluster_id`),
которые лежат в users.csv. На реальных данных такой разметки нет, но в
учебном проекте мы её создали в генераторе и используем ТОЛЬКО на этапе
оценки — не для обучения.

Считаем три уровня:
    1) Бинарная классификация: bot_candidate vs is_bot
       precision, recall, F1 + матрица ошибок.
    2) По методам отдельно: DBSCAN, Louvain — какой даёт что.
    3) Покрытие ферм: для каждого реального bot_cluster_id сколько % ботов
       мы поймали и попали ли они в одно сообщество Louvain.

Запуск:
    python -m src.evaluation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"


def evaluate(predictions: pd.DataFrame) -> dict:
    """predictions: DataFrame, индекс user_id, колонки is_bot, bot_candidate,
    in_dbscan_cluster, in_suspicious_community, bot_cluster_id, louvain_community."""
    y_true = predictions["is_bot"].astype(int).values
    results = {}

    for method, col in [
        ("combined", "bot_candidate"),
        ("dbscan_only", "in_dbscan_cluster"),
        ("louvain_only", "in_suspicious_community"),
    ]:
        y_pred = predictions[col].astype(int).values
        results[method] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
            "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
            "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
            "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        }

    return results


def cluster_recovery(predictions: pd.DataFrame) -> pd.DataFrame:
    """Для каждой реальной фермы (bot_cluster_id) считаем:
    - размер фермы
    - сколько ботов поймали (recall внутри фермы)
    - в скольки разных Louvain-сообществах оказались (1 = ферма целиком склеилась)"""
    bots = predictions[predictions["is_bot"] == 1].copy()
    rows = []
    for cluster_id, grp in bots.groupby("bot_cluster_id"):
        rows.append({
            "bot_cluster_id": cluster_id,
            "size": len(grp),
            "recall_in_cluster": grp["bot_candidate"].mean(),
            "n_louvain_communities": grp["louvain_community"].nunique(),
            "dominant_community": grp["louvain_community"].mode().iloc[0] if len(grp) else -1,
            "dominant_share": grp["louvain_community"].value_counts(normalize=True).iloc[0] if len(grp) else 0,
        })
    return pd.DataFrame(rows).sort_values("bot_cluster_id").reset_index(drop=True)


def print_report(predictions: pd.DataFrame) -> None:
    metrics = evaluate(predictions)
    print("=== Метрики бинарной классификации (vs is_bot) ===")
    rows = []
    for method, m in metrics.items():
        rows.append({
            "method": method,
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "f1": round(m["f1"], 3),
            "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("=== Покрытие ферм ===")
    print(cluster_recovery(predictions).to_string(index=False))


def main() -> None:
    predictions = pd.read_csv(PROCESSED_DIR / "predictions.csv", index_col="user_id")
    print_report(predictions)


if __name__ == "__main__":
    main()
