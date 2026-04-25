"""
Детекция ботоферм двумя независимыми методами:

    1) DBSCAN по поведенческим фичам (per-user) — ловит юзеров, чьи признаки
       плотно сидят в "подозрительной" области пространства.
    2) Louvain community detection на объединённом графе (подписки + общие
       IP-подсети) — ловит координированные кластеры через структуру связей.

Финальное решение: пользователь — кандидат в боты, если он попал хотя бы в одну
"подозрительную" группу. Это снижает recall-потери при сохранении precision.

Запуск:
    python -m src.detection
"""
from __future__ import annotations

import sys
from pathlib import Path

import community as community_louvain  # пакет python-louvain импортируется как `community`
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"


# ---------- DBSCAN по фичам ----------

DBSCAN_FEATURES = [
    "profile_completeness",
    "has_avatar",
    "account_age_days",
    "pyramid_action_share",
    "interval_median",
    "interval_p10",
    "min_window_5_sec",
    "duplicate_text_share",
    "mean_text_neighbours",
    "mutual_follow_share",
]


def _log_clip(x: pd.Series) -> pd.Series:
    """log1p со страховкой от отрицательных и NaN."""
    return np.log1p(x.clip(lower=0).fillna(0))


def prepare_dbscan_matrix(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    df = features[DBSCAN_FEATURES].copy()
    # лог-трансформ для скошенных (всё что в секундах или больших числах)
    for col in ["account_age_days", "interval_median", "interval_p10", "min_window_5_sec",
                "mean_text_neighbours"]:
        df[col] = _log_clip(df[col])
    df = df.fillna(0)
    X = StandardScaler().fit_transform(df.values)
    return X, list(df.columns)


def run_dbscan(features: pd.DataFrame, eps: float = 1.2, min_samples: int = 8) -> pd.Series:
    X, _ = prepare_dbscan_matrix(features)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return pd.Series(labels, index=features.index, name="dbscan_label")


def label_suspicious_dbscan_clusters(dbscan_labels: pd.Series,
                                     min_size: int = 10,
                                     max_size: int = 100) -> pd.Series:
    """
    DBSCAN метит ВСЕ плотные группы, включая гигантский кластер "норма".
    Подозрительными считаем только малые кластеры (фермы по 20-50 человек),
    остальное — нормальная масса или шум.
    """
    sizes = dbscan_labels[dbscan_labels >= 0].value_counts()
    suspicious_clusters = sizes[(sizes >= min_size) & (sizes <= max_size)].index.tolist()
    return dbscan_labels.isin(suspicious_clusters).astype(int).rename("in_dbscan_cluster")


# ---------- Louvain на графе ----------

def build_graph(follows: pd.DataFrame, actions: pd.DataFrame,
                shared_subnet_min_users: int = 5) -> nx.Graph:
    """
    Граф взаимодействий:
    - ребро = подписка (направление игнорируем, но взаимные считаются плотнее)
    - ребро = два юзера хотя бы раз заходили с одной /24 подсети
      (только если подсетью пользуется не больше N юзеров — иначе это
      публичный прокси / NAT; бесполезно)
    """
    g = nx.Graph()

    # подписки
    for follower, followed in zip(follows["follower_id"], follows["followed_id"]):
        if g.has_edge(follower, followed):
            g[follower][followed]["weight"] += 1
        else:
            g.add_edge(follower, followed, weight=1)

    # общие подсети
    actions = actions.copy()
    actions["subnet"] = actions["ip"].str.rsplit(".", n=1).str[0]
    user_subnet = actions.groupby("subnet")["user_id"].agg(set)
    user_subnet = user_subnet[user_subnet.map(len) <= shared_subnet_min_users * 20]
    # парные рёбра внутри малых подсетей
    for subnet, users in user_subnet.items():
        if 2 <= len(users) <= shared_subnet_min_users * 20:
            users = list(users)
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    a, b = users[i], users[j]
                    if g.has_edge(a, b):
                        g[a][b]["weight"] += 1
                    else:
                        g.add_edge(a, b, weight=1)
    return g


def run_louvain(graph: nx.Graph, seed: int = 42) -> pd.Series:
    partition = community_louvain.best_partition(graph, weight="weight", random_state=seed)
    return pd.Series(partition, name="louvain_community").rename_axis("user_id")


# ---------- сборка ----------

def detect(features: pd.DataFrame, follows: pd.DataFrame, actions: pd.DataFrame,
           dbscan_eps: float = 1.2, dbscan_min_samples: int = 8) -> pd.DataFrame:
    print("[detect] DBSCAN по фичам...")
    dbscan_labels = run_dbscan(features, eps=dbscan_eps, min_samples=dbscan_min_samples)

    print("[detect] Строю граф (подписки + общие /24 подсети)...")
    graph = build_graph(follows, actions)
    print(f"           узлов: {graph.number_of_nodes()}, рёбер: {graph.number_of_edges()}")

    print("[detect] Louvain community detection...")
    louvain_labels = run_louvain(graph)

    # размер сообществ — для последующей фильтрации "подозрительных"
    community_sizes = louvain_labels.value_counts()
    louvain_size = louvain_labels.map(community_sizes)

    # подозрительные сообщества: маленькие (10..70 пользователей) и со средним
    # mutual_follow_share выше порога
    feats_with_community = features.join(
        louvain_labels.rename("louvain_community"), how="left"
    ).join(
        louvain_size.rename("louvain_community_size"), how="left"
    )
    feats_with_community["louvain_community"] = feats_with_community["louvain_community"].fillna(-1).astype(int)
    feats_with_community["louvain_community_size"] = feats_with_community["louvain_community_size"].fillna(1).astype(int)

    suspicious_communities = (
        feats_with_community[
            (feats_with_community["louvain_community"] >= 0)
            & (feats_with_community["louvain_community_size"].between(10, 80))
        ]
        .groupby("louvain_community")
        .agg(
            avg_mutual_follow=("mutual_follow_share", "mean"),
            avg_duplicate_text=("duplicate_text_share", "mean"),
            size=("louvain_community", "size"),
        )
        .query("avg_mutual_follow > 0.3 or avg_duplicate_text > 0.5")
        .index.tolist()
    )

    feats_with_community["dbscan_label"] = dbscan_labels.reindex(feats_with_community.index, fill_value=-1)
    feats_with_community["in_dbscan_cluster"] = label_suspicious_dbscan_clusters(
        feats_with_community["dbscan_label"]
    )
    feats_with_community["in_suspicious_community"] = feats_with_community["louvain_community"].isin(suspicious_communities).astype(int)
    feats_with_community["bot_candidate"] = (
        (feats_with_community["in_dbscan_cluster"] == 1)
        | (feats_with_community["in_suspicious_community"] == 1)
    ).astype(int)

    print(f"[detect] Подозрительных Louvain-сообществ: {len(suspicious_communities)}")
    print(f"[detect] Кандидатов в боты: {feats_with_community['bot_candidate'].sum()} "
          f"из {len(feats_with_community)}")
    return feats_with_community


def main() -> None:
    features = pd.read_csv(PROCESSED_DIR / "features.csv", index_col="user_id")
    actions = pd.read_csv(RAW_DIR / "actions.csv", parse_dates=["ts"])
    follows = pd.read_csv(RAW_DIR / "follows.csv", parse_dates=["ts"])

    result = detect(features, follows, actions)
    out_path = PROCESSED_DIR / "predictions.csv"
    result.to_csv(out_path)
    print(f"[detect] Сохранено: {out_path}")


if __name__ == "__main__":
    main()
