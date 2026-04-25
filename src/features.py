"""
Feature engineering: считаем признаки по каждому пользователю на основе его
активности, профиля, технических атрибутов и графа подписок.

Все фичи разбиты на 4 группы:
    1. Profile    — статика профиля (заполненность, аватар, возраст аккаунта).
    2. Activity   — поведение во времени (объём, регулярность, всплески,
                    связь с пирамидами).
    3. Technical  — IP / User-Agent: концентрация и энтропия.
    4. Text       — комментарии: повторы, шаблонность через MinHash.
    5. Graph      — подписки: in/out degree и доля взаимных.

Главная функция: build_features(...) -> pd.DataFrame, индекс = user_id.

Запуск:
    python -m src.features
    результат сохраняется в data/processed/features.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

OBSERVATION_END = pd.Timestamp("2026-03-31")  # для подсчёта возраста на момент окончания окна


# ---------- утилиты ----------

def _shannon_entropy(values: pd.Series) -> float:
    if len(values) == 0:
        return 0.0
    counts = values.value_counts(normalize=True).values
    return float(-(counts * np.log2(counts + 1e-12)).sum())


def _ip_to_subnet24(ip: str) -> str:
    return ip.rsplit(".", 1)[0]


# ---------- группа 1: профиль ----------

def profile_features(users: pd.DataFrame) -> pd.DataFrame:
    df = users[["user_id", "profile_completeness", "has_avatar"]].copy()
    df["account_age_days"] = (
        OBSERVATION_END - pd.to_datetime(users["registration_date"])
    ).dt.days.clip(lower=0)
    return df.set_index("user_id")


# ---------- группа 2: активность ----------

def activity_features(actions: pd.DataFrame, pyramid_ids: set[int]) -> pd.DataFrame:
    a = actions.sort_values(["user_id", "ts"]).copy()
    a["ts_int"] = a["ts"].astype("int64") // 10**9

    # интервалы между соседними действиями пользователя (групповая разница)
    a["prev_ts"] = a.groupby("user_id")["ts_int"].shift(1)
    a["interval"] = (a["ts_int"] - a["prev_ts"]).astype("Float64")

    grouped = a.groupby("user_id")
    feats = pd.DataFrame(index=grouped.size().index)
    feats["n_actions"] = grouped.size()
    feats["n_subscribes"] = grouped["action_type"].apply(lambda s: (s == "subscribe").sum())
    feats["n_likes"] = grouped["action_type"].apply(lambda s: (s == "like").sum())
    feats["n_comments_action"] = grouped["action_type"].apply(lambda s: (s == "comment").sum())

    # связь с пирамидами
    is_pyramid = a["target_id"].isin(pyramid_ids)
    feats["n_pyramid_actions"] = a[is_pyramid].groupby("user_id").size().reindex(feats.index, fill_value=0)
    feats["pyramid_action_share"] = feats["n_pyramid_actions"] / feats["n_actions"].clip(lower=1)

    # интервалы: статистики
    intervals_by_user = a.dropna(subset=["interval"]).groupby("user_id")["interval"]
    feats["interval_mean"] = intervals_by_user.mean().reindex(feats.index)
    feats["interval_median"] = intervals_by_user.median().reindex(feats.index)
    feats["interval_std"] = intervals_by_user.std().reindex(feats.index)
    feats["interval_cv"] = feats["interval_std"] / feats["interval_mean"].clip(lower=1e-6)

    # бёрсты: 10-й перцентиль интервала — у ботов быстрые залпы
    feats["interval_p10"] = intervals_by_user.quantile(0.1).reindex(feats.index)
    # минимальное окно из 5 действий подряд (sec)
    def min_window_5(group: pd.Series) -> float:
        ts = group.sort_values().to_numpy()
        if len(ts) < 5:
            return np.nan
        return float((ts[4:] - ts[:-4]).min())
    feats["min_window_5_sec"] = grouped["ts_int"].apply(min_window_5)

    return feats


# ---------- группа 3: технические сигналы ----------

def technical_features(actions: pd.DataFrame) -> pd.DataFrame:
    a = actions.copy()
    a["subnet_24"] = a["ip"].map(_ip_to_subnet24)
    grouped = a.groupby("user_id")
    feats = pd.DataFrame(index=grouped.size().index)
    feats["n_unique_ips"] = grouped["ip"].nunique()
    feats["n_unique_subnets"] = grouped["subnet_24"].nunique()
    feats["n_unique_uas"] = grouped["user_agent"].nunique()
    feats["ip_entropy"] = grouped["ip"].apply(_shannon_entropy)
    feats["ua_entropy"] = grouped["user_agent"].apply(_shannon_entropy)
    return feats


# ---------- группа 4: тексты комментариев ----------

def _text_to_minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # шинглы по 3 слова, токенизация без знаков
    tokens = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    for i in range(max(0, len(tokens) - 2)):
        shingle = " ".join(tokens[i:i + 3]).encode()
        m.update(shingle)
    return m


def text_features(comments: pd.DataFrame, num_perm: int = 64,
                  similarity_threshold: float = 0.7) -> pd.DataFrame:
    """
    Для каждого комментария считаем MinHash, ищем "близкие двойники" среди
    остальных через LSH. На уровне пользователя агрегируем:
    - n_comments
    - n_distinct_texts
    - duplicate_text_share — доля комментов, у которых есть >=N близких двойников
    - mean_neighbours — среднее число "близких" комментов
    """
    if len(comments) == 0:
        return pd.DataFrame(columns=[
            "n_comments", "n_distinct_texts", "duplicate_text_share", "mean_text_neighbours"
        ])

    c = comments.reset_index(drop=True).copy()
    c["row_idx"] = c.index

    minhashes = [_text_to_minhash(t, num_perm=num_perm) for t in c["text"].astype(str)]
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
    for i, m in enumerate(minhashes):
        lsh.insert(str(i), m)

    neighbours = []
    for i, m in enumerate(minhashes):
        result = lsh.query(m)
        # сам комментарий тоже находится — вычитаем 1
        neighbours.append(max(0, len(result) - 1))
    c["n_neighbours"] = neighbours
    c["has_duplicate"] = c["n_neighbours"] > 0

    grouped = c.groupby("user_id")
    feats = pd.DataFrame(index=grouped.size().index)
    feats["n_comments"] = grouped.size()
    feats["n_distinct_texts"] = grouped["text"].nunique()
    feats["duplicate_text_share"] = grouped["has_duplicate"].mean()
    feats["mean_text_neighbours"] = grouped["n_neighbours"].mean()
    return feats


# ---------- группа 5: граф подписок ----------

def graph_features(follows: pd.DataFrame) -> pd.DataFrame:
    out_deg = follows.groupby("follower_id").size().rename("following_count")
    in_deg = follows.groupby("followed_id").size().rename("followers_count")

    # взаимные подписки: edge (a,b) встречается и (b,a)
    pairs = set(zip(follows["follower_id"], follows["followed_id"]))
    follows = follows.copy()
    follows["is_mutual"] = [
        (b, a) in pairs for a, b in zip(follows["follower_id"], follows["followed_id"])
    ]
    mutual_share = follows.groupby("follower_id")["is_mutual"].mean().rename("mutual_follow_share")

    feats = pd.concat([out_deg, in_deg, mutual_share], axis=1).fillna(0)
    feats.index.name = "user_id"
    return feats


# ---------- сборка ----------

def build_features() -> pd.DataFrame:
    print("[features] Загружаю сырые данные...")
    users = pd.read_csv(RAW_DIR / "users.csv", parse_dates=["registration_date"])
    actions = pd.read_csv(RAW_DIR / "actions.csv", parse_dates=["ts"])
    comments = pd.read_csv(RAW_DIR / "comments.csv", parse_dates=["ts"])
    follows = pd.read_csv(RAW_DIR / "follows.csv", parse_dates=["ts"])
    pyramids = pd.read_csv(RAW_DIR / "pyramids.csv")
    pyramid_ids = set(pyramids["target_id"])

    print("[features] Профильные...")
    f_profile = profile_features(users)
    print("[features] Активностные...")
    f_activity = activity_features(actions, pyramid_ids)
    print("[features] Технические...")
    f_tech = technical_features(actions)
    print("[features] Текстовые (MinHash, может занять минуту)...")
    f_text = text_features(comments)
    print("[features] Графовые...")
    f_graph = graph_features(follows)

    feats = (
        f_profile
        .join(f_activity, how="left")
        .join(f_tech, how="left")
        .join(f_text, how="left")
        .join(f_graph, how="left")
    )
    feats = feats.fillna(0)

    # ground truth — оставляем рядом, но в обучении НЕ используем
    truth = users.set_index("user_id")[["is_bot", "bot_cluster_id"]]
    feats = feats.join(truth, how="left")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "features.csv"
    feats.to_csv(out_path)
    print(f"[features] Сохранено: {out_path}  shape={feats.shape}")
    return feats


def main() -> None:
    feats = build_features()
    print()
    print("=== Сравнение средних: боты vs норма ===")
    bot_means = feats[feats["is_bot"] == 1].drop(columns=["is_bot", "bot_cluster_id"]).mean()
    norm_means = feats[feats["is_bot"] == 0].drop(columns=["is_bot", "bot_cluster_id"]).mean()
    cmp = pd.DataFrame({"bot": bot_means, "normal": norm_means})
    cmp["ratio_bot/norm"] = cmp["bot"] / cmp["normal"].replace(0, np.nan)
    print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
