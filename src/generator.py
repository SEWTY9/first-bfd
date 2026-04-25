"""
Генератор синтетического датасета для задачи детекции ботоферм.

Моделируем соцсеть, где часть пользователей — реальные люди, а часть — боты,
собранные в фермы по 20-50 аккаунтов. Боты подписываются на "финансовые пирамиды"
координированно: в узком временном окне, с общих IP-подсетей, с шаблонными
комментариями и кольцевыми подписками внутри фермы.

Таблицы на выходе (data/raw/):
    users.csv     - профили (включая ground truth: is_bot, bot_cluster_id)
    actions.csv   - действия (subscribe / like / comment)
    comments.csv  - тексты комментариев (подмножество actions с action_type='comment')
    follows.csv   - подписки пользователей друг на друга
    pyramids.csv  - справочник подозрительных сообществ

Запуск:
    python -m src.generator
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Windows-консоль по умолчанию cp1252 — форсируем UTF-8, иначе print с кириллицей падает.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ---------- конфигурация ----------

@dataclass
class Config:
    seed: int = 42

    # объёмы
    n_normal_users: int = 2000
    n_bot_clusters: int = 5
    bot_cluster_size_range: tuple[int, int] = (20, 50)
    n_pyramids: int = 10
    n_normal_communities: int = 200  # обычные сообщества, не пирамиды

    # временное окно наблюдения
    observation_start: datetime = datetime(2026, 1, 1)
    observation_end: datetime = datetime(2026, 3, 31)

    # нормальные пользователи
    normal_registration_min: datetime = datetime(2023, 1, 1)
    normal_actions_mean: float = 40.0       # среднее число действий в окне
    normal_pyramid_subscribe_prob: float = 0.1  # ~10% нормальных тоже подпишутся на какую-то пирамиду (шум)
    normal_follow_count_mean: float = 15.0

    # боты
    bot_registration_window_days: int = 14  # все боты фермы регистрируются в узком окне
    bot_actions_per_user_range: tuple[int, int] = (30, 80)
    bot_sync_window_minutes: int = 5        # окно синхронной атаки на пирамиду
    bot_ring_follow_prob: float = 0.7       # вероятность кольцевой подписки внутри фермы
    bot_ua_variants_per_cluster: int = 3    # из скольких UA выбирает ферма
    bot_ip_subnets_per_cluster: int = 2     # число /24-подсетей на ферму
    bot_comment_templates_per_cluster: int = 4

    # общие
    output_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------- шаблоны для бот-комментариев ----------

BOT_COMMENT_TEMPLATES = [
    "Отличный проект, уже заработал {amount} рублей за {days} дней! Советую всем {emoji}",
    "Вот это тема! Зашел сюда, вывел {amount} на карту. Кто еще в деле?",
    "Ребят, проверено лично — платят. Вложил {amount}, получил х{mult} за неделю!",
    "Не верил сначала, но {amount} уже на руках. Спасибо команде {emoji}",
    "Лучшая возможность для заработка в {year}. Уже {days} дней в проекте!",
    "Залетайте пока не поздно, я вывел {amount} без проблем",
    "Друзья, это реально работает, {amount} — мой результат за {days} дней",
    "Админы молодцы, проект топ, {amount} вывел, жду следующую выплату",
]

EMOJIS = ["🔥", "💰", "🚀", "💎", "✅", "👍", "🤑"]


# ---------- вспомогательные функции ----------

def _random_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _random_ip_in_subnet(subnet_prefix: str, rng: random.Random) -> str:
    """subnet_prefix вида '185.220.101' — возвращаем IP из этой /24."""
    return f"{subnet_prefix}.{rng.randint(1, 254)}"


def _random_subnet_24(rng: random.Random) -> str:
    return f"{rng.randint(10, 230)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}"


def _random_datetime(start: datetime, end: datetime, rng: random.Random) -> datetime:
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=rng.uniform(0, delta))


def _fill_bot_comment(template: str, rng: random.Random) -> str:
    return template.format(
        amount=rng.choice([50_000, 75_000, 100_000, 150_000, 200_000]),
        days=rng.choice([3, 5, 7, 10, 14]),
        mult=rng.choice([2, 3, 5]),
        emoji=rng.choice(EMOJIS),
        year=2026,
    )


# ---------- генератор ----------

class DatasetGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        self.faker = Faker("ru_RU")
        Faker.seed(cfg.seed)

        self.users: list[dict] = []
        self.actions: list[dict] = []
        self.comments: list[dict] = []
        self.follows: list[dict] = []
        self.pyramids: list[dict] = []

        self._next_user_id = 1
        self._next_action_id = 1

    # ---- справочники ----

    def _build_pyramids(self) -> list[int]:
        ids = []
        for _ in range(self.cfg.n_pyramids):
            pid = 900_000 + len(ids)
            self.pyramids.append({
                "target_id": pid,
                "name": f"Пирамида_{self.faker.company()}",
                "is_pyramid": True,
            })
            ids.append(pid)
        return ids

    def _build_normal_communities(self) -> list[int]:
        return list(range(100_000, 100_000 + self.cfg.n_normal_communities))

    # ---- пользователи ----

    def _new_user_id(self) -> int:
        uid = self._next_user_id
        self._next_user_id += 1
        return uid

    def _add_normal_user(self) -> int:
        uid = self._new_user_id()
        reg_date = _random_datetime(
            self.cfg.normal_registration_min,
            self.cfg.observation_start,
            self.rng,
        )
        self.users.append({
            "user_id": uid,
            "registration_date": reg_date,
            "profile_completeness": round(self.np_rng.beta(5, 2), 3),  # смещено к полным
            "has_avatar": int(self.rng.random() < 0.85),
            "is_bot": 0,
            "bot_cluster_id": -1,
        })
        return uid

    def _add_bot_user(self, cluster_id: int, cluster_reg_start: datetime) -> int:
        uid = self._new_user_id()
        reg_date = cluster_reg_start + timedelta(
            seconds=self.rng.uniform(0, self.cfg.bot_registration_window_days * 86400)
        )
        self.users.append({
            "user_id": uid,
            "registration_date": reg_date,
            "profile_completeness": round(self.np_rng.beta(1.5, 5), 3),  # смещено к пустым
            "has_avatar": int(self.rng.random() < 0.2),
            "is_bot": 1,
            "bot_cluster_id": cluster_id,
        })
        return uid

    # ---- действия ----

    def _new_action_id(self) -> int:
        aid = self._next_action_id
        self._next_action_id += 1
        return aid

    def _emit_action(self, user_id: int, action_type: str, target_id: int,
                     ts: datetime, ip: str, user_agent: str,
                     comment_text: str | None = None) -> None:
        aid = self._new_action_id()
        self.actions.append({
            "action_id": aid,
            "user_id": user_id,
            "action_type": action_type,
            "target_id": target_id,
            "ts": ts,
            "ip": ip,
            "user_agent": user_agent,
        })
        if action_type == "comment" and comment_text is not None:
            self.comments.append({
                "action_id": aid,
                "user_id": user_id,
                "target_id": target_id,
                "ts": ts,
                "text": comment_text,
            })

    # ---- нормальные юзеры ----

    def _generate_normal_activity(self, uid: int, targets_normal: list[int],
                                  targets_pyramids: list[int]) -> None:
        n_actions = max(1, int(self.np_rng.poisson(self.cfg.normal_actions_mean)))
        # пуассоновские интервалы
        total_seconds = (self.cfg.observation_end - self.cfg.observation_start).total_seconds()
        timestamps = sorted(
            self.cfg.observation_start + timedelta(seconds=self.rng.uniform(0, total_seconds))
            for _ in range(n_actions)
        )
        ua = self.faker.user_agent()  # один UA на сессию-ишь (упрощаем: один на юзера в окне)
        # примерно фиксированный IP с небольшим разбросом (разные сессии)
        base_ip_prefix = f"{self.rng.randint(10, 230)}.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}"

        subscribe_to_pyramid = self.rng.random() < self.cfg.normal_pyramid_subscribe_prob
        pyramid_target = self.rng.choice(targets_pyramids) if subscribe_to_pyramid else None

        for ts in timestamps:
            ip = _random_ip_in_subnet(base_ip_prefix, self.rng)
            action_type = self.rng.choices(
                ["like", "comment", "subscribe"], weights=[0.7, 0.2, 0.1]
            )[0]
            if action_type == "subscribe" and pyramid_target is not None and self.rng.random() < 0.3:
                target = pyramid_target
            else:
                target = self.rng.choice(targets_normal)

            text = None
            if action_type == "comment":
                text = self.faker.sentence(nb_words=self.rng.randint(4, 15))
            self._emit_action(uid, action_type, target, ts, ip, ua, text)

    def _generate_normal_follows(self, all_user_ids: list[int]) -> None:
        for user in self.users:
            if user["is_bot"]:
                continue
            uid = user["user_id"]
            n_follows = max(0, int(self.np_rng.poisson(self.cfg.normal_follow_count_mean)))
            followees = self.rng.sample(all_user_ids, min(n_follows, len(all_user_ids) - 1))
            for f in followees:
                if f == uid:
                    continue
                ts = _random_datetime(
                    user["registration_date"], self.cfg.observation_end, self.rng
                )
                self.follows.append({
                    "follower_id": uid, "followed_id": f, "ts": ts,
                })

    # ---- боты ----

    def _generate_bot_cluster(self, cluster_id: int, targets_pyramids: list[int],
                              targets_normal: list[int]) -> list[int]:
        size = self.rng.randint(*self.cfg.bot_cluster_size_range)
        cluster_reg_start = self.cfg.observation_start - timedelta(
            days=self.rng.randint(0, 10)
        )
        bot_ids = [self._add_bot_user(cluster_id, cluster_reg_start) for _ in range(size)]

        # ресурсы фермы
        subnets = [_random_subnet_24(self.rng) for _ in range(self.cfg.bot_ip_subnets_per_cluster)]
        uas = [self.faker.user_agent() for _ in range(self.cfg.bot_ua_variants_per_cluster)]
        templates = self.rng.sample(
            BOT_COMMENT_TEMPLATES,
            min(self.cfg.bot_comment_templates_per_cluster, len(BOT_COMMENT_TEMPLATES)),
        )
        # на какую пирамиду атакуемся (может быть 1-2)
        target_pyramids = self.rng.sample(targets_pyramids, k=self.rng.randint(1, 2))

        # ---- синхронная атака: все боты подписываются на пирамиду в узком окне ----
        for pyramid_id in target_pyramids:
            attack_start = _random_datetime(
                self.cfg.observation_start, self.cfg.observation_end, self.rng
            )
            window_sec = self.cfg.bot_sync_window_minutes * 60
            for bid in bot_ids:
                ts = attack_start + timedelta(seconds=self.rng.uniform(0, window_sec))
                ip = _random_ip_in_subnet(self.rng.choice(subnets), self.rng)
                ua = self.rng.choice(uas)
                self._emit_action(bid, "subscribe", pyramid_id, ts, ip, ua)

                # сразу лайкают и оставляют шаблонный коммент
                ts_like = ts + timedelta(seconds=self.rng.uniform(5, 60))
                self._emit_action(bid, "like", pyramid_id, ts_like, ip, ua)

                ts_comment = ts_like + timedelta(seconds=self.rng.uniform(5, 120))
                text = _fill_bot_comment(self.rng.choice(templates), self.rng)
                self._emit_action(bid, "comment", pyramid_id, ts_comment, ip, ua, text)

        # ---- фоновая активность бота (с подозрительно ровными интервалами) ----
        for bid in bot_ids:
            n_actions = self.rng.randint(*self.cfg.bot_actions_per_user_range)
            # боты идут по ровному расписанию: базовый интервал + малый шум
            base_interval = self.rng.uniform(600, 3600)  # 10 мин - 1 час
            jitter = base_interval * 0.05               # всего 5% шума
            start_ts = _random_datetime(
                self.cfg.observation_start, self.cfg.observation_end, self.rng
            )
            ts = start_ts
            for _ in range(n_actions):
                ts += timedelta(seconds=max(1, self.np_rng.normal(base_interval, jitter)))
                if ts > self.cfg.observation_end:
                    break
                ip = _random_ip_in_subnet(self.rng.choice(subnets), self.rng)
                ua = self.rng.choice(uas)
                action_type = self.rng.choices(
                    ["like", "comment", "subscribe"], weights=[0.6, 0.3, 0.1]
                )[0]
                target = self.rng.choice(targets_normal)
                text = None
                if action_type == "comment":
                    text = _fill_bot_comment(self.rng.choice(templates), self.rng)
                self._emit_action(bid, action_type, target, ts, ip, ua, text)

        # ---- кольцевые подписки внутри фермы ----
        for i, a in enumerate(bot_ids):
            for b in bot_ids:
                if a == b:
                    continue
                if self.rng.random() < self.cfg.bot_ring_follow_prob:
                    ts = _random_datetime(
                        self.cfg.observation_start, self.cfg.observation_end, self.rng
                    )
                    self.follows.append({
                        "follower_id": a, "followed_id": b, "ts": ts,
                    })

        return bot_ids

    # ---- главный пайплайн ----

    def generate(self) -> None:
        targets_pyramids = self._build_pyramids()
        targets_normal = self._build_normal_communities()

        print(f"[gen] Генерирую {self.cfg.n_normal_users} нормальных юзеров...")
        normal_ids = [self._add_normal_user() for _ in range(self.cfg.n_normal_users)]
        for uid in normal_ids:
            self._generate_normal_activity(uid, targets_normal, targets_pyramids)

        print(f"[gen] Генерирую {self.cfg.n_bot_clusters} бот-кластеров...")
        bot_ids_all = []
        for cluster_id in range(self.cfg.n_bot_clusters):
            bot_ids_all.extend(
                self._generate_bot_cluster(cluster_id, targets_pyramids, targets_normal)
            )

        print("[gen] Генерирую подписки нормальных юзеров...")
        all_ids = normal_ids + bot_ids_all
        self._generate_normal_follows(all_ids)

        self._save()

    def _save(self) -> None:
        out = self.cfg.output_dir
        out.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.users).to_csv(out / "users.csv", index=False)
        pd.DataFrame(self.actions).to_csv(out / "actions.csv", index=False)
        pd.DataFrame(self.comments).to_csv(out / "comments.csv", index=False)
        pd.DataFrame(self.follows).to_csv(out / "follows.csv", index=False)
        pd.DataFrame(self.pyramids).to_csv(out / "pyramids.csv", index=False)

        print(f"[gen] Сохранено в {out.resolve()}")
        print(f"       users:    {len(self.users):>7}")
        print(f"       actions:  {len(self.actions):>7}")
        print(f"       comments: {len(self.comments):>7}")
        print(f"       follows:  {len(self.follows):>7}")
        print(f"       pyramids: {len(self.pyramids):>7}")
        bots = sum(u["is_bot"] for u in self.users)
        print(f"       из них ботов: {bots} ({bots / len(self.users):.1%})")


def main() -> None:
    cfg = Config()
    DatasetGenerator(cfg).generate()


if __name__ == "__main__":
    main()
