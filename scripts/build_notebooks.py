"""
Программная сборка четырёх ноутбуков проекта.
Запуск: python scripts/build_notebooks.py
Затем:  jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

KERNELSPEC = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
}


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def save(nb: nbf.NotebookNode, name: str) -> None:
    nb.metadata = KERNELSPEC["metadata"]
    path = NOTEBOOKS_DIR / name
    nbf.write(nb, str(path))
    print(f"  -> {path}")


# =====================================================================
# 01_eda.ipynb
# =====================================================================

def build_eda() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 01 — EDA: знакомство с данными\n\n"
            "Цель этого ноутбука — посмотреть на сырые данные глазами антифрод-аналитика и "
            "**подсветить аномалии**, на которых дальше будем строить признаки.\n\n"
            "Пять секций под пять признаков из разбора кейса:\n"
            "1. Профили: возраст аккаунта, заполненность, аватар.\n"
            "2. Тайминги: распределение действий по времени, синхронные всплески подписок на пирамиды.\n"
            "3. Регулярность: интервалы между действиями (и почему наивный CV нас обманывает).\n"
            "4. Тексты: как одинаковые комментарии расходятся по сети.\n"
            "5. Связи: подписки, кольца внутри ферм.\n\n"
            "В конце — список фич, которые перенесём в `src/features.py`."
        ),
        code(
            "import sys\n"
            "sys.path.append('..')\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "sns.set_theme(style='whitegrid')\n"
            "pd.options.display.float_format = '{:.3f}'.format\n"
        ),
        code(
            "users = pd.read_csv('../data/raw/users.csv', parse_dates=['registration_date'])\n"
            "actions = pd.read_csv('../data/raw/actions.csv', parse_dates=['ts'])\n"
            "comments = pd.read_csv('../data/raw/comments.csv', parse_dates=['ts'])\n"
            "follows = pd.read_csv('../data/raw/follows.csv', parse_dates=['ts'])\n"
            "pyramids = pd.read_csv('../data/raw/pyramids.csv')\n"
            "\n"
            "print('users   ', users.shape)\n"
            "print('actions ', actions.shape)\n"
            "print('comments', comments.shape)\n"
            "print('follows ', follows.shape)\n"
            "print('pyramids', pyramids.shape)\n"
            "print('доля ботов в users:', users.is_bot.mean())\n"
        ),
        md(
            "## 1. Профили\n\n"
            "В реальной задаче `is_bot` нам недоступен, но в синтетике — есть, и я использую его "
            "**только для визуализации** (чтобы было понятно, что мы вообще ловим). На обучении ниже "
            "ground truth не уйдёт."
        ),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
            "sns.histplot(data=users, x='profile_completeness', hue='is_bot', bins=30, ax=axes[0], stat='density', common_norm=False)\n"
            "axes[0].set_title('Заполненность профиля')\n"
            "users['account_age_days'] = (pd.Timestamp('2026-03-31') - users.registration_date).dt.days\n"
            "sns.histplot(data=users, x='account_age_days', hue='is_bot', bins=40, ax=axes[1], stat='density', common_norm=False)\n"
            "axes[1].set_title('Возраст аккаунта на конец окна, дней')\n"
            "users.groupby('is_bot')['has_avatar'].mean().plot.bar(ax=axes[2])\n"
            "axes[2].set_title('Доля с аватаром')\n"
            "axes[2].set_ylabel('share')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "**Что видно:** боты живут <100 дней, с пустыми профилями и без аватаров. "
            "Это не one-shot-сигнал (новички существуют), но в комбо с поведением — мощный фильтр."
        ),
        md(
            "## 2. Тайминги: всплески подписок на пирамиды\n\n"
            "Атака фермы выглядит как вертикальный скачок на CDF подписок. На обычных сообществах подписки "
            "приходят равномерно (~линейный наклон во времени)."
        ),
        code(
            "pyramid_ids = set(pyramids.target_id)\n"
            "subs = actions[(actions.action_type=='subscribe')].copy()\n"
            "subs['is_pyramid'] = subs.target_id.isin(pyramid_ids)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
            "for pid, grp in subs[subs.is_pyramid].groupby('target_id'):\n"
            "    grp_sorted = grp.sort_values('ts')\n"
            "    axes[0].plot(grp_sorted.ts, np.arange(1, len(grp_sorted)+1), label=str(pid))\n"
            "axes[0].set_title('CDF подписок на каждую пирамиду')\n"
            "axes[0].set_ylabel('накопленное число подписок')\n"
            "axes[0].legend(fontsize=7, ncol=2)\n"
            "\n"
            "# сравнение с обычным сообществом (выберем самое популярное)\n"
            "popular_normal = subs[~subs.is_pyramid].target_id.value_counts().head(5).index\n"
            "for tid in popular_normal:\n"
            "    grp_sorted = subs[subs.target_id == tid].sort_values('ts')\n"
            "    axes[1].plot(grp_sorted.ts, np.arange(1, len(grp_sorted)+1), label=str(tid))\n"
            "axes[1].set_title('CDF подписок на топ-5 обычных сообществ')\n"
            "axes[1].legend(fontsize=7, ncol=2)\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "На пирамидах видны **вертикальные ступени** — ферма пришла залпом за минуту-две. "
            "На обычных сообществах рост плавный."
        ),
        code(
            "# таблично: для каждой пирамиды найдём самое тесное окно из 10 подряд подписок\n"
            "rows = []\n"
            "for pid, grp in subs[subs.is_pyramid].groupby('target_id'):\n"
            "    g = grp.sort_values('ts').reset_index(drop=True)\n"
            "    if len(g) < 10: continue\n"
            "    deltas = g.ts.iloc[9:].values - g.ts.iloc[:-9].values\n"
            "    min_win_min = pd.to_timedelta(deltas.min()).total_seconds() / 60\n"
            "    rows.append({'pyramid': pid, 'subscribers': len(g), 'min_window_10_min': round(min_win_min, 2)})\n"
            "pd.DataFrame(rows).sort_values('min_window_10_min')\n"
        ),
        md(
            "## 3. Интервалы между действиями: парадокс CV\n\n"
            "Гипотеза из устного ответа: у ботов интервалы **слишком ровные**, значит CV (std/mean) низкий. "
            "Проверим, и сразу удивимся."
        ),
        code(
            "actions_sorted = actions.sort_values(['user_id','ts'])\n"
            "actions_sorted['interval'] = actions_sorted.groupby('user_id')['ts'].diff().dt.total_seconds()\n"
            "\n"
            "user_cv = actions_sorted.dropna(subset=['interval']).groupby('user_id')['interval'].agg(\n"
            "    interval_mean='mean', interval_std='std', interval_median='median', n='size'\n"
            ").assign(cv=lambda d: d.interval_std / d.interval_mean.clip(lower=1))\n"
            "user_cv = user_cv.join(users.set_index('user_id')[['is_bot']])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n"
            "sns.kdeplot(data=user_cv, x='cv', hue='is_bot', fill=True, common_norm=False, clip=(0, 15), ax=axes[0])\n"
            "axes[0].set_title('CV интервалов: боты vs норма')\n"
            "axes[0].axvline(1, color='red', ls='--', alpha=0.4)\n"
            "axes[0].text(1.02, 0.05, 'CV=1\\n(пуассон)', color='red', transform=axes[0].get_xaxis_transform())\n"
            "\n"
            "sns.kdeplot(data=user_cv, x='interval_median', hue='is_bot', fill=True, common_norm=False, log_scale=True, ax=axes[1])\n"
            "axes[1].set_title('Медиана интервала, сек (лог-шкала)')\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "print(user_cv.groupby('is_bot')[['interval_mean','interval_median','cv']].median())\n"
        ),
        md(
            "**Парадокс:** у ботов CV получился **выше** нормы (≈6 против ≈1). Почему?\n\n"
            "Бот делает миксованную активность: фоновое расписание (раз в 10-60 минут, ровные интервалы) "
            "плюс **залп из 3 действий за 1 минуту** во время атаки на пирамиду. "
            "Небольшие интервалы залпа на фоне больших фоновых ломают равенство std≈mean.\n\n"
            "Что брать в фичи вместо CV:\n"
            "- `interval_median` — у ботов 2000 сек vs 130000 у нормы (видно на правом графике).\n"
            "- `interval_p10` — нижний хвост распределения, ловит залпы.\n"
            "- `min_window_5_sec` — минимальное окно из 5 действий подряд."
        ),
        md(
            "## 4. Тексты: одинаковые комменты у разных пользователей\n\n"
            "Считаем точные совпадения. В `features.py` мы добавим MinHash-фуззи поиск, но даже "
            "точные дубли уже всё показывают."
        ),
        code(
            "comments_with_label = comments.merge(users[['user_id','is_bot']], on='user_id')\n"
            "text_user_count = comments_with_label.groupby('text')['user_id'].nunique().sort_values(ascending=False)\n"
            "print('Топ повторяющихся текстов (число РАЗНЫХ юзеров, написавших одно и то же):')\n"
            "print(text_user_count.head(15).to_string())\n"
            "print()\n"
            "popular = text_user_count[text_user_count >= 5].index\n"
            "shared_users = comments_with_label[comments_with_label.text.isin(popular)]\n"
            "print('Доля is_bot среди авторов часто повторяющихся текстов:',\n"
            "      shared_users.is_bot.mean())\n"
        ),
        md(
            "## 5. Граф подписок: плотные сообщества\n\n"
            "Считаем долю взаимных подписок. У реального юзера это редкое явление "
            "(подписка обычно асимметрична), у фермы — почти всегда."
        ),
        code(
            "edges = set(zip(follows.follower_id, follows.followed_id))\n"
            "follows2 = follows.copy()\n"
            "follows2['is_mutual'] = [(b,a) in edges for a,b in zip(follows2.follower_id, follows2.followed_id)]\n"
            "mutual = follows2.groupby('follower_id')['is_mutual'].mean().rename('mutual_share')\n"
            "mutual = mutual.to_frame().join(users.set_index('user_id')[['is_bot']])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "sns.histplot(data=mutual, x='mutual_share', hue='is_bot', stat='density', common_norm=False, bins=30, ax=ax)\n"
            "ax.set_title('Доля взаимных подписок у пользователя')\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "print(mutual.groupby('is_bot')['mutual_share'].describe())\n"
        ),
        md(
            "## Итог: что забираем в фичи\n\n"
            "| Признак | Файл | Смысл |\n"
            "|---|---|---|\n"
            "| profile_completeness, has_avatar, account_age_days | profile | базовая статика |\n"
            "| n_actions, n_subscribes, n_comments_action | activity | объём |\n"
            "| pyramid_action_share | activity | связь с пирамидой |\n"
            "| interval_median, interval_p10, min_window_5_sec | activity | бёрсты вместо наивного CV |\n"
            "| n_unique_subnets, n_unique_uas, ip_entropy, ua_entropy | technical | концентрация устройств |\n"
            "| duplicate_text_share, mean_text_neighbours | text | MinHash-сходство комментов |\n"
            "| followers_count, following_count, mutual_follow_share | graph | кольцевые подписки |\n\n"
            "Считаются в `src/features.py` — переходим в `02_features.ipynb`."
        ),
    ]
    save(nb, "01_eda.ipynb")


# =====================================================================
# 02_features.ipynb
# =====================================================================

def build_features_nb() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 02 — Feature Engineering\n\n"
            "Запускаем `src.features.build_features()` и смотрим:\n"
            "- какие фичи получились\n"
            "- насколько они разделяют ботов от нормы\n"
            "- какие коррелируют между собой (избыточность)\n"
        ),
        code(
            "import sys\n"
            "sys.path.append('..')\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "sns.set_theme(style='whitegrid')\n"
            "\n"
            "from src.features import build_features\n"
        ),
        code(
            "feats = build_features()\n"
            "feats.shape\n"
        ),
        md("## Сводка по группам"),
        code(
            "feats.head()\n"
        ),
        md("## Сравнение средних: боты vs норма"),
        code(
            "bot_mean = feats[feats.is_bot==1].drop(columns=['is_bot','bot_cluster_id']).mean()\n"
            "norm_mean = feats[feats.is_bot==0].drop(columns=['is_bot','bot_cluster_id']).mean()\n"
            "cmp = pd.DataFrame({'bot': bot_mean, 'normal': norm_mean})\n"
            "cmp['ratio'] = cmp['bot'] / cmp['normal'].replace(0, np.nan)\n"
            "cmp.sort_values('ratio', key=lambda s: s.abs(), ascending=False)\n"
        ),
        md(
            "Самые разделяющие фичи (по отклонению ratio от 1):\n"
            "- `mutual_follow_share` — кольцевые подписки\n"
            "- `duplicate_text_share` — шаблонные комменты\n"
            "- `pyramid_action_share` — связь с пирамидами\n"
            "- `interval_median`, `min_window_5_sec` — бёрсты во времени\n"
        ),
        md("## Коррелограмма"),
        code(
            "feats_no_truth = feats.drop(columns=['is_bot','bot_cluster_id'])\n"
            "corr = feats_no_truth.corr()\n"
            "fig, ax = plt.subplots(figsize=(11, 9))\n"
            "sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax, annot=False, cbar_kws={'shrink':0.6})\n"
            "ax.set_title('Корреляции фич')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "Сильные корреляции — естественны: `n_comments` коррелирует с `mean_text_neighbours` "
            "(больше пишешь — больше совпадений), `n_actions` ~ `n_likes` ~ `n_comments_action`. "
            "В DBSCAN мы оставляем только дискриминирующие фичи (см. `DBSCAN_FEATURES`)."
        ),
        md("## 2D-проекция фич: видны ли фермы?"),
        code(
            "from sklearn.preprocessing import StandardScaler\n"
            "import umap\n"
            "from src.detection import DBSCAN_FEATURES, prepare_dbscan_matrix\n"
            "\n"
            "X, cols = prepare_dbscan_matrix(feats)\n"
            "embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8,6))\n"
            "is_bot = feats['is_bot'].values\n"
            "ax.scatter(embedding[is_bot==0,0], embedding[is_bot==0,1], s=4, alpha=0.3, label='норма', c='steelblue')\n"
            "ax.scatter(embedding[is_bot==1,0], embedding[is_bot==1,1], s=10, alpha=0.9, label='бот', c='crimson')\n"
            "ax.set_title('UMAP проекция признакового пространства')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "На UMAP-проекции боты должны сидеть отдельной структурой (или несколькими) от основной массы. "
            "Если фермы хорошо разделены — это будет несколько компактных пятен.\n\n"
            "Дальше: `03_detection.ipynb` — кластеризуем."
        ),
    ]
    save(nb, "02_features.ipynb")


# =====================================================================
# 03_detection.ipynb
# =====================================================================

def build_detection_nb() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 03 — Детекция: DBSCAN + Louvain\n\n"
            "Запускаем оба метода, сравниваем их по отдельности и в комбо."
        ),
        code(
            "import sys\n"
            "sys.path.append('..')\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import networkx as nx\n"
            "sns.set_theme(style='whitegrid')\n"
            "\n"
            "from src.detection import detect, build_graph, run_louvain\n"
            "\n"
            "feats = pd.read_csv('../data/processed/features.csv', index_col='user_id')\n"
            "actions = pd.read_csv('../data/raw/actions.csv', parse_dates=['ts'])\n"
            "follows = pd.read_csv('../data/raw/follows.csv', parse_dates=['ts'])\n"
        ),
        code(
            "predictions = detect(feats, follows, actions)\n"
            "predictions[['is_bot','bot_candidate','in_dbscan_cluster','in_suspicious_community']].head()\n"
        ),
        md("## DBSCAN: размеры кластеров"),
        code(
            "dbs = predictions['dbscan_label'].value_counts().sort_index()\n"
            "dbs.head(20)\n"
        ),
        md(
            "Гигантский кластер (label со многими тысячами) — это нормальная масса; маленькие "
            "кластеры (10..100 человек) — кандидаты в фермы. Метка -1 означает шум."
        ),
        md("## Louvain: распределение размеров сообществ"),
        code(
            "louv = predictions['louvain_community'].value_counts()\n"
            "fig, ax = plt.subplots(figsize=(8,4))\n"
            "louv.plot.hist(bins=40, ax=ax)\n"
            "ax.set_xlabel('размер сообщества')\n"
            "ax.set_title('Гистограмма размеров Louvain-сообществ')\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "print('Топ-10 крупнейших сообществ:')\n"
            "print(louv.head(10))\n"
            "\n"
            "print('\\nПодозрительные сообщества (in_suspicious_community=1):')\n"
            "susp = predictions[predictions.in_suspicious_community==1]\n"
            "print(susp.groupby('louvain_community').agg(\n"
            "    size=('louvain_community','size'),\n"
            "    bots=('is_bot','sum'),\n"
            "    bot_share=('is_bot','mean'),\n"
            "    avg_mutual=('mutual_follow_share','mean'),\n"
            "    avg_dup=('duplicate_text_share','mean'),\n"
            "))\n"
        ),
        md("## Визуализация графа: подграф вокруг подозрительных сообществ"),
        code(
            "graph = build_graph(follows, actions)\n"
            "susp_nodes = set(predictions[predictions.in_suspicious_community==1].index)\n"
            "# для читаемости берём только узлы из подозрительных и небольшой 1-hop кружок\n"
            "subgraph_nodes = set(susp_nodes)\n"
            "for n in list(susp_nodes)[:200]:\n"
            "    subgraph_nodes.update(list(graph.neighbors(n))[:5])\n"
            "subg = graph.subgraph(subgraph_nodes)\n"
            "\n"
            "node_color = ['crimson' if n in susp_nodes else 'lightgray' for n in subg.nodes]\n"
            "fig, ax = plt.subplots(figsize=(10,8))\n"
            "pos = nx.spring_layout(subg, seed=42, k=0.4)\n"
            "nx.draw_networkx_nodes(subg, pos, node_size=8, node_color=node_color, alpha=0.85, ax=ax)\n"
            "nx.draw_networkx_edges(subg, pos, alpha=0.15, width=0.4, ax=ax)\n"
            "ax.set_title('Подграф: красные = подозрительные (Louvain), серые = соседи')\n"
            "ax.axis('off')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "Видно несколько плотных красных сгустков — это и есть фермы. Серые точки — нормальные "
            "юзеры, попавшие в радиус по случайным подпискам."
        ),
        md("Дальше: оценка качества в `04_report.ipynb`."),
    ]
    save(nb, "03_detection.ipynb")


# =====================================================================
# 04_report.ipynb
# =====================================================================

def build_report_nb() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# 04 — Итоговый отчёт\n\n"
            "Задача: оценить, насколько хорошо мы поймали ботоферм против ground truth."
        ),
        code(
            "import sys\n"
            "sys.path.append('..')\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.metrics import confusion_matrix\n"
            "sns.set_theme(style='whitegrid')\n"
            "\n"
            "from src.evaluation import evaluate, cluster_recovery\n"
            "\n"
            "predictions = pd.read_csv('../data/processed/predictions.csv', index_col='user_id')\n"
        ),
        md("## Сводные метрики"),
        code(
            "metrics = evaluate(predictions)\n"
            "rows = []\n"
            "for method, m in metrics.items():\n"
            "    rows.append({'method': method, **m})\n"
            "summary = pd.DataFrame(rows).set_index('method')\n"
            "summary.round(3)\n"
        ),
        md("## Матрицы ошибок"),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
            "for ax, (method, col) in zip(axes, [('combined','bot_candidate'),\n"
            "                                     ('dbscan','in_dbscan_cluster'),\n"
            "                                     ('louvain','in_suspicious_community')]):\n"
            "    cm = confusion_matrix(predictions.is_bot, predictions[col])\n"
            "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n"
            "                xticklabels=['pred норма','pred бот'], yticklabels=['real норма','real бот'])\n"
            "    ax.set_title(method)\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md("## Покрытие ферм"),
        code(
            "cluster_recovery(predictions).style.background_gradient(subset=['recall_in_cluster','dominant_share'])\n"
        ),
        md(
            "## Выводы\n\n"
            "**1. Louvain победил DBSCAN с разгромным счётом.**\n"
            "На наших данных бот-сигнал в первую очередь *структурный* (кольцевые подписки, общие IP), "
            "а не *градиентный по фичам*. Louvain находит сообщества из плотных связей естественно, "
            "DBSCAN же требует, чтобы боты сидели плотным облаком в фиче-пространстве — а они сидят, "
            "но **рядом сидит и нормальная масса**, и DBSCAN либо сливает их в один кластер, либо ловит "
            "крошечные очаги.\n\n"
            "**2. Combined-метрика хуже Louvain в одиночку.**\n"
            "Объединение двух методов через OR подняло Recall до 1.0, но снизило Precision (≈0.86): "
            "DBSCAN добавил FP без новых TP. Урок: **ансамбль через OR не всегда хорош** — иногда "
            "слабый член только шумит.\n\n"
            "**3. Все 5 ферм идеально склеились в свои Louvain-сообщества** (`dominant_share=1.0`). "
            "В реальной задаче такого не будет — настоящие боты сильнее маскируются — но это показывает, "
            "что метод чувствителен к внутренней связности фермы (взаимные подписки, общие подсети).\n\n"
            "## Что улучшать\n\n"
            "- Сейчас сигнал слишком сильный: фермы 100% подписаны друг на друга. Нужно усложнить "
            "генератор (доля колец 30-50%) и посмотреть, где Louvain ломается.\n"
            "- Добавить semi-supervised: ручную разметку топ-10 подозрительных и обучить классификатор "
            "(catboost/xgb) на этих метках с фичами.\n"
            "- Вместо OR — мета-классификатор поверх предсказаний обоих методов.\n"
            "- На реальных данных вместо `is_bot` использовать **прокси-метку**: жалобы, баны "
            "модерации, исчезнувшие аккаунты — и валидировать всё на отложенной выборке.\n\n"
            "## Что показать на собеседовании\n\n"
            "1. Воспроизводимый pipeline: `python -m src.generator → features → detection → evaluation`.\n"
            "2. Один график на каждый признак из устного ответа (есть в `01_eda.ipynb`).\n"
            "3. Историю про **парадокс CV** — наивная гипотеза проверена и сломалась, "
            "проблема разобрана, замена найдена. Это лучший сигнал зрелости аналитика.\n"
            "4. Историю про **DBSCAN vs Louvain** — выбор метода обоснован эмпирически, а не по моде."
        ),
    ]
    save(nb, "04_report.ipynb")


def main() -> None:
    print("Собираю ноутбуки...")
    build_eda()
    build_features_nb()
    build_detection_nb()
    build_report_nb()
    print("Готово.")


if __name__ == "__main__":
    main()
