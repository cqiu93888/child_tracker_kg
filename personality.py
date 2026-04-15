# 個性推論：依互動 JSON 對每位幼兒貼標籤（代理指標，非 CBQ／觀察量表）
#
# 說明（供報告／論文方法段參考）：
# - 文獻中的「抑制／外放、社交整合」多以問卷（如 CBQ）、實驗室或訓練觀察員編碼測量；
#   本模組僅能從影片追蹤得到的「同框次數、框距離靠近次數、互動對象數」做 **proxy**。
# - 標籤採 **全班內相對分位**：約最低／最高各 PERSONALITY_EXTREME_GROUP_FRAC 比例者，
#   概念上接近常見「極端組約 15%、中間多數」的分組精神，但閾值隨你這批資料變動。
# - 請勿宣稱等同任何特定論文的操作型定義。
from __future__ import annotations

import json
from collections import defaultdict
from typing import Set, Tuple

from config import (
    interactions_file,
    PERSONALITY_EXTREME_GROUP_FRAC,
    PERSONALITY_MIN_PEOPLE,
    PERSONALITY_NEAR_CO_RATIO_PCT,
    PERSONALITY_SOCIAL_PARTNER_PCT,
)


def load_interactions(path: str | None = None):
    p = path or interactions_file()
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _percentile_threshold(sorted_vals: list, pct: float) -> float:
    """pct 為 0–100；回傳 >= 該百分位的最小值（線性內插）。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    t = k - lo
    return float(sorted_vals[lo] * (1 - t) + sorted_vals[hi] * t)


def _extreme_group_masks(values: dict) -> Tuple[Set[str], Set[str]]:
    """
    依分數由低到高，最低與最高各約 EXTREME_GROUP_FRAC 比例的人（至少各 1 人）。
    回傳 (low_names, high_names)。僅 1 人時不劃極端組（避免同時落入低/高）。
    """
    items = sorted(values.items(), key=lambda x: x[1])
    n = len(items)
    if n <= 1:
        return set(), set()
    k = max(1, int(round(n * float(PERSONALITY_EXTREME_GROUP_FRAC))))
    if k * 2 > n:
        k = max(1, n // 2)
    low = set(name for name, _ in items[:k])
    high = set(name for name, _ in items[-k:])
    return low, high


def _gather_stats(interactions: dict):
    """從 interactions 整理每人：同框加總、靠近加總、互動對象集合。"""
    cooccurrence = interactions.get("cooccurrence", {})
    near_count = interactions.get("near_count", {})

    person_total_co = defaultdict(int)
    person_total_near = defaultdict(int)
    person_partners = defaultdict(set)

    for key, count in cooccurrence.items():
        parts = key.split(",", 1)
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not a or not b:
            continue
        person_total_co[a] += count
        person_total_co[b] += count
        person_partners[a].add(b)
        person_partners[b].add(a)

    for key, count in near_count.items():
        parts = key.split(",", 1)
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not a or not b:
            continue
        person_total_near[a] += count
        person_total_near[b] += count

    all_names = set(person_total_co.keys()) | set(person_total_near.keys())
    return person_total_co, person_total_near, person_partners, all_names


def _infer_simple_thresholds(all_names, person_total_co, person_total_near, person_partners):
    """人數過少時沿用較直覺的全班極值比例（與舊版相容）。"""
    max_co = max(person_total_co.values()) if person_total_co else 0
    max_near = max(person_total_near.values()) if person_total_near else 0
    max_partners = max(len(person_partners[p]) for p in all_names) if all_names else 0

    result = {}
    for name in all_names:
        labels = []
        co = person_total_co.get(name, 0)
        near = person_total_near.get(name, 0)
        partners = len(person_partners.get(name, set()))

        if partners >= 2 and max_partners and partners >= max_partners * 0.7:
            labels.append("社交型")
        if co > 0 and near >= co * 0.5 and max_near and near >= max_near * 0.5:
            labels.append("親密型")
        if max_co and co >= max_co * 0.6:
            labels.append("活躍型")
        if partners <= 1 and co < (max_co or 1) * 0.3:
            labels.append("內向型")
        if not labels:
            labels.append("觀察中")
        result[name] = labels[:3]
    return result


def infer_personality(interactions: dict) -> dict:
    """
    依互動資料推論每位幼兒的標籤列表（最多 3 個）。
    優先使用全班分位；人數不足時退回簡化規則。
    """
    if not interactions:
        return {}

    person_total_co, person_total_near, person_partners, all_names = _gather_stats(interactions)
    if not all_names:
        return {}

    n = len(all_names)
    if n < int(PERSONALITY_MIN_PEOPLE):
        return _infer_simple_thresholds(
            all_names, person_total_co, person_total_near, person_partners
        )

    # 整合度：同框 + 靠近加權（與 knowledge_graph 邊權重精神一致）
    integration = {}
    for name in all_names:
        co = person_total_co.get(name, 0)
        near = person_total_near.get(name, 0)
        integration[name] = float(co) + 0.5 * float(near)

    partners_count = {name: len(person_partners.get(name, set())) for name in all_names}

    low_int, high_int = _extreme_group_masks(integration)
    low_p, high_p = _extreme_group_masks(partners_count)

    # 靠近/同框 比：僅對有同框者計算，避免除以零
    ratios = []
    ratio_by_name = {}
    for name in all_names:
        co = person_total_co.get(name, 0)
        near = person_total_near.get(name, 0)
        if co > 0:
            r = float(near) / float(co)
            ratio_by_name[name] = r
            ratios.append(r)
    ratios.sort()
    thr_near_co = _percentile_threshold(ratios, float(PERSONALITY_NEAR_CO_RATIO_PCT))

    sorted_partners = sorted(partners_count.values())
    thr_partners = _percentile_threshold(
        sorted_partners, float(PERSONALITY_SOCIAL_PARTNER_PCT)
    )

    co_list = sorted(person_total_co.get(nm, 0) for nm in all_names)
    pc_list = sorted(partners_count[nm] for nm in all_names)
    median_co = _percentile_threshold(co_list, 50)
    median_pc = _percentile_threshold(pc_list, 50)

    result = {}
    for name in all_names:
        labels = []
        co = person_total_co.get(name, 0)
        pc = partners_count.get(name, 0)
        r = ratio_by_name.get(name)

        # 1) 低整合（類文獻「較少社交整合」）— 優先，利於著色與解讀
        if name in low_int and pc <= median_pc and co <= median_co:
            labels.append("內向型")
        # 2) 高整合（類「高參與／外放」敘事之 proxy）
        elif name in high_int:
            labels.append("活躍型")

        # 3) 互動對象多（類「對象多元／社交廣」）— 排除極低整合避免矛盾
        if name not in low_int and pc >= thr_partners and pc >= 2:
            if "社交型" not in labels:
                labels.append("社交型")

        # 4) 身體距離常近（靠近相對於同框特別多）
        if r is not None and r >= thr_near_co and co > 0:
            if "親密型" not in labels:
                labels.append("親密型")

        if not labels:
            labels.append("觀察中")

        result[name] = labels[:3]

    return result
