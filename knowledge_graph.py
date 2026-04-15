# 知識圖譜：幼兒節點、屬性（個性）、關係邊（朋友、常一起）
import os
import json
import networkx as nx

from config import (
    graph_dir,
    interactions_file,
    MIN_COOCCURRENCE_FOR_FRIEND,
    EGO_GRAPH_MIN_COOCCURRENCE,
    EGO_GRAPH_MIN_EDGE_WEIGHT,
    EGO_GRAPH_MAX_NEIGHBORS,
    ensure_dirs,
)
from personality import infer_personality
from face_registry import get_unique_registered_names


def _filter_interaction_timeline_for_nodes(timeline: dict, node_ids: set) -> dict:
    """僅保留圖上節點的互動時間序列，縮小 ego HTML 體積。"""
    if not timeline or not isinstance(timeline, dict) or not node_ids:
        return timeline
    ps = timeline.get("person_series") or {}
    filt = {k: v for k, v in ps.items() if k in node_ids}
    if not filt:
        return None
    out = dict(timeline)
    out["person_series"] = filt
    return out


def build_knowledge_graph(interactions_path: str = None, output_path: str = None) -> dict:
    """
    建立知識圖譜：
    - 節點：每個已註冊的幼兒（含僅獨自出現者），屬性含個性標籤
    - 邊：兩人「朋友」關係（同框次數 >= MIN_COOCCURRENCE_FOR_FRIEND），權重為同框+靠近
    回傳圖的資料結構，並可寫入 JSON。
    """
    ensure_dirs()
    interactions_path = interactions_path or interactions_file()
    if output_path is None:
        output_path = os.path.join(graph_dir(), "knowledge_graph.json")

    if os.path.isfile(interactions_path):
        with open(interactions_path, "r", encoding="utf-8") as f:
            interactions = json.load(f)
    else:
        interactions = None
    personality = infer_personality(interactions) if interactions else {}

    # 先納入所有已註冊的幼兒，確保即使只獨自出現也會在圖譜中
    try:
        registered_names = get_unique_registered_names()
        all_names = set(registered_names) if registered_names else set()
    except Exception:
        all_names = set()

    if interactions:
        cooccurrence = interactions.get("cooccurrence", {})
        near_count = interactions.get("near_count", {})
        for key in cooccurrence:
            parts = key.split(",", 1)
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                all_names.add(a)
                all_names.add(b)
        for key in near_count:
            parts = key.split(",", 1)
            if len(parts) != 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a and b:
                all_names.add(a)
                all_names.add(b)
    else:
        cooccurrence = {}
        near_count = {}

    if not all_names:
        tl_empty = interactions.get("interaction_timeline") if interactions else None
        data = {
            "nodes": [],
            "edges": [],
            "personality": personality,
            "interaction_timeline": tl_empty,
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    G = nx.Graph()
    # 節點：所有幼兒，屬性含個性
    for name in all_names:
        G.add_node(
            name,
            personality=personality.get(name, ["觀察中"]),
            label=name,
        )

    # 邊：朋友關係（同框次數達門檻）
    for key, co in cooccurrence.items():
        if co < MIN_COOCCURRENCE_FOR_FRIEND:
            continue
        parts = key.split(",", 1)
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if a not in all_names or b not in all_names:
            continue
        near = near_count.get(key, 0)
        weight = co + near * 0.5  # 靠近加分
        G.add_edge(a, b, weight=weight, cooccurrence=co, near=near, relation="朋友")

    # 匯出為可視化與前端用的結構
    nodes = []
    for n in G.nodes():
        nodes.append({
            "id": n,
            "label": n,
            "personality": G.nodes[n].get("personality", []),
        })
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": d.get("weight", 1),
            "cooccurrence": d.get("cooccurrence", 0),
            "relation": d.get("relation", "互動"),
        })

    data = {
        "nodes": nodes,
        "edges": edges,
        "personality": personality,
        "interaction_timeline": interactions.get("interaction_timeline")
        if interactions
        else None,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def ego_knowledge_graph(
    kg_data: dict,
    focal_id: str,
    min_cooccurrence: int = None,
    min_edge_weight: float = None,
    max_neighbors: int = None,
) -> dict:
    """
    以 focal_id 為中心之子圖：
    - 僅保留與中心相連、且同框 >= min_cooccurrence、權重 >= min_edge_weight 的邊
    - 若 max_neighbors 為整數，依權重由高到低最多保留這麼多「同伴節點」（中心永遠保留）
    - 中心節點若在 kg_data.nodes 中不存在，回傳僅含空邊的單節點（個性觀察中）
    """
    min_cooccurrence = (
        EGO_GRAPH_MIN_COOCCURRENCE
        if min_cooccurrence is None
        else min_cooccurrence
    )
    if min_edge_weight is None:
        min_edge_weight = EGO_GRAPH_MIN_EDGE_WEIGHT
    if max_neighbors is None:
        max_neighbors = EGO_GRAPH_MAX_NEIGHBORS

    node_map = {n["id"]: n for n in kg_data.get("nodes", [])}
    focal = focal_id
    if focal not in node_map:
        tl0 = _filter_interaction_timeline_for_nodes(
            kg_data.get("interaction_timeline"), {focal}
        )
        return {
            "nodes": [
                {
                    "id": focal,
                    "label": focal,
                    "personality": ["觀察中"],
                }
            ],
            "edges": [],
            "personality": kg_data.get("personality", {}),
            "focal_id": focal,
            "interaction_timeline": tl0,
        }

    candidates = []
    for e in kg_data.get("edges", []):
        u, v = e.get("source"), e.get("target")
        if u != focal and v != focal:
            continue
        co = e.get("cooccurrence", 0)
        w = float(e.get("weight", 0))
        if co < min_cooccurrence or w < min_edge_weight:
            continue
        peer = v if u == focal else u
        candidates.append((w, co, peer, e))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

    chosen_peers = []
    seen = set()
    for w, co, peer, e in candidates:
        if peer in seen:
            continue
        if max_neighbors is not None and len(chosen_peers) >= max_neighbors:
            break
        seen.add(peer)
        chosen_peers.append((peer, e))

    peer_ids = {p for p, _ in chosen_peers}
    nodes_out = [node_map[focal]]
    for pid in sorted(peer_ids):
        if pid in node_map:
            nodes_out.append(node_map[pid])

    edges_out = [dict(e) for _, e in chosen_peers]

    node_ids = {n["id"] for n in nodes_out}
    tl_ego = _filter_interaction_timeline_for_nodes(
        kg_data.get("interaction_timeline"), node_ids
    )

    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "personality": kg_data.get("personality", {}),
        "focal_id": focal,
        "interaction_timeline": tl_ego,
    }
