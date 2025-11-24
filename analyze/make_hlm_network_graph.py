# -*- coding: utf-8 -*-
"""
make_hlm_network_graph.py

根據 hlm_network_nodes.csv / hlm_network_edges.csv 畫一張人物共現網路圖：
- 節點：贾宝玉、林黛玉、薛宝钗、王熙凤、贾母
- 節點大小：按 weighted_degree 放大
- 邊粗細：按共現次數 weight 放大
輸出：results/hlm_network_graph.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ==== 全局字體設定：讓中文不變方框 ====
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 負號正常顯示

# ==== 基本路徑 ====
BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results")

nodes_path = os.path.join(RESULTS_DIR, "hlm_network_nodes.csv")
edges_path = os.path.join(RESULTS_DIR, "hlm_network_edges.csv")

print("[INFO] 讀取節點與邊文件…")
nodes = pd.read_csv(nodes_path, encoding="utf-8")
edges = pd.read_csv(edges_path, encoding="utf-8")

# ==== 構建 NetworkX 圖 ====
G = nx.Graph()

for _, row in nodes.iterrows():
    cid = row["char_id"]
    label = row["label"]
    wdeg = row.get("weighted_degree", 0.0)
    G.add_node(cid, label=label, weighted_degree=float(wdeg))

for _, row in edges.iterrows():
    a = row["char_a"]
    b = row["char_b"]
    w = float(row["weight"])
    G.add_edge(a, b, weight=w)

print(f"[INFO] 圖中節點數：{G.number_of_nodes()}，邊數：{G.number_of_edges()}")

# ==== 佈局：彈簧佈局（spring layout） ====
pos = nx.spring_layout(G, k=1.0, weight="weight", seed=42)  # seed 保證每次位置一致

# ==== 節點大小：按 weighted_degree 放大 ====
weighted_degrees = [G.nodes[n].get("weighted_degree", 0.0) for n in G.nodes()]
if len(weighted_degrees) > 0:
    w_min = min(weighted_degrees)
    w_max = max(weighted_degrees)
    node_sizes = []
    for w in weighted_degrees:
        if w_max == w_min:
            node_sizes.append(1200.0)
        else:
            # 映射到 [800, 2200]
            size = 800 + (w - w_min) / (w_max - w_min) * 1400
            node_sizes.append(size)
else:
    node_sizes = [1200.0 for _ in G.nodes()]

# ==== 邊粗細：按 weight 放大 ====
edge_weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
if len(edge_weights) > 0:
    ew_min = min(edge_weights)
    ew_max = max(edge_weights)
    edge_widths = []
    for w in edge_weights:
        if ew_max == ew_min:
            edge_widths.append(1.5)
        else:
            # 映射到 [1.0, 5.0]
            width = 1.0 + (w - ew_min) / (ew_max - ew_min) * 4.0
            edge_widths.append(width)
else:
    edge_widths = [1.5 for _ in G.edges()]

# ==== 畫圖 ====
fig = plt.figure(figsize=(8, 6))

# 畫邊
nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)

# 畫節點
nx.draw_networkx_nodes(G, pos, node_size=node_sizes)

# 中文標籤（使用全局 rcParams 字體）
labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=13)

plt.title("《紅樓夢》人物共現網路（贾宝玉/黛玉/宝钗/熙凤/贾母）", fontsize=14)
plt.axis("off")
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, "hlm_network_graph.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"[OK] 網路圖已保存：{out_path}")
