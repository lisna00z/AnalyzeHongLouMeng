# -*- coding: utf-8 -*-
"""
make_hlm_figures.py
從 hlm_network_nodes.csv / hlm_network_edges.csv 產出兩張圖：
1) 5 人物的加權度對比條形圖
2) 人物配對關係強度（邊權）條形圖
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# 解决中文显示为方框的问题：指定中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results")

nodes_path = os.path.join(RESULTS_DIR, "hlm_network_nodes.csv")
edges_path = os.path.join(RESULTS_DIR, "hlm_network_edges.csv")

print("[INFO] 讀取節點與邊文件…")
nodes = pd.read_csv(nodes_path, encoding="utf-8")
edges = pd.read_csv(edges_path, encoding="utf-8")

# -------- 圖 1：5 人物加權度對比 --------
nodes_sorted = nodes.sort_values("weighted_degree", ascending=True)

plt.figure(figsize=(6, 4))
plt.barh(nodes_sorted["label"], nodes_sorted["weighted_degree"])
plt.xlabel("加權度（Weighted Degree）")
plt.ylabel("角色")
plt.title("《紅樓夢》人物共現網路：加權度對比")
plt.tight_layout()
fig1_path = os.path.join(RESULTS_DIR, "hlm_weighted_degree_bar.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"[OK] 圖 1 已保存：{fig1_path}")

# -------- 圖 2：人物配對關係強度 --------
# 為了畫得清楚，只顯示前 10 條最強關係
edges_sorted = edges.sort_values("weight", ascending=False).head(10).copy()
edges_sorted["pair"] = edges_sorted["char_a_label"] + " - " + edges_sorted["char_b_label"]
edges_sorted = edges_sorted.sort_values("weight", ascending=True)

plt.figure(figsize=(6, 4))
plt.barh(edges_sorted["pair"], edges_sorted["weight"])
plt.xlabel("共現次數（weight）")
plt.ylabel("人物配對")
plt.title("《紅樓夢》人物共現網路：最強的 10 對關係")
plt.tight_layout()
fig2_path = os.path.join(RESULTS_DIR, "hlm_top_edges_bar.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"[OK] 圖 2 已保存：{fig2_path}")

print("[DONE] 圖片已生成，可直接插入 PPT。")
