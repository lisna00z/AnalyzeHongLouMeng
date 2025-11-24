# -*- coding: utf-8 -*-
"""
app.py - Streamlit dashboard (EN version)

Displays the co-occurrence network of five key characters in
Hongloumeng (Dream of the Red Chamber):

- Jia Baoyu
- Lin Daiyu
- Xue Baochai
- Wang Xifeng
- Jia Mu

It reads:
- results/hlm_network_nodes.csv
- results/hlm_network_edges.csv
- results/hlm_weighted_degree_bar.png
- results/hlm_top_edges_bar.png
- results/hlm_network_graph.png
"""

import streamlit as st
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

nodes_path = RESULTS / "hlm_network_nodes.csv"
edges_path = RESULTS / "hlm_network_edges.csv"

nodes = pd.read_csv(nodes_path, encoding="utf-8")
edges = pd.read_csv(edges_path, encoding="utf-8")

st.set_page_config(
    page_title="Hongloumeng Character Network",
    layout="wide"
)

# ===== Title & intro =====
st.title("Character Co-occurrence Network in *Hongloumeng*")

st.markdown(
    """
This dashboard presents a small-scale **digital humanities** analysis
of **five key characters** in *Hongloumeng* (Dream of the Red Chamber):

- **Jia Baoyu**
- **Lin Daiyu**
- **Xue Baochai**
- **Wang Xifeng**
- **Jia Mu**

We use sentence-level co-occurrence to build a simple social network,
based on the CText version of the full 120 chapters.
"""
)

# ===== 1. Node centrality =====
st.subheader("1. Node-level centrality")

nodes_display = nodes.copy()
nodes_display = nodes_display.sort_values("weighted_degree", ascending=False)

st.markdown(
    """
**How to read the node metrics:**

- `degree`: number of distinct neighbours (unweighted degree).  
- `weighted_degree`: sum of edge weights, i.e. total co-occurrence strength.  
- `betweenness`: betweenness centrality – how often a node lies on shortest paths.  
- `closeness`: closeness centrality – how close a node is to all others.
"""
)

st.dataframe(nodes_display, use_container_width=True)

# ===== 2. Top 10 strongest pairs =====
st.subheader("2. Top 10 strongest character pairs (by co-occurrence)")

edges_top = edges.sort_values("weight", ascending=False).head(10).copy()
edges_top["pair"] = (
    edges_top["char_a_label"] + " – " + edges_top["char_b_label"]
)
edges_top = edges_top[["pair", "weight", "chapters", "chapter_count"]]

st.markdown(
    """
Below are the **10 strongest character pairs**, ranked by
co-occurrence frequency (`weight`). You can roughly read this as
“who appears together the most often” under our window-based definition.
"""
)
st.dataframe(edges_top, use_container_width=True)

# ===== 3. Visualisations =====
st.subheader("3. Visualisations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**(a) Weighted degree of the five characters**")
    img1_path = RESULTS / "hlm_weighted_degree_bar.png"
    if img1_path.exists():
        st.image(str(img1_path), use_container_width=True)
    else:
        st.info("`hlm_weighted_degree_bar.png` not found. Please run `make_hlm_figures.py` first.")

with col2:
    st.markdown("**(b) Top 10 strongest relationships**")
    img2_path = RESULTS / "hlm_top_edges_bar.png"
    if img2_path.exists():
        st.image(str(img2_path), use_container_width=True)
    else:
        st.info("`hlm_top_edges_bar.png` not found. Please run `make_hlm_figures.py` first.")

st.markdown("**(c) Character co-occurrence network graph**")
img3_path = RESULTS / "hlm_network_graph.png"
if img3_path.exists():
    st.image(str(img3_path), use_container_width=True)
else:
    st.info("`hlm_network_graph.png` not found. Please run `make_hlm_network_graph.py` first.")

st.markdown("---")
st.markdown(
    """
**Reading guide**

- **Node size** in the network graph is proportional to `weighted_degree`:  
  the more a character co-occurs with others, the larger the node.
- **Edge thickness** is proportional to `weight`:  
  thicker edges mean stronger co-occurrence between two characters.
- The network is built with a **±1 sentence window** and only covers the
  five characters listed above. It highlights a small but interpretable
  slice of the much richer social world in *Hongloumeng*.
"""
)

