# -*- coding: utf-8 -*-
"""
analyze_hongloumeng_network.py

從 hongloumeng_sentences.csv 構建 5 個人物的共現網路：
- 賈寶玉（BAOYU）
- 林黛玉（DAIYU）
- 薛寶釵（BAOCHAI）
- 王熙鳳（WANGXIFENG）
- 賈母（JIAMU）

輸出：
- hlm_network_nodes.csv
- hlm_network_edges.csv
- hlm_network.gexf  (可用 Gephi 打開)

這個版本加了大量 print，方便你看到程序在做什麼。
"""

import argparse
import os
from itertools import combinations
from typing import Dict, List, Set, Tuple

import pandas as pd
import networkx as nx

# ---- 角色別名 ----

CHAR_ALIASES: Dict[str, List[str]] = {
    "BAOYU": [
        "賈寶玉", "贾宝玉",
        "寶玉", "宝玉",
        "寶二爺", "宝二爷",
        "寶玉兒", "宝玉儿",
    ],
    "DAIYU": [
        "林黛玉",
        "黛玉",
        "黛玉姐", "黛玉姐姐",
        "瀟湘妃子", "潇湘妃子",
    ],
    "BAOCHAI": [
        "薛寶釵", "薛宝钗",
        "寶釵", "宝钗",
        "寶姐姐", "宝姐姐",
    ],
    "WANGXIFENG": [
        "王熙鳳", "王熙凤",
        "熙鳳", "熙凤",
        "鳳姐", "凤姐",
        "鳳姐兒", "凤姐儿",
    ],
    "JIAMU": [
        "賈母", "贾母",
        "老太太", "老祖宗", "老太君",
    ],
}

CHAR_LABELS: Dict[str, str] = {
    "BAOYU": "贾宝玉",
    "DAIYU": "林黛玉",
    "BAOCHAI": "薛宝钗",
    "WANGXIFENG": "王熙凤",
    "JIAMU": "贾母",
}


def parse_chapter_range(spec: str, all_chapters: List[int]) -> List[int]:
    """解析章回範圍字串，若 spec 為空，則返回 CSV 中所有章回。"""
    if not spec or not spec.strip():
        return sorted(set(all_chapters))
    nums: List[int] = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if a <= b else -1
            nums.extend(range(a, b + step, step))
        else:
            nums.append(int(part))
    valid = set(all_chapters)
    return sorted(set(n for n in nums if n in valid))


def detect_chars_in_text(text: str) -> Set[str]:
    """檢測一句話中出現了哪些角色（簡單 substring 匹配）。"""
    present: Set[str] = set()
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    for cid, aliases in CHAR_ALIASES.items():
        for alias in aliases:
            if alias and alias in text:
                present.add(cid)
                break
    return present


def build_edges_by_window(
    df: pd.DataFrame,
    chapters: List[int],
    window: int = 1,
) -> Dict[Tuple[str, str], Dict]:
    """
    用「句子窗口 ±window 句」構建共現邊：
    - 每個窗口中若 ≥2 個角色同現，就對每一對角色加 1 權重
    - 同時記錄該邊出現過的章回號
    """
    edges: Dict[Tuple[str, str], Dict] = {}

    for chap in chapters:
        df_c = df[df["chapter_no"] == chap].sort_values("sentence_index")
        texts = df_c["text"].fillna("").tolist()
        char_sets: List[Set[str]] = [detect_chars_in_text(t) for t in texts]
        n = len(texts)
        if n == 0:
            continue

        print(f"[INFO] 處理第 {chap} 回，句子數：{n}")

        for i in range(n):
            start = max(0, i - window)
            end = min(n - 1, i + window)
            window_chars: Set[str] = set()
            for j in range(start, end + 1):
                window_chars.update(char_sets[j])
            if len(window_chars) < 2:
                continue
            for a, b in combinations(sorted(window_chars), 2):
                key = (a, b)
                if key not in edges:
                    edges[key] = {"weight": 0, "chapters": set()}
                edges[key]["weight"] += 1
                edges[key]["chapters"].add(chap)

    return edges


def main():
    print("=== Hongloumeng network analysis started ===")

    parser = argparse.ArgumentParser(
        description="《紅樓夢》人物共現網路分析（寶玉/黛玉/寶釵/王熙鳳/賈母）"
    )
    parser.add_argument(
        "--sentences",
        type=str,
        required=True,
        help="hongloumeng_sentences.csv 路徑",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./results",
        help="輸出目錄（默認 ./results）",
    )
    parser.add_argument(
        "--chapters",
        type=str,
        default="",
        help="分析章回，如 '1-20,30'；默認為 CSV 中所有章回",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="句子窗口大小（±window 句，共現判定）。默認 1。",
    )

    args = parser.parse_args()
    print(f"[ARGS] sentences={args.sentences}")
    print(f"[ARGS] outdir={args.outdir}")
    print(f"[ARGS] chapters='{args.chapters}'")
    print(f"[ARGS] window={args.window}")

    os.makedirs(args.outdir, exist_ok=True)

    # 讀取句子級 CSV
    if not os.path.exists(args.sentences):
        print(f"[ERROR] 找不到文件：{args.sentences}")
        return

    print(f"[INFO] 讀取句子文件：{args.sentences}")
    df = pd.read_csv(args.sentences, encoding="utf-8")
    print(f"[INFO] 讀入行數：{len(df)}")

    # 保險起見，把 chapter_no 轉成 int
    if "chapter_no" not in df.columns:
        print("[ERROR] CSV 中沒有 'chapter_no' 列，請檢查文件格式。")
        print(f"[DEBUG] 列名：{list(df.columns)}")
        return

    df = df.copy()
    df["chapter_no"] = pd.to_numeric(df["chapter_no"], errors="coerce")
    df = df.dropna(subset=["chapter_no"])
    df["chapter_no"] = df["chapter_no"].astype(int)

    all_chaps = sorted(df["chapter_no"].unique().tolist())
    print(f"[INFO] CSV 中共有 {len(all_chaps)} 個章回：{all_chaps[:20]}{' ...' if len(all_chaps) > 20 else ''}")

    chapters = parse_chapter_range(args.chapters, all_chaps)
    print(f"[INFO] 將實際分析的章回：{chapters[:20]}{' ...' if len(chapters) > 20 else ''}")

    df = df[df["chapter_no"].isin(chapters)]
    print(f"[INFO] 篩選後行數：{len(df)}")

    if len(df) == 0:
        print("[WARN] 篩選後沒有任何句子，請檢查 --chapters 參數。")
        return

    print(f"[INFO] 使用句子窗口 ±{args.window} 構建共現網路…")
    edges = build_edges_by_window(df, chapters, window=args.window)
    print(f"[INFO] 共現邊數量：{len(edges)}")

    # 構建 NetworkX 圖
    G = nx.Graph()
    for cid, label in CHAR_LABELS.items():
        G.add_node(cid, label=label)

    for (a, b), info in edges.items():
        G.add_edge(
            a,
            b,
            weight=info["weight"],
            chapters="|".join(str(c) for c in sorted(info["chapters"])),
            chapter_count=len(info["chapters"]),
        )

    print(f"[INFO] 圖中節點數：{G.number_of_nodes()}，邊數：{G.number_of_edges()}")

    # 計算中心性指標
    print("[INFO] 計算節點中心性指標…")
    deg_unweighted = dict(G.degree())
    deg_weighted = dict(G.degree(weight="weight"))
    bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    clo = nx.closeness_centrality(G)

    # 節點表
    nodes_rows: List[Dict] = []
    for cid in G.nodes():
        nodes_rows.append({
            "char_id": cid,
            "label": CHAR_LABELS.get(cid, cid),
            "degree": deg_unweighted.get(cid, 0),
            "weighted_degree": deg_weighted.get(cid, 0.0),
            "betweenness": bet.get(cid, 0.0),
            "closeness": clo.get(cid, 0.0),
        })
    nodes_df = pd.DataFrame(nodes_rows)
    nodes_path = os.path.join(args.outdir, "hlm_network_nodes.csv")
    nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 節點統計已保存：{nodes_path}")

    # 邊表
    edge_rows: List[Dict] = []
    for (a, b), info in edges.items():
        edge_rows.append({
            "char_a": a,
            "char_a_label": CHAR_LABELS.get(a, a),
            "char_b": b,
            "char_b_label": CHAR_LABELS.get(b, b),
            "weight": info["weight"],
            "chapters": "|".join(str(c) for c in sorted(info["chapters"])),
            "chapter_count": len(info["chapters"]),
        })
    edges_df = pd.DataFrame(edge_rows)
    edges_path = os.path.join(args.outdir, "hlm_network_edges.csv")
    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 邊列表已保存：{edges_path}")

    # GEXF
    gexf_path = os.path.join(args.outdir, "hlm_network.gexf")
    nx.write_gexf(G, gexf_path)
    print(f"[OK] 網路文件已保存：{gexf_path}")

    print("[DONE] 分析完成。")


if __name__ == "__main__":
    main()
