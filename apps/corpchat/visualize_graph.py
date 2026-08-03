#!/usr/bin/env python3
"""
visualize_graph.py — 可视化 txtai 搜索索引的图结构
=====================================================
从已构建的搜索索引中提取图数据（节点 + 边），生成交互式 HTML 可视化。

用法:
  python3 apps/corpchat/visualize_graph.py                     # 使用默认索引
  python3 apps/corpchat/visualize_graph.py --index-path <path> # 指定索引路径
  python3 apps/corpchat/visualize_graph.py --output graph.html # 指定输出文件
  python3 apps/corpchat/visualize_graph.py --max-nodes 50      # 限制节点数

依赖:
  pip install pyvis networkx
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("visualize-graph")

# ── 确保项目根目录在 Python 路径上 ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from apps.corpchat.search import (
    load_index,
    DEFAULT_INDEX_PATH,
    _clean_text_from_enriched,
)


def extract_graph_data(index_path: str, max_nodes: int = 0, min_edge_weight: float = 0.0) -> Dict:
    """
    从 txtai 索引中提取图数据。

    Args:
        index_path: 索引路径
        max_nodes: 最大节点数 (0=全部)
        min_edge_weight: 最小边权重 (过滤低相似度边)

    Returns:
        {"nodes": [{"id": int, "doc_id": str, "label": str, "text_preview": str, "group": str}, ...],
         "edges": [{"source": int, "target": int, "weight": float}, ...]}
    """
    embeddings = load_index(index_path)
    g = embeddings.graph
    if not g:
        raise RuntimeError("索引中未启用图功能 (graph mode=off 时不会构建图)")

    nx_graph = g.backend
    total_nodes = nx_graph.number_of_nodes()
    total_edges = nx_graph.number_of_edges()
    logger.info(f"图: {total_nodes} 个节点, {total_edges} 条边")

    # 收集所有节点 (按 id 排序以保持一致性)
    all_nodes = sorted(nx_graph.nodes())
    if max_nodes > 0 and max_nodes < len(all_nodes):
        # 取前 max_nodes 个节点 (按 id 顺序)
        selected = set(all_nodes[:max_nodes])
        logger.info(f"限制为前 {max_nodes} 个节点")
    else:
        selected = set(all_nodes)

    # 构建节点列表
    nodes = []
    for nid in sorted(selected):
        attrs = nx_graph.nodes[nid]
        doc_id = attrs.get("id", str(nid))
        text = attrs.get("text", "")

        # 从文本中提取标签作为分组
        label = "unknown"
        if "Metadata:" in text:
            meta_part = text.split("Metadata:")[-1]
            for part in meta_part.split(";"):
                part = part.strip()
                if part.startswith("label="):
                    label = part.split("=", 1)[1]
                    break

        # 生成简短的显示标签
        text_clean = _clean_text_from_enriched(text)
        text_preview = text_clean[:60] + "..." if len(text_clean) > 60 else text_clean

        nodes.append({
            "id": nid,
            "doc_id": doc_id,
            "label": doc_id.split("__")[0][:30] if "__" in doc_id else doc_id[:30],
            "text_preview": text_preview,
            "group": label,
        })

    # 构建边列表
    edges = []
    for src, dst, data in nx_graph.edges(data=True):
        if src in selected and dst in selected:
            weight = data.get("weight", 0.0)
            if weight >= min_edge_weight:
                edges.append({
                    "source": src,
                    "target": dst,
                    "weight": weight,
                })

    logger.info(f"输出: {len(nodes)} 个节点, {len(edges)} 条边")
    return {"nodes": nodes, "edges": edges}


def generate_html(graph_data: Dict, output_path: str) -> str:
    """
    使用 pyvis 生成交互式 HTML 可视化。
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("请安装 pyvis: pip install pyvis")
        sys.exit(1)

    net = Network(height="800px", width="100%", directed=False, notebook=False)

    # 配置物理引擎
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.4
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based",
        "timestep": 0.35
      },
      "edges": {
        "smooth": false,
        "color": {"inherit": "from"}
      },
      "nodes": {
        "font": {"size": 10, "face": "Arial"}
      }
    }
    """)

    # 添加节点
    for node in graph_data["nodes"]:
        # 使用标签类型作为颜色分组
        group_colors = {
            "product_inquiry": "#3498db",
            "order_confirmation": "#2ecc71",
            "tech_support": "#e74c3c",
            "meeting_schedule": "#f39c12",
            "invoice_issue": "#9b59b6",
            "software_license": "#1abc9c",
            "contract_renewal": "#e67e22",
            "quality_issue": "#c0392b",
            "marketing_campaign": "#16a085",
            "warranty_claim": "#8e44ad",
            "old_friend_reconnect": "#d35400",
            "investment_opportunity": "#27ae60",
        }
        color = group_colors.get(node["group"], "#95a5a6")

        title = (
            f"<b>ID:</b> {node['doc_id']}<br>"
            f"<b>Label:</b> {node['group']}<br>"
            f"<b>Content:</b> {node['text_preview']}"
        )

        net.add_node(
            node["id"],
            label=node["label"],
            title=title,
            color=color,
            group=node["group"],
            size=8,
        )

    # 添加边
    for edge in graph_data["edges"]:
        # 权重映射为透明度/粗细
        width = max(0.3, min(3.0, edge["weight"] * 3.0))
        opacity = max(0.1, min(0.8, edge["weight"]))
        color_val = f"rgba(100, 100, 100, {opacity})"
        net.add_edge(
            edge["source"],
            edge["target"],
            value=width,
            title=f"相似度: {edge['weight']:.4f}",
            color=color_val,
        )

    # 添加图例说明 (通过添加不可见的节点)
    legend_colors = [
        ("product_inquiry", "#3498db"),
        ("order_confirmation", "#2ecc71"),
        ("tech_support", "#e74c3c"),
        ("meeting_schedule", "#f39c12"),
        ("invoice_issue", "#9b59b6"),
        ("software_license", "#1abc9c"),
        ("contract_renewal", "#e67e22"),
        ("old_friend_reconnect", "#d35400"),
        ("investment_opportunity", "#27ae60"),
        ("other", "#95a5a6"),
    ]
    legend_html = '<div style="position:absolute;top:10px;right:10px;background:white;border:1px solid #ccc;border-radius:4px;padding:8px;font-size:12px;z-index:1000">'
    legend_html += "<b>Label</b><br>"
    for lbl, clr in legend_colors:
        legend_html += f'<span style="display:inline-block;width:10px;height:10px;background:{clr};border-radius:50%;margin-right:4px"></span>{lbl}<br>'
    legend_html += "</div>"

    # 保存 HTML
    net.save_graph(output_path)

    # 将图例注入到 HTML 中
    with open(output_path, "r") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", f"{legend_html}</body>")
    with open(output_path, "w") as f:
        f.write(html_content)

    abs_path = os.path.abspath(output_path)
    logger.info(f"可视化已保存至: {abs_path}")
    return abs_path


def main():
    parser = argparse.ArgumentParser(
        description="可视化 txtai 搜索索引的图结构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH,
                        help=f"索引路径 (默认: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--output", default="corpchat_graph.html",
                        help="输出 HTML 文件名 (默认: corpchat_graph.html)")
    parser.add_argument("--max-nodes", type=int, default=0,
                        help="最大节点数 (0=全部, 默认: 0)")
    parser.add_argument("--min-edge-weight", type=float, default=0.7,
                        help="最小边权重 (默认: 0.7, 设为0显示所有边)")
    args = parser.parse_args()

    if not os.path.exists(args.index_path):
        logger.error(f"索引不存在: {args.index_path}")
        logger.error("请先运行: python apps/corpchat/search.py build --force")
        sys.exit(1)

    logger.info(f"索引: {args.index_path}")
    logger.info(f"最大节点: {'全部' if args.max_nodes == 0 else args.max_nodes}")
    logger.info(f"最小边权重: {args.min_edge_weight}")

    graph_data = extract_graph_data(
        args.index_path,
        max_nodes=args.max_nodes,
        min_edge_weight=args.min_edge_weight,
    )

    if not graph_data["nodes"]:
        logger.warning("没有节点可显示")
        return

    output_path = generate_html(graph_data, args.output)
    print(f"\n✅ 可视化完成: {output_path}")
    print(f"   节点: {len(graph_data['nodes'])}")
    print(f"   边: {len(graph_data['edges'])}")
    print(f"\n用浏览器打开即可查看交互式图。")


if __name__ == "__main__":
    main()