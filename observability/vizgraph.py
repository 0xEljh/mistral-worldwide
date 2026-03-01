from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _object_node_id(obj: dict[str, Any]) -> str:
    return f"{obj.get('type', 'object')}_{obj.get('id', 'unknown')}"


def _confidence_to_size(confidence: Any) -> float:
    if not isinstance(confidence, (int, float)):
        return 700.0

    bounded_conf = max(0.0, min(1.0, float(confidence)))
    return 450.0 + (bounded_conf * 900.0)


def _relation_color(relation: str) -> str:
    parts = {part.strip().lower() for part in relation.split(",")}
    if "overlapping" in parts:
        return "#c0392b"
    if "near" in parts:
        return "#e67e22"
    return "#7f8c8d"


def _build_type_palette(objects: list[dict[str, Any]]) -> dict[str, Any]:
    object_types = sorted(
        {
            str(obj.get("type", "object"))
            for obj in objects
            if obj.get("type") is not None
        }
    )
    cmap = plt.get_cmap("tab20")
    return {
        object_type: cmap(idx % cmap.N) for idx, object_type in enumerate(object_types)
    }


def render_world_graph(snapshot: dict[str, Any], output_path: Path) -> Path:
    objects = snapshot.get("objects", [])
    relations = snapshot.get("relations", [])

    graph = nx.DiGraph()
    positions: dict[str, tuple[float, float]] = {}
    type_palette = _build_type_palette(objects)

    for obj in objects:
        node_id = _object_node_id(obj)
        object_type = str(obj.get("type", "object"))

        position = obj.get("position", {})
        x = float(position.get("x", 0.0))
        y = float(position.get("y", 0.0))

        graph.add_node(
            node_id,
            type=object_type,
            confidence=float(obj.get("confidence", 0.0)),
        )
        positions[node_id] = (x, y)

    for relation in relations:
        subject = str(relation.get("subject", ""))
        object_id = str(relation.get("object", ""))
        relation_text = str(relation.get("relation", ""))

        if subject not in positions or object_id not in positions:
            continue

        graph.add_edge(
            subject,
            object_id,
            label=relation_text,
            color=_relation_color(relation_text),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(12, 8), dpi=150)

    world_version = snapshot.get("world_version", 0)
    timestamp = snapshot.get("timestamp", 0)
    axes.set_title(f"World Graph | world_version={world_version} frame={timestamp}")

    if not positions:
        axes.text(
            0.5,
            0.5,
            "No visible objects in this snapshot",
            transform=axes.transAxes,
            ha="center",
            va="center",
        )
        axes.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path, format="png")
        plt.close(figure)
        return output_path

    node_list = list(graph.nodes)
    node_sizes = [
        _confidence_to_size(graph.nodes[node_id].get("confidence"))
        for node_id in node_list
    ]
    node_colors = [
        type_palette.get(
            str(graph.nodes[node_id].get("type", "object")),
            "#2980b9",
        )
        for node_id in node_list
    ]

    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=1.5,
        edgecolors="#1f2d3d",
        ax=axes,
    )
    nx.draw_networkx_labels(graph, pos=positions, font_size=9, ax=axes)

    edge_list = list(graph.edges)
    if edge_list:
        edge_colors = [
            str(graph.edges[(source, target)].get("color", "#7f8c8d"))
            for source, target in edge_list
        ]
        nx.draw_networkx_edges(
            graph,
            pos=positions,
            edgelist=edge_list,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            width=1.5,
            connectionstyle="arc3,rad=0.08",
            ax=axes,
        )
        edge_labels = {
            (source, target): str(graph.edges[(source, target)].get("label", ""))
            for source, target in edge_list
        }
        nx.draw_networkx_edge_labels(
            graph,
            pos=positions,
            edge_labels=edge_labels,
            font_size=7,
            rotate=False,
            label_pos=0.55,
            ax=axes,
        )

    x_values = [point[0] for point in positions.values()]
    y_values = [point[1] for point in positions.values()]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_padding = max(30.0, (x_max - x_min) * 0.15)
    y_padding = max(30.0, (y_max - y_min) * 0.15)

    axes.set_xlim(x_min - x_padding, x_max + x_padding)
    axes.set_ylim(y_max + y_padding, y_min - y_padding)
    axes.set_xlabel("x")
    axes.set_ylabel("y (image coordinates)")
    axes.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="#1f2d3d",
            markersize=8,
            label=object_type,
        )
        for object_type, color in type_palette.items()
    ]
    if legend_handles:
        axes.legend(handles=legend_handles, title="Object Types", loc="upper right")

    figure.tight_layout()
    figure.savefig(output_path, format="png")
    plt.close(figure)
    return output_path


class GraphSnapshotRecorder:
    def __init__(self, output_dir: Path, interval: int = 1000):
        if interval <= 0:
            raise ValueError("interval must be greater than 0")

        self.output_dir = Path(output_dir)
        self.interval = interval
        self._lock = threading.Lock()
        self._next_snapshot_version = interval

    def maybe_record(self, snapshot: dict[str, Any]) -> Path | None:
        try:
            world_version = int(snapshot.get("world_version", 0))
        except (TypeError, ValueError):
            return None

        if world_version <= 0:
            return None

        with self._lock:
            if world_version < self._next_snapshot_version:
                return None

            while self._next_snapshot_version <= world_version:
                self._next_snapshot_version += self.interval

            try:
                timestamp = int(snapshot.get("timestamp", 0))
            except (TypeError, ValueError):
                timestamp = 0

            output_path = self.output_dir / f"graph_v{world_version}_f{timestamp}.png"
            return render_world_graph(snapshot, output_path)
