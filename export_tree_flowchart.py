from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained scikit-learn decision tree as a Mermaid flowchart."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the pickled decision tree model.",
    )
    parser.add_argument(
        "--output-path",
        help="Output markdown file. Defaults to tree_flowchart.md next to the model.",
    )
    parser.add_argument(
        "--title",
        help="Optional chart title shown in the markdown output.",
    )
    return parser.parse_args()


def format_percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def clean_text(text: str) -> str:
    return text.replace('"', '\\"')


def format_node_label(model, node_id: int) -> tuple[str, str]:
    tree = model.tree_
    samples = int(tree.n_node_samples[node_id])
    value = tree.value[node_id][0]
    is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

    if hasattr(model, "classes_"):
        counts = value.astype(float)
        total = float(counts.sum())
        positive_rate = counts[1] / total if total else 0.0
        predicted_class = int(np.argmax(counts))
        if is_leaf:
            lines = [
                f"Leaf {node_id}",
                f"predict: {'UP' if predicted_class == 1 else 'DOWN'}",
                f"p(up): {positive_rate:.1%}",
                f"samples: {samples}",
            ]
            return "leaf", "<br/>".join(lines)

        feature_name = model.feature_names_in_[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        lines = [
            f"Node {node_id}",
            f"{feature_name} <= {threshold:.4f} ?",
            f"node p(up): {positive_rate:.1%}",
            f"samples: {samples}",
        ]
        return "decision", "<br/>".join(lines)

    mean_value = float(value[0])
    if is_leaf:
        lines = [
            f"Leaf {node_id}",
            f"predict 5d return: {format_percent(mean_value)}",
            f"samples: {samples}",
        ]
        return "leaf", "<br/>".join(lines)

    feature_name = model.feature_names_in_[tree.feature[node_id]]
    threshold = tree.threshold[node_id]
    lines = [
        f"Node {node_id}",
        f"{feature_name} <= {threshold:.4f} ?",
        f"node avg return: {format_percent(mean_value)}",
        f"samples: {samples}",
    ]
    return "decision", "<br/>".join(lines)


def build_mermaid(model) -> str:
    tree = model.tree_
    lines: list[str] = ["flowchart TD"]

    def visit(node_id: int) -> None:
        node_type, label = format_node_label(model, node_id)
        label = clean_text(label)
        if node_type == "decision":
            lines.append(f'    N{node_id}{{"{label}"}}')
        else:
            lines.append(f'    N{node_id}["{label}"]')

        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        if left_id == right_id:
            return

        visit(left_id)
        visit(right_id)
        lines.append(f"    N{node_id} -- yes --> N{left_id}")
        lines.append(f"    N{node_id} -- no --> N{right_id}")

    visit(0)

    lines.extend(
        [
            "    classDef decision fill:#f7f3d6,stroke:#7a6a00,color:#2f2a00,stroke-width:1px;",
            "    classDef leaf fill:#e3f4ea,stroke:#1b6b45,color:#123524,stroke-width:1px;",
        ]
    )

    for node_id in range(tree.node_count):
        node_type, _ = format_node_label(model, node_id)
        lines.append(f"    class N{node_id} {node_type};")

    return "\n".join(lines)


def build_markdown(title: str, model_path: Path, model) -> str:
    summary = {
        "model_file": str(model_path),
        "model_type": type(model).__name__,
        "max_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "feature_names": list(model.feature_names_in_),
    }
    header = [
        f"# {title}",
        "",
        "```mermaid",
        build_mermaid(model),
        "```",
        "",
        "## Notes",
        "",
        "- `yes` means the condition is true and the path goes left.",
        "- `no` means the condition is false and the path goes right.",
        "- `samples` is the number of training rows that reached that node.",
    ]

    if hasattr(model, "classes_"):
        header.append("- `p(up)` is the share of training samples labeled as up in that node.")
    else:
        header.append("- `predict 5d return` is the average target return for the training rows in that leaf.")

    header.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(header)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else model_path.with_name("tree_flowchart.md")

    with model_path.open("rb") as file_obj:
        model = pickle.load(file_obj)

    title = args.title or f"{type(model).__name__} Flowchart"
    markdown = build_markdown(title, model_path, model)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Flowchart written to: {output_path}")


if __name__ == "__main__":
    main()
