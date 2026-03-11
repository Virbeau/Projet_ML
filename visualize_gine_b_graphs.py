import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch

from GINE_B_repartition import GINE_Allocation_Predictor

DATASET_JSON = "JSON/dataset_hybrid_mesh_sp_er_v2_1000.json"
MODEL_PATH = "checkpoints/gine_b_repartition_v3.pt"
OUTPUT_PATH = "gine_b_3_families_comparison.png"
DEVICE = "cpu"


def prepare_model_inputs_from_instance(inst):
    x = torch.tensor(inst["x"], dtype=torch.float32)
    x[:, 1] = x[:, 1] / 10.0
    x[:, 4] = x[:, 4] / 15.0
    x[:, 5] = x[:, 5] / 15.0
    x[:, 6] = x[:, 6] / 15.0
    x[:, 7] = x[:, 7] / 65.0
    x[:, 8] = x[:, 8] / 25.0

    edge_index = torch.tensor(inst["graph"]["edges"], dtype=torch.long).t().contiguous()
    edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    b_total = torch.tensor([inst["B"]], dtype=torch.float32)
    return x, edge_index, edge_attr, batch, b_total


def family_from_topology(inst):
    topology = str(inst.get("topology_type", "unknown"))
    if topology.startswith("mesh"):
        return "mesh"
    if topology.startswith("sp"):
        return "sp"
    if topology.startswith("er"):
        return "er"
    return None


def pick_three_families(instances):
    picked = {}
    for inst in instances:
        fam = family_from_topology(inst)
        if fam in ("mesh", "sp", "er") and fam not in picked:
            picked[fam] = inst
        if len(picked) == 3:
            break

    missing = [fam for fam in ("mesh", "sp", "er") if fam not in picked]
    if missing:
        raise RuntimeError(f"Familles introuvables dans le dataset: {missing}")

    return [picked["mesh"], picked["sp"], picked["er"]]


def build_graph(inst):
    g = nx.Graph()
    nodes = inst.get("graph", {}).get("nodes", list(range(len(inst["x"]))))
    edges = inst.get("graph", {}).get("edges", [])
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def predict_allocation(model, inst, device):
    x, edge_index, edge_attr, batch, b_total = prepare_model_inputs_from_instance(inst)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    batch = batch.to(device)
    b_total = b_total.to(device)
    with torch.no_grad():
        pred = model(x, edge_index, edge_attr, batch, b_total)
    return pred.detach().cpu().numpy()


def plot_comparison(instances, model, output_path, device):
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for row, inst in enumerate(instances):
        family = family_from_topology(inst)
        topology = inst.get("topology_type", "unknown")
        g = build_graph(inst)
        pos = nx.spring_layout(g, seed=42)

        solver_alloc = inst["y"]
        model_alloc = predict_allocation(model, inst, device)

        vmin = min(min(solver_alloc), float(model_alloc.min()))
        vmax = max(max(solver_alloc), float(model_alloc.max()))

        ax_left = axes[row, 0]
        nx.draw_networkx_edges(g, pos, ax=ax_left, alpha=0.4)
        nodes_left = nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax_left,
            node_color=solver_alloc,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            node_size=280,
        )
        nx.draw_networkx_labels(g, pos, ax=ax_left, font_size=8)
        ax_left.set_title(f"{family.upper()} | Solveur (y) | {topology}")
        ax_left.axis("off")

        ax_right = axes[row, 1]
        nx.draw_networkx_edges(g, pos, ax=ax_right, alpha=0.4)
        nodes_right = nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax_right,
            node_color=model_alloc,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            node_size=280,
        )
        nx.draw_networkx_labels(g, pos, ax=ax_right, font_size=8)
        ax_right.set_title(f"{family.upper()} | Modele entraine")
        ax_right.axis("off")

        cbar = fig.colorbar(nodes_right, ax=[ax_left, ax_right], fraction=0.03, pad=0.01)
        cbar.set_label("Allocation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualiser solveur vs modele sur 3 familles de graphes")
    parser.add_argument("--dataset", default=DATASET_JSON, help="Chemin du dataset JSON")
    parser.add_argument("--model", default=MODEL_PATH, help="Chemin du modele .pt")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Image de sortie")
    parser.add_argument("--device", default=DEVICE, help="cpu ou cuda")
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        raw = json.load(f)
    instances = raw.get("instances", [])

    selected_instances = pick_three_families(instances)

    device = torch.device(args.device)
    model = GINE_Allocation_Predictor().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    plot_comparison(selected_instances, model, args.output, device)
    print(f"Figure sauvegardee: {args.output}")


if __name__ == "__main__":
    main()
