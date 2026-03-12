import argparse
import json

import matplotlib.pyplot as plt
import networkx as nx
import torch

from GINE_B_repartition import GINE_Allocation_Predictor

DATASET_JSON = "JSON/dataset_hybrid_mesh_sp_er_v2_1000.json"
MODEL_PATH = "checkpoints/gine_b_repartition_v4.pt"
OUTPUT_PATH = "gine_b_3_families_comparison.png"
DEVICE = "cpu"
BUDGET_TOL = 1e-4


def is_valid_instance(inst):
    num_nodes = len(inst.get("x", []))
    edges = inst.get("graph", {}).get("edges", [])
    for src, dst in edges:
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            return False
    return True


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

    nodes = inst["graph"].get("nodes", list(range(len(inst["x"]))))
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
    terminal_mask = torch.zeros(x.size(0), dtype=torch.bool)
    for terminal in inst.get("terminals", []):
        if terminal in node_to_idx:
            terminal_mask[node_to_idx[terminal]] = True
        elif isinstance(terminal, int) and 0 <= terminal < x.size(0):
            terminal_mask[terminal] = True

    return x, edge_index, edge_attr, batch, b_total, terminal_mask


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
    is_directed = bool(inst.get("graph", {}).get("is_directed", True))
    g = nx.DiGraph() if is_directed else nx.Graph()
    nodes = inst.get("graph", {}).get("nodes", list(range(len(inst["x"]))))
    edges = inst.get("graph", {}).get("edges", [])
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def predict_allocation(model, inst, device):
    x, edge_index, edge_attr, batch, b_total, terminal_mask = prepare_model_inputs_from_instance(inst)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    batch = batch.to(device)
    b_total = b_total.to(device)
    terminal_mask = terminal_mask.to(device)
    with torch.no_grad():
        pred = model(x, edge_index, edge_attr, batch, b_total, terminal_mask)
    return pred.detach().cpu().numpy().reshape(-1)


def draw_source_target(ax, g, pos, inst):
    terminals = inst.get("terminals", [])
    if len(terminals) >= 2:
        source = terminals[0]
        target = terminals[1]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[source],
            node_color="red",
            node_size=460,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=[target],
            node_color="orange",
            node_size=460,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )


def check_budget_constraint(instances, model, device, tol=BUDGET_TOL):
    violations = 0
    max_abs_err = 0.0
    mean_abs_err = 0.0

    for inst in instances:
        try:
            pred = predict_allocation(model, inst, device)
            b = float(inst["B"])
            abs_err = abs(float(pred.sum()) - b)
            mean_abs_err += abs_err
            max_abs_err = max(max_abs_err, abs_err)
            if abs_err > tol:
                violations += 1
        except Exception:
            # Ignore malformed instances for robustness in large mixed datasets.
            continue

    n = max(1, len(instances))
    return {
        "n_instances": len(instances),
        "violations": violations,
        "violation_rate": violations / n,
        "mean_abs_err": mean_abs_err / n,
        "max_abs_err": max_abs_err,
    }


def plot_comparison(instances, model, output_path, device):
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    for row, inst in enumerate(instances):
        family = family_from_topology(inst)
        topology = inst.get("topology_type", "unknown")
        is_directed = bool(inst.get("graph", {}).get("is_directed", True))
        g = build_graph(inst)
        pos = nx.spring_layout(g, seed=42)

        solver_alloc = inst["y"]
        model_alloc = predict_allocation(model, inst, device)
        budget = float(inst["B"])
        solver_sum = float(sum(solver_alloc))
        model_sum = float(model_alloc.sum())
        solver_abs_err = abs(solver_sum - budget)
        model_abs_err = abs(model_sum - budget)

        vmin = min(min(solver_alloc), float(model_alloc.min()))
        vmax = max(max(solver_alloc), float(model_alloc.max()))

        ax_left = axes[row, 0]
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax_left,
            alpha=0.4,
            arrows=is_directed,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.06",
        )
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
        draw_source_target(ax_left, g, pos, inst)
        ax_left.set_title(
            f"{family.upper()} | Solveur (y) | {topology}\n"
            f"B={budget:.3f} | Somme(y)={solver_sum:.3f} | |err|={solver_abs_err:.2e}"
            + (" | dirige" if is_directed else " | non dirige")
        )
        ax_left.axis("off")

        ax_right = axes[row, 1]
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax_right,
            alpha=0.4,
            arrows=is_directed,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.06",
        )
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
        draw_source_target(ax_right, g, pos, inst)
        ax_right.set_title(
            f"{family.upper()} | Modele entraine\n"
            f"B={budget:.3f} | Somme(pred)={model_sum:.3f} | |err|={model_abs_err:.2e}"
        )
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
    parser.add_argument(
        "--budget-tol",
        type=float,
        default=BUDGET_TOL,
        help="Tolerance absolue pour verifier la contrainte budget",
    )
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        raw = json.load(f)
    instances = raw.get("instances", [])
    valid_instances = [inst for inst in instances if is_valid_instance(inst)]

    if len(valid_instances) < len(instances):
        print(
            f"Instances invalides ignorees: {len(instances) - len(valid_instances)} / {len(instances)}"
        )

    selected_instances = pick_three_families(valid_instances)

    device = torch.device(args.device)
    model = GINE_Allocation_Predictor().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    plot_comparison(selected_instances, model, args.output, device)
    budget_stats = check_budget_constraint(valid_instances, model, device, tol=args.budget_tol)
    print(
        "Budget check | "
        f"instances={budget_stats['n_instances']} | "
        f"violations={budget_stats['violations']} | "
        f"taux={budget_stats['violation_rate']*100:.2f}% | "
        f"mean_abs_err={budget_stats['mean_abs_err']:.2e} | "
        f"max_abs_err={budget_stats['max_abs_err']:.2e}"
    )
    print(f"Figure sauvegardee: {args.output}")


if __name__ == "__main__":
    main()
