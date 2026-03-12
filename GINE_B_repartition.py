import json
import os
import torch
import wandb
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.loader import DataLoader
import numpy as np

# --- Configuration ---
TRAIN_JSON = "datasetV6.json"
TRAIN_CLEAN_INVALID_EDGES = True
TRAIN_JSTAR_MIN = None
TRAIN_JSTAR_MAX = None

BENCHMARK_JSON = "benchmark_v6_40.json"
BENCHMARK_CLEAN_INVALID_EDGES = True
BENCHMARK_JSTAR_MIN = 0.01
BENCHMARK_JSTAR_MAX = 0.99

TRAIN_VAL_SPLIT = 0.8
SPLIT_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
WANDB_RUN_NAME = "GINE_B_V6_20e"
WANDB_GROUP = "Gine_B"
SAVE_MODEL = True
MODEL_SAVE_PATH = "checkpoints/gine_b_repartition_v6.pt"
COMPUTE_DELTAJ = True
DELTAJ_N_SIMS = 1000
DELTAJ_MAX_INSTANCES = None
BUDGET_TOL = 1e-4

# --- Fonctions Utilitaires ---
def is_valid_instance(inst):
    num_nodes = len(inst.get("x", []))
    edges = inst.get("graph", {}).get("edges", [])
    for src, dst in edges:
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            return False
    return True


def split_valid_instances(instances):
    valid_instances = []
    invalid_count = 0
    for inst in instances:
        if is_valid_instance(inst):
            valid_instances.append(inst)
        else:
            invalid_count += 1
    return valid_instances, invalid_count


def filter_instances_by_jstar(instances, jstar_min=None, jstar_max=None):
    if jstar_min is None and jstar_max is None:
        return instances, 0

    if jstar_min is not None and jstar_max is not None and jstar_min > jstar_max:
        raise ValueError("JSTAR_MIN doit etre inferieur ou egal a JSTAR_MAX")

    filtered_instances = []
    removed_count = 0

    for inst in instances:
        value = inst.get("J_star")
        if not isinstance(value, (int, float)):
            filtered_instances.append(inst)
            continue

        value = float(value)
        if jstar_min is not None and value < jstar_min:
            removed_count += 1
            continue
        if jstar_max is not None and value > jstar_max:
            removed_count += 1
            continue

        filtered_instances.append(inst)

    return filtered_instances, removed_count


# --- Dataset ---
class ReliabilityDataset(Dataset):
    def __init__(
        self,
        json_file,
        clean_invalid_edges=True,
        jstar_min=None,
        jstar_max=None,
    ):
        super().__init__()
        with open(json_file, "r") as f:
            raw_data = json.load(f)

        instances = raw_data.get("instances", [])
        if clean_invalid_edges:
            self.instances, invalid_count = split_valid_instances(instances)
            if invalid_count > 0:
                print(
                    f"[ReliabilityDataset] {invalid_count} instances invalides ignorees "
                    f"sur {len(instances)}"
                )
        else:
            self.instances = instances

        if jstar_min is not None or jstar_max is not None:
            original_len = len(self.instances)
            self.instances, removed = filter_instances_by_jstar(
                self.instances,
                jstar_min=jstar_min,
                jstar_max=jstar_max,
            )
            print(
                f"[ReliabilityDataset] Filtrage direct sur 'J_star' avec bornes "
                f"[{jstar_min if jstar_min is not None else '-inf'}, "
                f"{jstar_max if jstar_max is not None else '+inf'}] | "
                f"supprimees: {removed} / {original_len}"
            )

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]

        x = torch.tensor(inst["x"], dtype=torch.float32)
        x[:, 1] = x[:, 1] / 10.0
        x[:, 4] = x[:, 4] / 15.0
        x[:, 5] = x[:, 5] / 15.0
        x[:, 6] = x[:, 6] / 15.0
        x[:, 7] = x[:, 7] / 65.0
        x[:, 8] = x[:, 8] / 25.0

        edges = inst["graph"]["edges"]
        nodes = inst["graph"].get("nodes", list(range(len(inst["x"]))))
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float32)

        y_graph = torch.tensor([inst["J_star"]], dtype=torch.float32)
        y_node = torch.tensor(inst["y"], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_graph,
            y_node=y_node,
        )

        terminal_mask = torch.zeros(x.size(0), dtype=torch.bool)
        for terminal in inst.get("terminals", []):
            if terminal in node_to_idx:
                terminal_mask[node_to_idx[terminal]] = True
            elif isinstance(terminal, int) and 0 <= terminal < x.size(0):
                terminal_mask[terminal] = True

        data.B = torch.tensor([inst["B"]], dtype=torch.float32)
        data.terminal_mask = terminal_mask

        # Extraction du coût réel (c_cost) depuis x[:,1] AVANT normalisation
        # x[:,1] a déjà été divisé par 10 ci-dessus, on récupère la valeur brute
        c_list = inst.get("c_cost", inst.get("params", {}).get("c_cost", None))
        if c_list is not None:
            data.c_cost = torch.tensor(c_list, dtype=torch.float32)
        else:
            # Extraire directement depuis le JSON (valeur brute, avant normalisation)
            raw_x = inst["x"]
            data.c_cost = torch.tensor([row[1] for row in raw_x], dtype=torch.float32)

        return data


# --- Modèle ---
class GINE_Allocation_Predictor(torch.nn.Module):
    def __init__(self, num_node_features=9, hidden_dim=64, edge_dim=1):
        super(GINE_Allocation_Predictor, self).__init__()
        
        self.conv1 = GINEConv(
            nn=Sequential(
                Linear(num_node_features, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            ),
            edge_dim=edge_dim,
        )
        self.conv2 = GINEConv(
            nn=Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
            ),
            edge_dim=edge_dim,
        )
        
        # Tête de Régression avec Sigmoid pour forcer des probas [0, 1]
        self.mlp_readout = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1),
            Sigmoid() 
        )

    def forward(self, x, edge_index, edge_attr, batch, B_total, terminal_mask=None, c_cost=None):
        # 1. Message Passing
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 2. Prédiction des probabilités brutes (entre 0 et 1)
        pi_raw = self.mlp_readout(x).squeeze(-1)
        
        # 3. Contrainte métier : aucun budget sur les terminaux
        if terminal_mask is not None:
            non_terminal = (~terminal_mask.bool()).float()
            pi_raw = pi_raw * non_terminal
            
        if c_cost is None:
            c_cost = torch.ones_like(pi_raw)
            
        # 4. Calcul de la dépense totale : Somme(pi_i * c_i)
        node_expenses = pi_raw * c_cost
        total_expenses = global_add_pool(node_expenses.view(-1, 1), batch).view(-1)
        
        # 5. Projection (Inégalité <= B)
        B_broadcasted = B_total[batch]
        expenses_broadcasted = total_expenses[batch]
        
        # Ratio : si dépense <= B, ratio=1. Sinon ratio = B / dépense
        ratio = torch.clamp(B_broadcasted / (expenses_broadcasted + 1e-12), max=1.0)
        final_alloc = pi_raw * ratio
        
        return final_alloc


# --- Préparation Inférence ---
def _prepare_model_inputs_from_instance(inst):
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
    B_total = torch.tensor([inst["B"]], dtype=torch.float32)

    nodes = inst["graph"].get("nodes", list(range(len(inst["x"]))))
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
    terminal_mask = torch.zeros(x.size(0), dtype=torch.bool)
    for terminal in inst.get("terminals", []):
        if terminal in node_to_idx:
            terminal_mask[node_to_idx[terminal]] = True
        elif isinstance(terminal, int) and 0 <= terminal < x.size(0):
            terminal_mask[terminal] = True

    c_list = inst.get("c_cost", inst.get("params", {}).get("c_cost", None))
    if c_list is not None:
        c_cost = torch.tensor(c_list, dtype=torch.float32)
    else:
        # Extraire depuis le JSON brut (avant normalisation)
        c_cost = torch.tensor([row[1] for row in inst["x"]], dtype=torch.float32)

    return x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost

def _extract_raw_instances(dataset_or_instances):
    if hasattr(dataset_or_instances, "instances"):
        return dataset_or_instances.instances
    return dataset_or_instances


# --- Évaluation ---
def evaluate_industrial_regret(model, dataset_or_instances, simulate_monte_carlo, n_sims=10000):
    model.eval()
    delta_j_abs = []
    delta_j_rel = []
    raw_instances = _extract_raw_instances(dataset_or_instances)

    print("Calcul deltaJ...")

    with torch.no_grad():
        for idx, inst in enumerate(raw_instances):
            x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost = _prepare_model_inputs_from_instance(inst)

            pi_gnn_tensor = model(x, edge_index, edge_attr, batch, B_total, terminal_mask, c_cost)
            pi_gnn = pi_gnn_tensor.cpu().numpy().tolist()

            J_star = inst["J_star"]

            inst_to_simulate = inst.copy()
            inst_to_simulate["pi_evaluated"] = pi_gnn

            J_gnn = simulate_monte_carlo(inst_to_simulate, n_sims=n_sims)

            delta_abs = max(0.0, J_gnn - J_star)
            delta_rel = (delta_abs / J_star) if J_star > 1e-12 else 0.0

            delta_j_abs.append(delta_abs)
            delta_j_rel.append(delta_rel)

            if idx % 200 == 0:
                print(f"deltaJ: {idx}/{len(raw_instances)}")

    mean_delta_abs = float(np.mean(delta_j_abs)) if delta_j_abs else 0.0
    max_delta_abs = float(np.max(delta_j_abs)) if delta_j_abs else 0.0
    median_delta_abs = float(np.median(delta_j_abs)) if delta_j_abs else 0.0

    mean_delta_rel = float(np.mean(delta_j_rel)) if delta_j_rel else 0.0
    max_delta_rel = float(np.max(delta_j_rel)) if delta_j_rel else 0.0
    median_delta_rel = float(np.median(delta_j_rel)) if delta_j_rel else 0.0

    print(f"deltaJ | abs_mean={mean_delta_abs:.6f} | rel_mean={mean_delta_rel*100:.2f}%")

    if wandb.run is not None:
        wandb.log({
            "deltaJ_abs_mean": mean_delta_abs,
            "deltaJ_rel_mean": mean_delta_rel,
            "deltaJ_num_instances": len(delta_j_abs),
        })
        wandb.summary["deltaJ_abs_mean"] = mean_delta_abs
        wandb.summary["deltaJ_rel_mean"] = mean_delta_rel

    return {
        "delta_j_abs": delta_j_abs,
        "delta_j_rel": delta_j_rel,
        "mean_delta_abs": mean_delta_abs,
        "median_delta_abs": median_delta_abs,
        "max_delta_abs": max_delta_abs,
        "mean_delta_rel": mean_delta_rel,
        "median_delta_rel": median_delta_rel,
        "max_delta_rel": max_delta_rel,
    }


# --- Script Principal ---
if __name__ == "__main__":
    run = wandb.init(
        project="Reliability_GNN",
        name=WANDB_RUN_NAME,
        group=WANDB_GROUP,
        config={
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "wandb_group": WANDB_GROUP,
            "train_json": TRAIN_JSON,
            "train_jstar_min": TRAIN_JSTAR_MIN,
            "train_jstar_max": TRAIN_JSTAR_MAX,
            "benchmark_json": BENCHMARK_JSON,
            "benchmark_jstar_min": BENCHMARK_JSTAR_MIN,
            "benchmark_jstar_max": BENCHMARK_JSTAR_MAX,
            "split_seed": SPLIT_SEED,
            "save_model": SAVE_MODEL,
            "model_save_path": MODEL_SAVE_PATH,
            "compute_deltaJ": COMPUTE_DELTAJ,
            "deltaJ_n_sims": DELTAJ_N_SIMS,
            "deltaJ_max_instances": DELTAJ_MAX_INSTANCES,
            "budget_tol": BUDGET_TOL,
        }
    )

    model = GINE_Allocation_Predictor()

    train_val_dataset = ReliabilityDataset(
        json_file=TRAIN_JSON,
        clean_invalid_edges=TRAIN_CLEAN_INVALID_EDGES,
        jstar_min=TRAIN_JSTAR_MIN,
        jstar_max=TRAIN_JSTAR_MAX,
    )

    benchmark_dataset = ReliabilityDataset(
        json_file=BENCHMARK_JSON,
        clean_invalid_edges=BENCHMARK_CLEAN_INVALID_EDGES,
        jstar_min=BENCHMARK_JSTAR_MIN,
        jstar_max=BENCHMARK_JSTAR_MAX,
    )

    print(f"Datasets | train_val={len(train_val_dataset)} | benchmark={len(benchmark_dataset)}")

    train_size = int(TRAIN_VAL_SPLIT * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    benchmark_loader = DataLoader(benchmark_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def evaluate_terminal_leakage(loader, tol=1e-10):
        model.eval()
        max_terminal_alloc = 0.0
        sum_terminal_alloc = 0.0
        n_terminal_nodes = 0
        n_non_zero = 0

        with torch.no_grad():
            for batch in loader:
                pred = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.B.view(-1),
                    batch.terminal_mask,
                    batch.c_cost,
                )

                terminal_values = pred[batch.terminal_mask.bool()].abs()
                if terminal_values.numel() == 0:
                    continue

                max_terminal_alloc = max(max_terminal_alloc, float(terminal_values.max().item()))
                sum_terminal_alloc += float(terminal_values.sum().item())
                n_terminal_nodes += int(terminal_values.numel())
                n_non_zero += int((terminal_values > tol).sum().item())

        mean_terminal_alloc = sum_terminal_alloc / max(1, n_terminal_nodes)
        non_zero_rate = n_non_zero / max(1, n_terminal_nodes)
        return {
            "max_terminal_alloc": max_terminal_alloc,
            "mean_terminal_alloc": mean_terminal_alloc,
            "non_zero_rate": non_zero_rate,
            "n_terminal_nodes": n_terminal_nodes,
        }

    def evaluate_budget_constraint(loader, tol=BUDGET_TOL):
        model.eval()
        violations = 0
        n_graphs = 0
        total_abs_err = 0.0
        max_abs_err = 0.0

        with torch.no_grad():
            for batch in loader:
                pred = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.B.view(-1),
                    batch.terminal_mask,
                    batch.c_cost,
                )
                
                # Le vrai calcul de l'argent dépensé (pi * coût)
                pred_sum = global_add_pool((pred * batch.c_cost).view(-1, 1), batch.batch).view(-1)
                b_true = batch.B.view(-1)
                
                # On compte l'erreur seulement si on DEPASSE le budget
                excess = F.relu(pred_sum - b_true)

                violations += int((excess > tol).sum().item())
                n_graphs += int(batch.num_graphs)
                total_abs_err += float(excess.sum().item())
                max_abs_err = max(max_abs_err, float(excess.max().item()))

        mean_abs_err = total_abs_err / max(1, n_graphs)
        violation_rate = violations / max(1, n_graphs)
        return {
            "violations": violations,
            "n_graphs": n_graphs,
            "violation_rate": violation_rate,
            "mean_abs_err": mean_abs_err,
            "max_abs_err": max_abs_err,
        }

    def evaluate_mae(loader):
        model.eval()
        total_mae = 0.0
        total_nodes = 0
        with torch.no_grad():
            for batch in loader:
                pred = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    batch.B.view(-1),
                    batch.terminal_mask,
                    batch.c_cost,
                )
                mae = F.l1_loss(pred.view(-1), batch.y_node.view(-1), reduction='sum')
                total_mae += mae.item()
                total_nodes += batch.num_nodes
        return total_mae / total_nodes


    # --- Entraînement ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
                data.B.view(-1),
                data.terminal_mask,
                data.c_cost,
            )
            loss = criterion(out.view(-1), data.y_node.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
            
        train_loss = total_loss / max(1, sum(d.num_nodes for d in train_loader.dataset))
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss})
        if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

    # --- Évaluations Finales ---
    benchmark_mae = evaluate_mae(benchmark_loader)
    final_val_budget = evaluate_budget_constraint(val_loader)
    final_benchmark_budget = evaluate_budget_constraint(benchmark_loader)
    final_benchmark_terminal = evaluate_terminal_leakage(benchmark_loader)

    wandb.log({
        "benchmark_mae": benchmark_mae,
        "budget_violation_rate": final_benchmark_budget["violation_rate"],
        "budget_mean_abs_err": final_benchmark_budget["mean_abs_err"],
        "budget_max_abs_err": final_benchmark_budget["max_abs_err"],
        "terminal_alloc_max": final_benchmark_terminal["max_terminal_alloc"],
    })

    wandb.summary["benchmark_mae"] = benchmark_mae
    wandb.summary["budget_violation_rate"] = final_benchmark_budget["violation_rate"]
    wandb.summary["budget_mean_abs_err"] = final_benchmark_budget["mean_abs_err"]
    wandb.summary["budget_max_abs_err"] = final_benchmark_budget["max_abs_err"]
    wandb.summary["terminal_alloc_max"] = final_benchmark_terminal["max_terminal_alloc"]

    print(
        "Final metrics | "
        f"node_mae={benchmark_mae:.4f} | "
        f"budget_viol={final_benchmark_budget['violations']}/{final_benchmark_budget['n_graphs']} "
        f"(mean_err={final_benchmark_budget['mean_abs_err']:.2e}) | "
        f"terminal_max={final_benchmark_terminal['max_terminal_alloc']:.2e}"
    )

    if SAVE_MODEL:
        save_dir = os.path.dirname(MODEL_SAVE_PATH)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Modele sauvegarde: {MODEL_SAVE_PATH}")
        if wandb.run is not None:
            wandb.log({"model_saved": 1})
            wandb.summary["model_save_path"] = MODEL_SAVE_PATH
    else:
        if wandb.run is not None:
            wandb.log({"model_saved": 0})

    if COMPUTE_DELTAJ:
        try:
            from monte_carlo_validation import simulate_monte_carlo

            delta_dataset = benchmark_dataset
            if DELTAJ_MAX_INSTANCES is not None:
                selected_instances = benchmark_dataset.instances[:DELTAJ_MAX_INSTANCES]
                delta_dataset = selected_instances

            delta_metrics = evaluate_industrial_regret(
                model=model,
                dataset_or_instances=delta_dataset,
                simulate_monte_carlo=simulate_monte_carlo,
                n_sims=DELTAJ_N_SIMS,
            )
            print(f"deltaJ done | abs_mean={delta_metrics['mean_delta_abs']:.6f} | rel_mean={delta_metrics['mean_delta_rel']*100:.2f}%")
        except Exception as exc:
            print(f"[WARN] Echec calcul deltaJ: {exc}")
            if wandb.run is not None:
                wandb.log({"deltaJ_error": 1})
                wandb.summary["deltaJ_error_message"] = str(exc)

    wandb.finish()