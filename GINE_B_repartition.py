import json

import torch
import wandb
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GINEConv, global_add_pool


TRAIN_JSON = "dataset_v3_10000.json"
TRAIN_CLEAN_INVALID_EDGES = True
TRAIN_JSTAR_MIN = None
TRAIN_JSTAR_MAX = None

BENCHMARK_JSON = "JSON/dataset_hybrid_mesh_sp_er_v2_1000.json"
BENCHMARK_CLEAN_INVALID_EDGES = True
BENCHMARK_JSTAR_MIN = 0.01
BENCHMARK_JSTAR_MAX = 0.99

TRAIN_VAL_SPLIT = 0.8
SPLIT_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
WANDB_RUN_NAME = "GINE_B_V3_50e"
WANDB_GROUP = "Gine_B"
SAVE_MODEL = True
MODEL_SAVE_PATH = "checkpoints/gine_b_repartition_v3.pt"
COMPUTE_DELTAJ = True
DELTAJ_N_SIMS = 10000
DELTAJ_MAX_INSTANCES = None


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
        data.B = torch.tensor([inst["B"]], dtype=torch.float32)
        return data



from torch_geometric.utils import softmax 

class GINE_Allocation_Predictor(torch.nn.Module):
    def __init__(self, num_node_features=9, hidden_dim=64, edge_dim=1):
        super(GINE_Allocation_Predictor, self).__init__()
        
        # 2 couches GINE
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
        
        # La Tête de Régression
        self.mlp_readout = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1)
            # ATTENTION : On enlève le dernier ReLU ! On laisse le réseau cracher 
            # des nombres bruts (positifs ou négatifs), le Softmax gérera le reste.
        )

    def forward(self, x, edge_index, edge_attr, batch, B_total):
        # 1. Message Passing
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        
        # 2. Prédiction "brute" (des scores d'importance sans limite)
        raw_scores = self.mlp_readout(x).squeeze()
        
        # 3. LE RESPECT DU BUDGET GRÂCE AU SOFTMAX PAR GRAPHE
        # Le softmax va transformer ces scores en pourcentages (entre 0 et 1)
        # Et il va garantir que la somme des pourcentages D'UN MÊME GRAPHE = 1.0
        alloc_probs = softmax(raw_scores, index=batch)
        
        # 4. On multiplie ces pourcentages par le Vrai Budget du graphe
        B_broadcasted = B_total[batch]
        final_alloc = alloc_probs * B_broadcasted
        
        return final_alloc


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

    print(f"Train/val dataset: {len(train_val_dataset)} graphes")
    print(f"Benchmark fixe: {len(benchmark_dataset)} graphes")

    train_size = int(TRAIN_VAL_SPLIT * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    split_generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    benchmark_loader = DataLoader(benchmark_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Definition de la fonction de perte et de l'optimiseur
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.B.view(-1))
                batch_loss = criterion(pred.view(-1), batch.y_node.view(-1))
                total_loss += batch_loss.item() * batch.num_nodes
        return total_loss / max(1, sum(d.num_nodes for d in loader.dataset))

    # Entraînement du modèle
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.B.view(-1))
            loss = criterion(out.view(-1), data.y_node.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
        train_loss = total_loss / max(1, sum(d.num_nodes for d in train_loader.dataset))
        val_loss = evaluate(val_loader)
        benchmark_loss = evaluate(benchmark_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "benchmark_loss": benchmark_loss,
        })

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Benchmark Loss: {benchmark_loss:.4f}"
        )
    # Calcul de la MAE par noeud et envoi a wandb
    def evaluate_mae(loader):
        model.eval()
        total_mae = 0.0
        total_nodes = 0
        with torch.no_grad():
            for batch in loader:
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.B.view(-1))
                mae = F.l1_loss(pred.view(-1), batch.y_node.view(-1), reduction='sum')
                total_mae += mae.item()
                total_nodes += batch.num_nodes
        return total_mae / total_nodes

    benchmark_mae = evaluate_mae(benchmark_loader)
    wandb.log({"benchmark_mae": benchmark_mae})
    print(f"Benchmark MAE: {benchmark_mae:.4f}")

    if SAVE_MODEL:
        import os

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
            print(
                f"deltaJ termine | abs_mean={delta_metrics['mean_delta_abs']:.6f} | "
                f"rel_mean={delta_metrics['mean_delta_rel']*100:.2f}%"
            )
        except Exception as exc:
            print(f"[WARN] Echec calcul deltaJ: {exc}")
            if wandb.run is not None:
                wandb.log({"deltaJ_error": 1})
                wandb.summary["deltaJ_error_message"] = str(exc)

    wandb.finish()


#Evaluation du deltaJ
import numpy as np

def _prepare_model_inputs_from_instance(inst):
    """Prepare tensors exactly like ReliabilityDataset.__getitem__ for consistency."""
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
    return x, edge_index, edge_attr, batch, B_total


def _extract_raw_instances(dataset_or_instances):
    if hasattr(dataset_or_instances, "instances"):
        return dataset_or_instances.instances
    return dataset_or_instances


def evaluate_industrial_regret(model, dataset_or_instances, simulate_monte_carlo, n_sims=10000):
    """
    Évalue le sur-risque (Regret) généré par les choix du GNN.
    
    Arguments:
    - model: Ton modèle GINE entraîné (Tâche B)
    - dataset_or_instances: ReliabilityDataset ou liste d'instances brutes
    - simulate_monte_carlo: Ta fonction python qui simule un graphe
    """
    model.eval() # Mode évaluation (désactive les gradients)
    delta_j_abs = []
    delta_j_rel = []
    raw_instances = _extract_raw_instances(dataset_or_instances)
    
    print("Calcul deltaJ en cours (Monte-Carlo)...")
    
    with torch.no_grad():
        for idx, inst in enumerate(raw_instances):
            # 1. Formater le graphe pour le GNN avec le meme preprocessing qu'en training
            x, edge_index, edge_attr, batch, B_total = _prepare_model_inputs_from_instance(inst)
            
            # 2. Le GNN fait sa prédiction d'allocation
            pi_gnn_tensor = model(x, edge_index, edge_attr, batch, B_total)
            pi_gnn = pi_gnn_tensor.cpu().numpy().tolist()
            
            # 3. La Vérité Absolue (déjà calculée par le solveur)
            J_star = inst['J_star']
            
            # 4. L'Épreuve du feu : On donne la politique du GNN au simulateur
            # /!\ Adapte cette ligne selon comment ton simulateur lit l'allocation /!\
            # Ici, on remplace virtuellement l'allocation optimale par celle du GNN
            inst_to_simulate = inst.copy()
            inst_to_simulate['pi_evaluated'] = pi_gnn 
            
            # On lance 10 000 tirages pour avoir une vraie précision physique
            J_gnn = simulate_monte_carlo(inst_to_simulate, n_sims=n_sims)
            
            # 5. Calcul de deltaJ
            # J* = risque optimal (plus petit est meilleur), donc sur-risque = J_gnn - J_star
            delta_abs = J_gnn - J_star
            
            # Astuce : Comme Monte-Carlo a une micro-marge d'erreur, il se peut
            # que J_gnn soit très légèrement inférieur à J_star (ex: -0.001).
            # On force le sur-risque à 0 dans ce cas pour ne pas fausser la moyenne.
            delta_abs = max(0.0, delta_abs)

            if J_star > 1e-12:
                delta_rel = delta_abs / J_star
            else:
                delta_rel = 0.0

            delta_j_abs.append(delta_abs)
            delta_j_rel.append(delta_rel)
            
            if idx % 100 == 0:
                print(f"Progression deltaJ: {idx}/{len(raw_instances)}")

    # --- RÉSULTATS ---
    mean_delta_abs = float(np.mean(delta_j_abs)) if delta_j_abs else 0.0
    max_delta_abs = float(np.max(delta_j_abs)) if delta_j_abs else 0.0
    median_delta_abs = float(np.median(delta_j_abs)) if delta_j_abs else 0.0

    mean_delta_rel = float(np.mean(delta_j_rel)) if delta_j_rel else 0.0
    max_delta_rel = float(np.max(delta_j_rel)) if delta_j_rel else 0.0
    median_delta_rel = float(np.median(delta_j_rel)) if delta_j_rel else 0.0
    
    print(
        "deltaJ | "
        f"abs(mean/med/max)={mean_delta_abs:.6f}/{median_delta_abs:.6f}/{max_delta_abs:.6f} | "
        f"rel%(mean/med/max)={mean_delta_rel*100:.2f}/{median_delta_rel*100:.2f}/{max_delta_rel*100:.2f}"
    )

    if wandb.run is not None:
        wandb.log(
            {
                "deltaJ_abs_mean": mean_delta_abs,
                "deltaJ_abs_median": median_delta_abs,
                "deltaJ_abs_max": max_delta_abs,
                "deltaJ_rel_mean": mean_delta_rel,
                "deltaJ_rel_median": median_delta_rel,
                "deltaJ_rel_max": max_delta_rel,
                "deltaJ_abs_hist": wandb.Histogram(np.array(delta_j_abs)),
                "deltaJ_rel_hist": wandb.Histogram(np.array(delta_j_rel)),
                "deltaJ_num_instances": len(delta_j_abs),
            }
        )

        wandb.summary["deltaJ_abs_mean"] = mean_delta_abs
        wandb.summary["deltaJ_abs_median"] = median_delta_abs
        wandb.summary["deltaJ_abs_max"] = max_delta_abs
        wandb.summary["deltaJ_rel_mean"] = mean_delta_rel
        wandb.summary["deltaJ_rel_median"] = median_delta_rel
        wandb.summary["deltaJ_rel_max"] = max_delta_rel

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
    