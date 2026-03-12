import json

import torch
import wandb
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GINEConv, global_mean_pool

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
WANDB_RUN_NAME = "GINE_J*_V6_20e"
WANDB_GROUP = "Gine_J"


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

class GINE_JStar_Predictor(torch.nn.Module):
    def __init__(self, num_node_features=9, hidden_dim=64, edge_dim=1):
        super(GINE_JStar_Predictor, self).__init__()
        
        # ---------------------------------------------------------
        # 1ère COUCHE GINE
        # Chaque couche GINE nécessite un petit réseau de neurones (MLP)
        # pour traiter les messages reçus par les voisins.
        # ---------------------------------------------------------
        nn1 = Sequential(
            Linear(num_node_features, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.conv1 = GINEConv(nn1, edge_dim=edge_dim)
        
        # ---------------------------------------------------------
        # 2ème COUCHE GINE
        # ---------------------------------------------------------
        nn2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.conv2 = GINEConv(nn2, edge_dim=edge_dim)
        
        # ---------------------------------------------------------
        # TÊTE DE RÉGRESSION (Pour prédire J*)
        # ---------------------------------------------------------
        self.mlp_readout = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, 1), # Sortie : 1 seul neurone (notre J*)
            Sigmoid() # Force la prédiction à rester entre 0.0 et 1.0 !
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: Caractéristiques des noeuds [Nombre total de noeuds, 9]
        edge_index: Connexions [2, Nombre total d'arêtes]
        edge_attr: Poids des arêtes [Nombre total d'arêtes, 1]
        batch: Vecteur qui indique quel noeud appartient à quel graphe
        """
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        
        x_graph = global_mean_pool(x, batch) 
        out = self.mlp_readout(x_graph)
        
        return out


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
        }
    )

    model = GINE_JStar_Predictor()

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
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                batch_loss = criterion(pred.view(-1), batch.y.view(-1))
                total_loss += batch_loss.item() * batch.num_graphs
        return total_loss / len(loader.dataset)

    # Entraînement du modèle
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out.view(-1), data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        train_loss = total_loss / len(train_loader.dataset)
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

    # --- Évaluation finale sur le benchmark fixe (MSE + MAE) ---
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in benchmark_loader:
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_preds.append(pred.view(-1))
            all_targets.append(data.y.view(-1))

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    final_mse = F.mse_loss(all_preds, all_targets).item()
    final_mae = F.l1_loss(all_preds, all_targets).item()

    print(f"=== Évaluation finale sur le benchmark fixe ===")
    print(f"MSE : {final_mse:.4f}")
    print(f"MAE : {final_mae:.4f}")

    wandb.summary["final_benchmark_mse"] = final_mse
    wandb.summary["final_benchmark_mae"] = final_mae
    wandb.finish()

