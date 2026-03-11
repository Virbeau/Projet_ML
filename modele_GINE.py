import torch
import wandb
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Sigmoid
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool

INPUT_JSON = "dataset_v3_10000_clean.json"

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
        name="GINE_Jstar_10000",
        config={"batch_size": 32, "lr": 0.001, "epochs": 10}
    )

    model = GINE_JStar_Predictor()

    # Chargement des données
    from dataloader import ReliabilityDataset
    dataset = ReliabilityDataset(INPUT_JSON)

    # Separation des données en train/test naïvement (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Definition de la fonction de perte et de l'optimiseur
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
    num_epochs = 10
    for epoch in range(num_epochs):
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
        test_loss = evaluate(test_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # --- Évaluation finale sur le test set (MSE + MAE) ---
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in test_loader:
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_preds.append(pred.view(-1))
            all_targets.append(data.y.view(-1))

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    final_mse = F.mse_loss(all_preds, all_targets).item()
    final_mae = F.l1_loss(all_preds, all_targets).item()

    print(f"=== Évaluation finale sur le test set ===")
    print(f"MSE : {final_mse:.4f}")
    print(f"MAE : {final_mae:.4f}")

    wandb.summary["final_test_mse"] = final_mse
    wandb.summary["final_test_mae"] = final_mae
    wandb.finish()

