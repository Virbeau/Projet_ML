import argparse
import json
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data, Dataset

DEFAULT_JSON_PATH = "JSON/dataset_hybrid_mesh_sp_er_v2_1000.json"


def is_valid_instance(inst: Dict) -> bool:
    num_nodes = len(inst.get("x", []))
    edges = inst.get("graph", {}).get("edges", [])
    for src, dst in edges:
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            return False
    return True


def split_valid_instances(instances: List[Dict]) -> Tuple[List[Dict], int]:
    valid_instances = []
    invalid_count = 0
    for inst in instances:
        if is_valid_instance(inst):
            valid_instances.append(inst)
        else:
            invalid_count += 1
    return valid_instances, invalid_count


def clean_json_file(input_file: str, output_file: str) -> Tuple[int, int]:
    with open(input_file, "r") as f:
        data = json.load(f)

    instances = data.get("instances", [])
    valid_instances, invalid_count = split_valid_instances(instances)
    data["instances"] = valid_instances

    with open(output_file, "w") as f:
        json.dump(data, f)

    return len(valid_instances), invalid_count


class ReliabilityDataset(Dataset):
    def __init__(self, json_file: str = DEFAULT_JSON_PATH, clean_invalid_edges: bool = True):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset loader + cleaner")
    parser.add_argument("--input", default=DEFAULT_JSON_PATH, help="Input JSON path")
    parser.add_argument(
        "--export-clean",
        default="",
        help="If set, export a cleaned JSON file containing only valid instances",
    )
    parser.add_argument(
        "--no-clean-loader",
        action="store_true",
        help="Disable automatic filtering of invalid edges in loader",
    )
    args = parser.parse_args()

    if args.export_clean:
        kept, removed = clean_json_file(args.input, args.export_clean)
        print(f"Clean file written to: {args.export_clean}")
        print(f"Instances kept: {kept} | removed: {removed}")

    dataset = ReliabilityDataset(
        json_file=args.input,
        clean_invalid_edges=not args.no_clean_loader,
    )
    print(f"Dataset charge avec {len(dataset)} graphes")
