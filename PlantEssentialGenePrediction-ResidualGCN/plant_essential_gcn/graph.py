import numpy as np
import torch
from torch_geometric.data import Data


def build_sparse_bipartite_graph_with_bio(main_feat: np.ndarray, kmer_count: np.ndarray,
                                          bio_feat: np.ndarray, K: int, config) -> Data:

    num_kmer = 4 ** K
    main_feat = main_feat.astype(np.float32)

    if bio_feat is not None and len(bio_feat) > 0:
        bio_feat = bio_feat.astype(np.float32)
        combined_feat = np.concatenate([main_feat, bio_feat])
        x_p = torch.tensor(combined_feat, dtype=torch.float32).unsqueeze(0)
    else:
        x_p = torch.tensor(main_feat, dtype=torch.float32).unsqueeze(0)


    x_f = torch.eye(num_kmer, dtype=torch.float32)

    valid_kmer_idx = np.where(kmer_count >= config.MIN_KMER_COUNT)[0]
    if len(valid_kmer_idx) == 0:
        valid_kmer_idx = np.arange(min(5, num_kmer))


    kmer_nodes = torch.tensor((valid_kmer_idx + 1).tolist(), dtype=torch.long)
    gene_nodes = torch.zeros(len(valid_kmer_idx), dtype=torch.long)

    edge_gene_to_kmer = torch.stack([gene_nodes, kmer_nodes], dim=0)
    edge_kmer_to_gene = torch.stack([kmer_nodes, gene_nodes], dim=0)
    edge_index = torch.cat([edge_gene_to_kmer, edge_kmer_to_gene], dim=1).contiguous()


    edge_values = kmer_count[valid_kmer_idx].astype(np.float32)
    edge_values = np.log1p(edge_values)
    max_edge_value = float(edge_values.max()) if edge_values.size > 0 else 1.0
    if max_edge_value <= 0:
        max_edge_value = 1.0
    edge_values = edge_values / max_edge_value
    edge_weight = torch.tensor(np.concatenate([edge_values, edge_values]), dtype=torch.float32)


    edge_index_pf = torch.stack([gene_nodes, kmer_nodes], dim=0).contiguous()
    edge_index_fp = torch.stack([kmer_nodes, gene_nodes], dim=0).contiguous()

    return Data(x_p=x_p, x_f=x_f,
                edge_index=edge_index, edge_weight=edge_weight,
                edge_index_pf=edge_index_pf, edge_index_fp=edge_index_fp,
                num_nodes_p=1, num_nodes_f=num_kmer, num_nodes=1 + num_kmer,
                y=torch.tensor([0], dtype=torch.long))


def augment_seq(seq_str: str, augment_prob: float) -> str:
    seq_list = list(seq_str)
    for i in range(len(seq_list)):
        if np.random.random() < augment_prob:
            if seq_list[i] == "A":
                seq_list[i] = "T"
            elif seq_list[i] == "T":
                seq_list[i] = "A"
            elif seq_list[i] == "C":
                seq_list[i] = "G"
            elif seq_list[i] == "G":
                seq_list[i] = "C"
    return "".join(seq_list)
