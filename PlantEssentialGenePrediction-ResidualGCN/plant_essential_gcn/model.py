from typing import Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


class ResidualGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        if x.size(0) > 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual


class SimpleModel(nn.Module):
    def __init__(self, config, main_feat_dim: int, bio_feat_dim: int = 0):
        super().__init__()
        self.config = config
        self.num_kmer = 4 ** config.K
        total_feat_dim = main_feat_dim + bio_feat_dim
        print(f"  - Model input dimension: {total_feat_dim}(main features{main_feat_dim} + biological features{bio_feat_dim})")
        print(f"  - Graph structure: 1gene node + {self.num_kmer}k-mer nodes, GCNConvlayers={config.GCN_LAYERS}")

        self.input_projection = nn.Sequential(
            nn.Linear(total_feat_dim, config.GCN_HIDDEN),
            nn.BatchNorm1d(config.GCN_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        self.kmer_projection = nn.Sequential(
            nn.Linear(self.num_kmer, config.GCN_HIDDEN),
            nn.BatchNorm1d(config.GCN_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        self.gcn_layers = nn.ModuleList()
        for _ in range(config.GCN_LAYERS):
            self.gcn_layers.append(ResidualGCNLayer(config.GCN_HIDDEN, config.GCN_HIDDEN, config.DROPOUT))

        self.classifier = nn.Sequential(
            nn.Linear(config.GCN_HIDDEN, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 2)
        )

    def _set_bn_eval_for_single_graph(self):
        for module_group in [self.input_projection, self.kmer_projection, self.classifier]:
            for m in module_group.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def _build_homogeneous_node_features(self, batch: Data) -> Tuple[torch.Tensor, int, int]:
        x_gene_raw = batch.x_p.squeeze(1)
        if x_gene_raw.dim() == 1:
            x_gene_raw = x_gene_raw.unsqueeze(0)
        if x_gene_raw.dtype != torch.float32:
            x_gene_raw = x_gene_raw.float()

        x_kmer_raw = batch.x_f
        if x_kmer_raw.dtype != torch.float32:
            x_kmer_raw = x_kmer_raw.float()

        batch_size = x_gene_raw.size(0)
        num_kmer = x_kmer_raw.size(0) // max(batch_size, 1)
        if num_kmer != self.num_kmer:
            raise ValueError(f"k-mer node count mismatch: current batch={num_kmer}, model expected={self.num_kmer}")

        x_gene = self.input_projection(x_gene_raw)
        x_kmer = self.kmer_projection(x_kmer_raw)

        x_kmer = x_kmer.view(batch_size, num_kmer, -1)
        x = torch.cat([x_gene.unsqueeze(1), x_kmer], dim=1).reshape(batch_size * (num_kmer + 1), -1)
        return x, batch_size, num_kmer

    def forward(self, batch: Data) -> torch.Tensor:
        if batch.x_p.squeeze(1).size(0) == 1:
            self._set_bn_eval_for_single_graph()

        x, batch_size, num_kmer = self._build_homogeneous_node_features(batch)

        edge_index = batch.edge_index
        edge_weight = getattr(batch, 'edge_weight', None)
        if edge_weight is not None and edge_weight.numel() == 0:
            edge_weight = None

        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, edge_index, edge_weight=edge_weight)

        nodes_per_graph = num_kmer + 1
        batch_vector = getattr(batch, 'batch', None)
        if batch_vector is None or batch_vector.numel() != x.size(0):
            batch_vector = torch.arange(batch_size, device=x.device).repeat_interleave(nodes_per_graph)

        pooled_graph = global_mean_pool(x, batch_vector)
        gene_node = x[torch.arange(batch_size, device=x.device) * nodes_per_graph]


        graph_repr = 0.5 * (gene_node + pooled_graph)
        logits = self.classifier(graph_repr)
        return logits
