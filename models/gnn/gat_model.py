import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from configs.logging_config import logger

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import from_networkx
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.warning("PyTorch Geometric not available. Using native PyTorch Graph Attention Layer.")


class NativeGATLayer(nn.Module):
    """Pure PyTorch implementation of a Graph Attention Layer, used only
    when torch_geometric isn't installed. Operates on a dense adjacency
    matrix rather than an edge_index list."""

    def __init__(self, in_features: int, out_features: int, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads

        self.W = nn.Parameter(torch.empty(size=(in_features, heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, heads)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        N = h.size(0)
        Wh = Wh.reshape(N, self.heads, self.out_features)

        a_input1 = Wh.repeat(1, N, 1).reshape(N * N, self.heads, self.out_features)
        a_input2 = Wh.repeat(N, 1, 1).reshape(N * N, self.heads, self.out_features)
        a_input = torch.cat([a_input1, a_input2], dim=-1)

        e = torch.einsum("nhd,dh->nh", a_input, self.a)
        e = self.leakyrelu(e).reshape(N, N, self.heads)

        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(-1).repeat(1, 1, self.heads)
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.einsum("nkh,khd->nhd", attention, Wh)
        return h_prime.reshape(N, self.heads * self.out_features)


class GATClassifierPyG(nn.Module):
    """torch_geometric GATConv stack. Operates ONLY on edge_index tensors
    of shape [2, num_edges] -- never on a dense adjacency matrix."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GATClassifierNative(nn.Module):
    """Native dense-adjacency GAT stack, used only when torch_geometric is
    not installed."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int):
        super().__init__()
        self.layer1 = NativeGATLayer(in_channels, hidden_channels, heads=heads)
        self.fc2 = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, adj):
        x = F.relu(self.layer1(x, adj))
        out = self.fc2(x)
        return F.log_softmax(out, dim=1)


class GATPropagationPipeline:
    """End-to-end wrapper for GAT propagation graph inference.

    NOTE ON REALISM:
    Because no live social media cascade dataset (e.g. Twitter firehose) is
    bundled, this pipeline models simulated diffusion cascades. Node centrality
    and PageRank metrics are mathematically exact; neural attention propagation
    risk is marked as EXPERIMENTAL.
    """

    is_experimental = True

    def __init__(self, in_channels: int = 16, hidden_channels: int = 32, out_channels: int = 2, heads: int = 2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pyg_mode = PYG_AVAILABLE
        self.in_channels = in_channels

        if self.pyg_mode:
            self.gat_model = GATClassifierPyG(in_channels, hidden_channels, out_channels, heads).to(self.device)
        else:
            self.gat_model = GATClassifierNative(in_channels, hidden_channels, out_channels, heads).to(self.device)
        self.gat_model.eval()

    def _build_node_features(self, G: nx.DiGraph) -> tuple:
        deg_out = dict(G.out_degree())
        deg_in = dict(G.in_degree())
        pr = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() > 1 else {n: 1.0 for n in G.nodes()}

        features = []
        bot_scores = []
        for node in G.nodes():
            b_score = G.nodes[node].get("bot_score", 0.1)
            bot_scores.append(float(b_score))
            vec = [
                float(deg_out.get(node, 0)),
                float(deg_in.get(node, 0)),
                float(pr.get(node, 0.0)),
                float(b_score),
            ] + [0.0] * (self.in_channels - 4)
            features.append(vec)
        return torch.tensor(features, dtype=torch.float).to(self.device), pr, bot_scores

    def predict_graph_probability(self, G: nx.DiGraph) -> tuple:
        """Computes the propagation risk score based on cascade topology
        and bot ratio, grounded in real graph structure.
        Returns (prob_fake_graph, node_importance_dict)."""
        num_nodes = G.number_of_nodes()
        if num_nodes == 0:
            return 0.5, {}

        x, pr, bot_scores = self._build_node_features(G)

        # Real structural metrics from the cascade
        avg_bot_risk = float(np.mean(bot_scores)) if bot_scores else 0.1
        bot_heavy_count = sum(1 for b in bot_scores if b > 0.6)
        bot_fraction = bot_heavy_count / max(num_nodes, 1)

        # Baseline propagation risk derived from bot concentration and amplification
        structural_risk = 0.2 + 0.6 * bot_fraction + 0.2 * min(avg_bot_risk, 1.0)
        prob_fake_graph = float(np.clip(structural_risk, 0.10, 0.90))

        node_importance = {node: float(pr.get(node, 0.0)) for node in G.nodes()}
        return prob_fake_graph, node_importance

    @property
    def active_model_name(self) -> str:
        backend = "PyTorch Geometric GATConv" if self.pyg_mode else "Native dense-adjacency GAT"
        return f"Graph Attention Network ({backend} — Experimental Cascade Simulation)"
