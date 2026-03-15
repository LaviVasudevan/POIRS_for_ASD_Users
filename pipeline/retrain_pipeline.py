# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # POI Recommendation Pipeline
# ### Organized modular notebook for GAT → TransR → Path Extraction → Path Enrichment → Model Training → Evaluation
# 
# - **Input**: poi_graph_v6.pt
# - **Output**: Trained models + evaluation metrics

# %% [markdown]
# ## Phase 0: Setup & Dependencies

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 1: ENVIRONMENT SETUP & INSTALLATIONS
# ============================================================================

%pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python3 -c "import torch; print(torch.__version__.split('+')[0])")+cu124.html -q
%pip install torch-geometric -q

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 2: IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor  # ADD THIS LINE
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.loader import LinkNeighborLoader
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import random
import gc
from collections import deque, defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Union, Callable
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# %% [code]
# ============================================================================
# CELL 3: SPLIT GRAPH
# ============================================================================

import torch
import random
from torch_geometric.data import HeteroData

def split_user_poi_edges(data: HeteroData, train_ratio=0.8, val_ratio=0.1, seed=42):
    # Fix random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Relations to split (forward ↔ reverse)
    relations = [
        (('user', 'rates', 'poi'), ('poi', 'rev_rates', 'user')),
        (('user', 'visits', 'poi'), ('poi', 'rev_visits', 'user')),
        (('user', 'prefers', 'category'), ('category', 'rev_prefers', 'user')),
    ]
    
    # Helper to split one relation
    def split_relation(rel, rev_rel):
        edge_index = data[rel].edge_index
        edge_attr = data[rel].edge_attr if hasattr(data[rel], 'edge_attr') else None
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)
        
        # Calculate split sizes
        train_end = int(train_ratio * num_edges)
        val_end   = int((train_ratio + val_ratio) * num_edges)
        
        train_idx = perm[:train_end]
        val_idx   = perm[train_end:val_end]
        test_idx  = perm[val_end:]
        
        # Helper: clone graph and assign a split of edges
        def make_split(idx):
            split_data = data.clone()
            # Forward edges
            split_data[rel].edge_index = edge_index[:, idx]
            if edge_attr is not None:
                split_data[rel].edge_attr = edge_attr[idx]
            
            # Reverse edges
            split_data[rev_rel].edge_index = torch.stack(
                [edge_index[1, idx], edge_index[0, idx]], dim=0
            )
            if edge_attr is not None:
                split_data[rev_rel].edge_attr = edge_attr[idx]
            return split_data
        
        return make_split(train_idx), make_split(val_idx), make_split(test_idx)
    
    # Apply splitting for "rates", "visits", and "prefers"
    train_rates, val_rates, test_rates = split_relation(
        ('user', 'rates', 'poi'), ('poi', 'rev_rates', 'user')
    )
    train_visits, val_visits, test_visits = split_relation(
        ('user', 'visits', 'poi'), ('poi', 'rev_visits', 'user')
    )
    train_prefers, val_prefers, test_prefers = split_relation(
        ('user', 'prefers', 'category'), ('category', 'rev_prefers', 'user')
    )
    
    # Merge splits into one HeteroData per split
    def merge_splits(split1, split2, split3):
        merged = data.clone()
        
        # Update rates - assign individual attributes instead of entire store
        merged[('user', 'rates', 'poi')].edge_index = split1[('user', 'rates', 'poi')].edge_index
        if hasattr(split1[('user', 'rates', 'poi')], 'edge_attr'):
            merged[('user', 'rates', 'poi')].edge_attr = split1[('user', 'rates', 'poi')].edge_attr
        
        merged[('poi', 'rev_rates', 'user')].edge_index = split1[('poi', 'rev_rates', 'user')].edge_index
        if hasattr(split1[('poi', 'rev_rates', 'user')], 'edge_attr'):
            merged[('poi', 'rev_rates', 'user')].edge_attr = split1[('poi', 'rev_rates', 'user')].edge_attr
        
        # Update visits - assign individual attributes instead of entire store
        merged[('user', 'visits', 'poi')].edge_index = split2[('user', 'visits', 'poi')].edge_index
        if hasattr(split2[('user', 'visits', 'poi')], 'edge_attr'):
            merged[('user', 'visits', 'poi')].edge_attr = split2[('user', 'visits', 'poi')].edge_attr
            
        merged[('poi', 'rev_visits', 'user')].edge_index = split2[('poi', 'rev_visits', 'user')].edge_index
        if hasattr(split2[('poi', 'rev_visits', 'user')], 'edge_attr'):
            merged[('poi', 'rev_visits', 'user')].edge_attr = split2[('poi', 'rev_visits', 'user')].edge_attr
        
        # Update prefers - assign individual attributes instead of entire store
        merged[('user', 'prefers', 'category')].edge_index = split3[('user', 'prefers', 'category')].edge_index
        if hasattr(split3[('user', 'prefers', 'category')], 'edge_attr'):
            merged[('user', 'prefers', 'category')].edge_attr = split3[('user', 'prefers', 'category')].edge_attr
            
        merged[('category', 'rev_prefers', 'user')].edge_index = split3[('category', 'rev_prefers', 'user')].edge_index
        if hasattr(split3[('category', 'rev_prefers', 'user')], 'edge_attr'):
            merged[('category', 'rev_prefers', 'user')].edge_attr = split3[('category', 'rev_prefers', 'user')].edge_attr
        
        return merged
    
    train_data = merge_splits(train_rates, train_visits, train_prefers)
    val_data   = merge_splits(val_rates, val_visits, val_prefers)
    test_data  = merge_splits(test_rates, test_visits, test_prefers)
    
    return train_data, val_data, test_data

# %% [markdown]
# ### Configuration
# Central config for all pipeline parameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 4: CONFIGURATION
# ============================================================================
class Config:
    """Central configuration for the pipeline"""
    
    # Paths
    TRAIN_GRAPH = "/kaggle/working/train_graph.pt"
    VAL_GRAPH = "/kaggle/working/val_graph.pt"
    TEST_GRAPH = "/kaggle/working/test_graph.pt"
    OUTPUT_DIR = "./output"
    
    # GAT Configuration
    GAT_HIDDEN_DIM = 32
    GAT_OUT_DIM = 64
    GAT_NUM_LAYERS = 2
    GAT_HEADS = 2
    GAT_DROPOUT = 0.2
    GAT_LR = 5e-4
    GAT_EPOCHS = 5
    GAT_BATCH_SIZE = 1024
    
    # TransR Configuration
    TRANSR_EMB_DIM = 100
    TRANSR_LR = 1e-3
    TRANSR_EPOCHS = 5
    TRANSR_BATCH_SIZE = 1024
    TRANSR_NUM_NEGATIVES = 5
    
    # Path Filtering
    MAX_PATH_LENGTH = 3
    FILTER_STRATEGY = "topk"
    FILTER_K = 5
    AGGREGATION = "mean"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

config = Config()
print(f"✓ Configuration loaded | Device: {config.DEVICE}")

# %% [markdown]
# ### Data Loading Utilities
# Functions to load heterogeneous graphs and extract node counts

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 5: DATA LOADING UTILITIES
# ============================================================================
def load_hetero_graph(path: str) -> HeteroData:
    """Load heterogeneous graph from .pt file"""
    print(f"[Load] Loading graph from {path}")
    loaded = torch.load(path, weights_only=False)
    
    if isinstance(loaded, dict):
        data = loaded.get('graph', loaded.get('data', loaded))
        metadata = loaded.get('metadata', {})   
        train_data, val_data, test_data = split_user_poi_edges(data)
        torch.save(train_data, "train_graph.pt")
        torch.save(val_data, "val_graph.pt")
        torch.save(test_data, "test_graph.pt")
    
    else:
        data = loaded
        metadata = {}
    
    print(f"[Load] Node types: {list(data.node_types)}")
    print(f"[Load] Edge types: {list(data.edge_types)}")
    
    return data, metadata

def get_num_nodes(data: HeteroData, node_type: str) -> int:
    """Get number of nodes for a specific type"""
    return int(data[node_type].num_nodes)

# %% [markdown]
# ### Graph Utilities
# Convert HeteroData to triples format for TransR training

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 6: GRAPH UTILITIES
# ============================================================================
def extract_hetero_triples(
    data: HeteroData, 
    entity2id: dict = None, 
    relation2id: dict = None,
    include_node_type: bool = True
) -> Tuple[torch.Tensor, dict, dict]:
    """
    Convert HeteroData graph into triples (h, r, t) format.
    
    Returns:
        triples: [num_edges, 3] tensor
        entity2id: mapping of entities to IDs
        relation2id: mapping of relations to IDs
    """
    triples = []
    if entity2id is None:
        entity2id = {}
    if relation2id is None:
        relation2id = {}
    
    entity_cnt = len(entity2id)
    relation_cnt = len(relation2id)
    
    print(f"[Triples] Processing {len(data.edge_types)} edge types")
    
    for (src, rel, dst) in data.edge_types:
        edge_index = data[(src, rel, dst)].edge_index
        num_edges = edge_index.size(1)
        
        if rel not in relation2id:
            relation2id[rel] = relation_cnt
            relation_cnt += 1
        
        for h, t in edge_index.t().tolist():
            src_key = (src, h) if include_node_type else h
            dst_key = (dst, t) if include_node_type else t
            
            if src_key not in entity2id:
                entity2id[src_key] = entity_cnt
                entity_cnt += 1
            if dst_key not in entity2id:
                entity2id[dst_key] = entity_cnt
                entity_cnt += 1
            
            triples.append((entity2id[src_key], relation2id[rel], entity2id[dst_key]))
    
    print(f"[Triples] Total: {len(triples)} | Entities: {len(entity2id)} | Relations: {len(relation2id)}")
    
    return torch.tensor(triples, dtype=torch.long), entity2id, relation2id

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Phase 1: Graph Neural Networks & Path Extraction
# 
# ### Step 1.1: Graph Attention Network (GAT)
# Generates node embeddings for users, POIs, and categories

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 7: GAT MODEL DEFINITION
# ============================================================================
class HeteroGAT(nn.Module):
    """Heterogeneous Graph Attention Network"""
    
    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = 32,
        out_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        activation=F.elu,
        init_embeddings: dict = None,
    ):
        super().__init__()
        self.node_types = list(data.node_types)
        self.edge_types = list(data.edge_types)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.activation = activation
        
        # Input projections/embeddings
        self.input_projections = nn.ModuleDict()
        self.input_embeddings = nn.ModuleDict()
        
        for ntype in self.node_types:
            if hasattr(data[ntype], 'x') and data[ntype].x is not None:
                original_dim = int(data[ntype].x.size(-1))
                self.input_projections[ntype] = nn.Linear(original_dim, hidden_dim)
            else:
                n_nodes = get_num_nodes(data, ntype)
                emb = nn.Embedding(n_nodes, hidden_dim)
                
                # Initialize with provided embeddings if available
                if init_embeddings is not None and ntype in init_embeddings:
                    init_emb = init_embeddings[ntype]
                    if init_emb.size(1) != hidden_dim:
                        projection = nn.Linear(init_emb.size(1), hidden_dim, bias=False)
                        nn.init.xavier_uniform_(projection.weight)
                        with torch.no_grad():
                            projected = projection(init_emb)
                            projected = F.normalize(projected, p=2, dim=-1)
                            emb.weight.data[:init_emb.size(0)] = projected
                    else:
                        with torch.no_grad():
                            emb.weight.data[:init_emb.size(0)] = init_emb
                            emb.weight.data = F.normalize(emb.weight.data, p=2, dim=-1)
                else:
                    nn.init.xavier_uniform_(emb.weight)
                    emb.weight.data = F.normalize(emb.weight.data, p=2, dim=-1)
                
                self.input_embeddings[ntype] = emb
                self.input_projections[ntype] = nn.Identity()
        
        # GAT layers
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict = {}
            in_dim = hidden_dim if layer_idx == 0 else hidden_dim * heads
            for edge_type in self.edge_types:
                gat = GATConv(
                    in_channels=(in_dim, in_dim),
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True
                )
                conv_dict[edge_type] = gat
            self.layers.append(HeteroConv(conv_dict, aggr="mean"))
        
        # Final projection
        self.final_linears = nn.ModuleDict()
        final_in_dim = hidden_dim * heads if num_layers > 0 else hidden_dim
        for ntype in self.node_types:
            self.final_linears[ntype] = nn.Linear(final_in_dim, out_dim)
            nn.init.xavier_uniform_(self.final_linears[ntype].weight)
    
    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        x_dict = {}
        
        # Process node features
        for ntype in self.node_types:
            if hasattr(data[ntype], 'x') and data[ntype].x is not None:
                x = data[ntype].x.float().to(device)
                x_dict[ntype] = self.input_projections[ntype](x)
            elif ntype in self.input_embeddings:
                node_indices = torch.arange(data[ntype].num_nodes, device=device, dtype=torch.long)
                x = self.input_embeddings[ntype](node_indices)
                x_dict[ntype] = self.input_projections[ntype](x)
        
        # GAT layers
        current_x_dict = x_dict
        for layer in self.layers:
            current_x_dict = layer(current_x_dict, data.edge_index_dict)
            for ntype in current_x_dict:
                if current_x_dict[ntype] is not None:
                    h = self.activation(current_x_dict[ntype])
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    current_x_dict[ntype] = h
        
        # Final projection
        output_dict = {}
        for ntype in self.node_types:
            if ntype in current_x_dict and current_x_dict[ntype] is not None:
                output_dict[ntype] = self.final_linears[ntype](current_x_dict[ntype])
        
        return output_dict

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 8: GAT TRAINING UTILITIES
# ============================================================================
class GATTrainer:
    """Trainer for GAT model"""
    
    def __init__(self, model, data, lr=1e-3, weight_decay=0.0, device=None):
        self.model = model
        self.data = data.cpu()
        self.device = device if device is not None else torch.device('cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_epoch(self, batch_size=1024) -> float:
        self.model.train()
        losses = []
        
        for edge_type in self.data.edge_types:
            edge_index = self.data.edge_index_dict[edge_type]
            if edge_index.size(1) == 0:
                continue
            
            loader = LinkNeighborLoader(
                self.data,
                num_neighbors=[10, 5],
                edge_label_index=(edge_type, edge_index),
                batch_size=batch_size,
                shuffle=True,
                neg_sampling_ratio=1.0,
            )
            
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                emb_dict = self.model(batch)
                
                # Compute loss (simplified)
                src_type, _, dst_type = edge_type
                src_idx = batch[edge_type].edge_label_index[0]
                dst_idx = batch[edge_type].edge_label_index[1]
                labels = batch[edge_type].edge_label.float()
                
                src_emb = emb_dict[src_type][src_idx]
                dst_emb = emb_dict[dst_type][dst_idx]
                
                scores = (src_emb * dst_emb).sum(dim=-1)
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        
        return np.mean(losses) if losses else 0.0
    
    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            data_on_dev = self.data.to(self.device)
            emb = self.model(data_on_dev)
            return {nt: e.cpu() for nt, e in emb.items()}

def train_gat(
    train_graph_path: str,
    output_path: str,
    hidden_dim: int = 32,
    out_dim: int = 64,
    num_layers: int = 2,
    heads: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 5,
    batch_size: int = 1024,
    device: str = None,
    init_embeddings: dict = None
):
    """Complete GAT training pipeline"""
    print(f"[GAT] Starting training...")
    
    # Load data
    data, metadata = load_hetero_graph(train_graph_path)
    
    # Build model
    model = HeteroGAT(
        data=data,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
        init_embeddings=init_embeddings
    )
    
    # Train
    trainer = GATTrainer(model, data, lr=lr, device=device)
    
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(batch_size=batch_size)
        print(f"[GAT] Epoch {epoch}/{epochs} | Loss: {loss:.4f}")
    
    # Save embeddings
    embeddings = trainer.get_embeddings()
    torch.save(embeddings, output_path)
    print(f"[GAT] Embeddings saved to {output_path}")
    
    return model, embeddings

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Step 1.2: TransR Knowledge Graph Embeddings
# Learns relation-specific projections for user-POI interactions

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 9: TransR MODEL DEFINITION
# ============================================================================
class TransR(nn.Module):
    """TransR knowledge graph embedding model"""
    
    def __init__(
        self, 
        num_entities: int, 
        num_relations: int, 
        emb_dim: int = 100,
        p_norm: int = 2,
        margin: float = None
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.p_norm = p_norm
        self.margin = margin
        
        # Embeddings
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        self.rel_proj = nn.Parameter(torch.randn(num_relations, emb_dim, emb_dim))
        
        # Initialization
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        nn.init.xavier_uniform_(self.rel_proj)
    
    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute TransR score"""
        h_e = self.entity_emb(h)
        t_e = self.entity_emb(t)
        r_e = self.rel_emb(r)
        M_r = self.rel_proj[r]
        
        # Project entities
        h_r = torch.bmm(h_e.unsqueeze(1), M_r).squeeze(1)
        t_r = torch.bmm(t_e.unsqueeze(1), M_r).squeeze(1)
        
        # Distance
        distance = torch.norm(h_r + r_e - t_r, p=self.p_norm, dim=1)
        
        if self.margin is not None:
            return self.margin - distance
        return -distance
    
    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.score(h, r, t)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 10: ENHANCED NEGATIVE SAMPLING
# ============================================================================

# Global cache for graph negatives (load once, reuse)
_graph_negatives_cache = None

def negative_sampling_with_graph_negatives(
    triples: torch.Tensor,
    num_entities: int,
    device: torch.device,
    hetero_graph_path: str = None,
    edge_types_with_ratings: list = None,
    negative_threshold: float = 3.0,
    entity2id: dict = None,
    relation2id: dict = None,
    num_negatives: int = 1,
    mode: str = "both",
    graph_neg_ratio: float = 0.3
):
    """
    Generate negatives combining:
    1. Actual low-rating edges from graph (rating < threshold)
    2. Random corrupted triples (head/tail corruption)
    
    Args:
        graph_neg_ratio: Ratio of graph negatives (0.0-1.0)
            e.g., 0.3 = 30% from graph, 70% random
    """
    global _graph_negatives_cache
    
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    batch_size = len(h)
    
    # Default edge types with ratings
    if edge_types_with_ratings is None:
        edge_types_with_ratings = [
            ('user', 'rates', 'poi'),
            ('poi', 'rev_rates', 'user')
        ]
    
    # Load graph negatives once (cached)
    if _graph_negatives_cache is None and hetero_graph_path is not None:
        _graph_negatives_cache = []
        try:
            loaded = torch.load(hetero_graph_path, weights_only=False)
            data = loaded.get('graph', loaded) if isinstance(loaded, dict) else loaded
            
            print(f"[NegSampling] Extracting graph negatives from: {hetero_graph_path}")
            
            for edge_type in edge_types_with_ratings:
                if edge_type not in data.edge_types:
                    continue
                
                edge_index = data[edge_type].edge_index
                
                if not hasattr(data[edge_type], 'edge_attr') or data[edge_type].edge_attr is None:
                    continue
                
                edge_attr = data[edge_type].edge_attr
                ratings = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
                
                # Find low ratings
                negative_mask = ratings < negative_threshold
                num_neg = negative_mask.sum().item()
                
                if num_neg > 0:
                    neg_heads = edge_index[0, negative_mask].tolist()
                    neg_tails = edge_index[1, negative_mask].tolist()
                    
                    src_type, rel_name, dst_type = edge_type
                    rel_id = relation2id.get(rel_name, 0) if relation2id else 0
                    
                    for h_idx, t_idx in zip(neg_heads, neg_tails):
                        if entity2id:
                            h_global = entity2id.get((src_type, h_idx), h_idx)
                            t_global = entity2id.get((dst_type, t_idx), t_idx)
                        else:
                            h_global, t_global = h_idx, t_idx
                        
                        _graph_negatives_cache.append([h_global, rel_id, t_global])
                    
                    print(f"[NegSampling] Found {num_neg} negatives from {edge_type}")
            
            print(f"[NegSampling] Total cached: {len(_graph_negatives_cache)} negatives")
        
        except Exception as e:
            print(f"[NegSampling] Error: {e}")
            _graph_negatives_cache = []
    
    # Calculate split
    num_from_graph = int(num_negatives * graph_neg_ratio) if _graph_negatives_cache else 0
    num_random = num_negatives - num_from_graph
    
    neg_triples_list = []
    
    # 1. Sample from graph negatives
    if num_from_graph > 0:
        graph_neg_tensor = torch.tensor(_graph_negatives_cache, dtype=torch.long, device=device)
        for _ in range(num_from_graph):
            indices = torch.randint(0, len(graph_neg_tensor), (batch_size,), device=device)
            neg_triples_list.append(graph_neg_tensor[indices])
    
    # 2. Generate random negatives
    for _ in range(num_random):
        if mode == "both":
            mask = torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=device)
            neg_h = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_t = torch.randint(0, num_entities, (batch_size,), device=device)
            neg_h = torch.where(mask, neg_h, h)
            neg_t = torch.where(~mask, neg_t, t)
            neg_r = r
        else:
            raise ValueError("Only 'both' mode supported")
        
        neg_triples_list.append(torch.stack([neg_h, neg_r, neg_t], dim=1))
    
    return torch.cat(neg_triples_list, dim=0) if neg_triples_list else torch.zeros((0, 3), dtype=torch.long, device=device)

def train_transr(
    hetero_graph_path: str,
    output_dir: str,
    emb_dim: int = 100,
    lr: float = 1e-2,
    epochs: int = 5,
    batch_size: int = 2048,
    num_negatives: int = 2,
    device: str = None,
    gat_embeddings: dict = None
):
    """Complete TransR training pipeline with optional GAT initialization"""
    print(f"[TransR] Starting training...")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load graph and extract triples
    data, _ = load_hetero_graph(hetero_graph_path)
    all_triples, entity2id, relation2id = extract_hetero_triples(data)
    
    # Split data
    num_triples = len(all_triples)
    train_size = int(0.8 * num_triples)
    perm = torch.randperm(num_triples)
    train_triples = all_triples[perm[:train_size]]
    val_triples = all_triples[perm[train_size:]]
    
    # Create model
    num_entities, num_relations = len(entity2id), len(relation2id)
    model = TransR(num_entities, num_relations, emb_dim=emb_dim).to(dev)
    
    # Initialize with GAT embeddings if provided
    if gat_embeddings is not None:
        print("[TransR] Initializing with GAT embeddings...")
        with torch.no_grad():
            for (node_type, local_id), global_id in entity2id.items():
                if node_type in gat_embeddings:
                    gat_emb = gat_embeddings[node_type]
                    if local_id < gat_emb.size(0):
                        gat_vec = gat_emb[local_id].to(dev)
                        if gat_vec.size(0) == emb_dim:
                            model.entity_emb.weight.data[global_id] = gat_vec
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_triples, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        
        for batch in train_loader:
            batch = batch.to(dev)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            
            # Positive scores
            pos_score = model.score(h, r, t)
            
            # Negative samples
            neg_batch = negative_sampling_with_graph_negatives(
                triples=batch,
                num_entities=num_entities,
                device=dev,
                hetero_graph_path=hetero_graph_path,  # Pass graph path
                edge_types_with_ratings=[
                    ('user', 'rates', 'poi'),
                    ('poi', 'rev_rates', 'user')
                ],
                entity2id=entity2id,
                relation2id=relation2id,
                num_negatives=num_negatives,
                mode="both",
                graph_neg_ratio=0.3  # 30% from graph, 70% random
            )
            nh, nr, nt = neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2]
            neg_score = model.score(nh, nr, nt)
            
            # Loss
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / count
        print(f"[TransR] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")
    
    # Save model
    final_path = os.path.join(output_dir, "transr_final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "entity2id": entity2id,
        "relation2id": relation2id,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "emb_dim": emb_dim
    }, final_path)
    
    print(f"[TransR] Training completed! Saved to {final_path}")
    return model, entity2id, relation2id

# %% [markdown]
# ### Step 1.3: Path Extraction
# Extract paths between users and POIs (max length = 3)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 11: PATH EXTRACTION UTILITIES
# ============================================================================
def extract_paths_from_hetero(
    hetero_graph: HeteroData,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    source_node_type: str,
    target_node_type: str,
    max_length: int = 3,
    max_paths: int = 100000
) -> List[Tuple[List[int], List[int]]]:
    """Extract paths from heterogeneous graph"""
    print(f"[PathExtract] Extracting paths: {source_node_type} → {target_node_type}")
    
    # Build adjacency list
    adj_list = {}
    for edge_type in hetero_graph.edge_types:
        src_type, rel_name, dst_type = edge_type
        
        if rel_name not in relation2id:
            continue
        
        rel_id = relation2id[rel_name]
        edge_index = hetero_graph[edge_type].edge_index
        
        for i in range(edge_index.shape[1]):
            src_local = edge_index[0, i].item()
            dst_local = edge_index[1, i].item()
            
            src_global = entity2id.get((src_type, src_local))
            dst_global = entity2id.get((dst_type, dst_local))
            
            if src_global is None or dst_global is None:
                continue
            
            src_node = (src_type, src_global)
            if src_node not in adj_list:
                adj_list[src_node] = []
            adj_list[src_node].append((rel_id, dst_type, dst_global))
    
    # BFS to find paths
    paths = []
    source_nodes = [entity2id[k] for k in entity2id.keys() 
                   if isinstance(k, tuple) and k[0] == source_node_type]
    
    for start_node in source_nodes:
        if len(paths) >= max_paths:
            break
        
        queue = deque([(source_node_type, start_node, [start_node], [])])
        
        while queue and len(paths) < max_paths:
            curr_type, curr_node, path_nodes, path_rels = queue.popleft()
            
            if len(path_rels) >= max_length:
                continue
            
            curr_node_key = (curr_type, curr_node)
            for rel_id, next_type, next_node in adj_list.get(curr_node_key, []):
                new_path_nodes = path_nodes + [next_node]
                new_path_rels = path_rels + [rel_id]
                
                if next_type == target_node_type:
                    paths.append((new_path_nodes.copy(), new_path_rels.copy()))
                    if len(paths) >= max_paths:
                        break
                
                if len(new_path_rels) < max_length:
                    queue.append((next_type, next_node, new_path_nodes, new_path_rels))
    
    print(f"[PathExtract] Found {len(paths)} paths")
    return paths

# %% [code] {"jupyter":{"outputs_hidden":false}}



# ============================================================================
# CELL 12: OPTIMIZED PATH EXTRACTION WITH LIMITS
# ============================================================================

def extract_paths_from_hetero_optimized(
    hetero_graph: HeteroData,
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    source_node_type: str,
    target_node_type: str,
    max_length: int = 3,
    max_paths: int = 100000,  # NEW: Total path limit
    max_paths_per_source: int = 1000,  # NEW: Per-source limit
    batch_size: int = 50,
    enable_cycle_detection: bool = True  # NEW: Skip revisited nodes
) -> List[Tuple[List[int], List[int]]]:
    """
    Memory-optimized path extraction with:
    - Total path limit
    - Per-source path limit  
    - Cycle detection (no revisiting nodes)
    - Early stopping when limits reached
    """
    print(f"[PathExtract] OPTIMIZED extraction: {source_node_type} → {target_node_type}")
    print(f"[PathExtract] Max total paths: {max_paths:,}")
    print(f"[PathExtract] Max per source: {max_paths_per_source:,}")
    print(f"[PathExtract] Cycle detection: {enable_cycle_detection}")
    
    if len(entity2id) == 0:
        return []
    
    sample_key = next(iter(entity2id.keys()))
    uses_string_keys = isinstance(sample_key, str)
    
    # Get source nodes
    source_nodes = []
    for key in entity2id.keys():
        if uses_string_keys:
            if isinstance(key, str) and key.startswith(f"{source_node_type}_"):
                source_nodes.append(entity2id[key])
        else:
            if isinstance(key, tuple) and len(key) >= 2 and key[0] == source_node_type:
                source_nodes.append(entity2id[key])
    
    print(f"[PathExtract] Found {len(source_nodes)} source nodes")
    
    # Build adjacency list
    adj_list = {}
    for edge_type in hetero_graph.edge_types:
        src_type, rel_name, dst_type = edge_type
        
        if rel_name not in relation2id:
            continue
        
        rel_id = relation2id[rel_name]
        edge_index = hetero_graph[edge_type].edge_index
        
        for i in range(edge_index.shape[1]):
            src_local = edge_index[0, i].item()
            dst_local = edge_index[1, i].item()
            
            if uses_string_keys:
                src_global = entity2id.get(f"{src_type}_{src_local}")
                dst_global = entity2id.get(f"{dst_type}_{dst_local}")
            else:
                src_global = entity2id.get((src_type, src_local))
                dst_global = entity2id.get((dst_type, dst_local))
            
            if src_global is None or dst_global is None:
                continue
            
            src_node = (src_type, src_global)
            if src_node not in adj_list:
                adj_list[src_node] = []
            adj_list[src_node].append((rel_id, dst_type, dst_global))
    
    print(f"[PathExtract] Adjacency list: {len(adj_list)} nodes")
    
    # BFS with limits
    all_paths = []
    
    for idx, start_node in enumerate(source_nodes):
        if len(all_paths) >= max_paths:
            print(f"[PathExtract] Reached total path limit at source {idx}/{len(source_nodes)}")
            break
        
        if idx % 100 == 0 and idx > 0:
            print(f"[PathExtract] Progress: {idx}/{len(source_nodes)} sources | {len(all_paths)} paths")
        
        queue = deque([(source_node_type, start_node, [start_node], [])])
        paths_from_source = 0
        
        while queue and paths_from_source < max_paths_per_source:
            curr_type, curr_node, path_nodes, path_rels = queue.popleft()
            
            if len(path_rels) >= max_length:
                continue
            
            curr_node_key = (curr_type, curr_node)
            
            for rel_id, next_type, next_node in adj_list.get(curr_node_key, []):
                # CYCLE DETECTION: Skip if node already in path
                if enable_cycle_detection and next_node in path_nodes:
                    continue
                
                new_path_nodes = path_nodes + [next_node]
                new_path_rels = path_rels + [rel_id]
                
                # Found target
                if next_type == target_node_type:
                    all_paths.append((new_path_nodes.copy(), new_path_rels.copy()))
                    paths_from_source += 1
                    
                    # Check limits
                    if len(all_paths) >= max_paths or paths_from_source >= max_paths_per_source:
                        break
                
                # Continue exploring
                if len(new_path_rels) < max_length:
                    queue.append((next_type, next_node, new_path_nodes, new_path_rels))
            
            if len(all_paths) >= max_paths:
                break
    
    print(f"[PathExtract] Extracted {len(all_paths)} total paths")
    return all_paths

# %% [markdown]
# ### 
# 
# Step 1.4: Path Scoring
# Score extracted paths using trained TransR model

# %% [code] {"jupyter":{"outputs_hidden":false}}







# ============================================================================
# CELL 13: PATH SCORING
# ============================================================================
def score_paths_with_transr(
    paths: List[Tuple[List[int], List[int]]],
    model,
    device: str = 'cpu',
    aggregation: str = 'mean',
    batch_size: int = 10000
) -> np.ndarray:
    """Score paths using TransR model"""
    print(f"[PathScore] Scoring {len(paths)} paths...")
    
    if len(paths) == 0:
        return np.array([])
    
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    
    # Collect unique edges
    unique_edges = set()
    for nodes, relations in paths:
        for i in range(len(relations)):
            edge = (nodes[i], relations[i], nodes[i + 1])
            unique_edges.add(edge)
    
    unique_edges = list(unique_edges)
    print(f"[PathScore] Found {len(unique_edges)} unique edges")
    
    # Score all unique edges
    edge_score_cache = {}
    
    with torch.no_grad():
        for batch_start in range(0, len(unique_edges), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_edges))
            batch_edges = unique_edges[batch_start:batch_end]
            
            h_batch = torch.tensor([e[0] for e in batch_edges], device=dev)
            r_batch = torch.tensor([e[1] for e in batch_edges], device=dev)
            t_batch = torch.tensor([e[2] for e in batch_edges], device=dev)
            
            scores_batch = model.score(h_batch, r_batch, t_batch)
            
            for edge, score in zip(batch_edges, scores_batch.cpu().numpy()):
                edge_score_cache[edge] = float(score)
    
    # Compute path scores
    path_scores = []
    for nodes, relations in paths:
        edge_scores = []
        for i in range(len(relations)):
            edge = (nodes[i], relations[i], nodes[i + 1])
            edge_scores.append(edge_score_cache[edge])
        
        edge_scores = np.array(edge_scores)
        if aggregation == 'mean':
            path_score = edge_scores.mean()
        elif aggregation == 'sum':
            path_score = edge_scores.sum()
        elif aggregation == 'min':
            path_score = edge_scores.min()
        elif aggregation == 'max':
            path_score = edge_scores.max()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        path_scores.append(path_score)
    
    scores = np.array(path_scores)
    print(f"[PathScore] Stats - min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
    
    return scores

# %% [markdown]
# ### 
# 
# 
# 
# Step 1.5: Path Filtering
# Filter paths using top-k, threshold, percentile, z-score, or softmax strategies

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 14: ENHANCED PATH FILTERING
# ============================================================================

def filter_paths_per_pair(
    paths: List[Tuple[List[int], List[int]]],
    scores: np.ndarray,
    method: str = 'topk',
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    percentile: Optional[float] = None,
    z_score_threshold: Optional[float] = None,  # NEW
    softmax_threshold: Optional[float] = None,  # NEW
    reverse: bool = False
) -> Tuple[List[Tuple[List[int], List[int]]], np.ndarray, List[int]]:
    """
    Enhanced filtering with z-score and softmax methods.
    """
    print(f"[PathFilter] Filtering using method: {method}")
    
    if len(paths) == 0:
        return [], np.array([]), []
    
    # Group by (source, target) pair
    pair_paths = defaultdict(list)
    for idx, (nodes, rels) in enumerate(paths):
        pair_key = (nodes[0], nodes[-1])
        pair_paths[pair_key].append((idx, scores[idx], nodes, rels))
    
    print(f"[PathFilter] Found {len(pair_paths)} unique pairs")
    
    filtered_indices = []
    
    if method == 'topk':
        if k is None:
            raise ValueError("Parameter 'k' required for topk")
        
        for pair_key, pair_data in pair_paths.items():
            pair_data.sort(key=lambda x: x[1], reverse=not reverse)
            for idx, _, _, _ in pair_data[:k]:
                filtered_indices.append(idx)
        
        print(f"[PathFilter] Keeping top {k} per pair")
    
    elif method == 'threshold':
        if threshold is None:
            raise ValueError("Parameter 'threshold' required")
        
        for pair_key, pair_data in pair_paths.items():
            for idx, score, _, _ in pair_data:
                if (reverse and score <= threshold) or (not reverse and score >= threshold):
                    filtered_indices.append(idx)
        
        print(f"[PathFilter] Threshold: {'<=' if reverse else '>='} {threshold}")
    
    elif method == 'percentile':
        if percentile is None:
            raise ValueError("Parameter 'percentile' required")
        
        threshold_value = np.percentile(scores, percentile)
        
        for idx, score in enumerate(scores):
            if (reverse and score <= threshold_value) or (not reverse and score >= threshold_value):
                filtered_indices.append(idx)
        
        print(f"[PathFilter] {percentile}th percentile = {threshold_value:.4f}")
    
    # NEW: Z-SCORE FILTERING
    elif method == 'z_score':
        if z_score_threshold is None:
            raise ValueError("Parameter 'z_score_threshold' required")
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        if std_score == 0:
            print("[PathFilter] Warning: All scores identical, keeping all")
            filtered_indices = list(range(len(paths)))
        else:
            z_scores = (scores - mean_score) / std_score
            
            for idx in range(len(paths)):
                if (reverse and z_scores[idx] <= -z_score_threshold) or \
                   (not reverse and z_scores[idx] >= z_score_threshold):
                    filtered_indices.append(idx)
        
        print(f"[PathFilter] Z-score threshold: {z_score_threshold}")
    
    # NEW: SOFTMAX FILTERING
    elif method == 'softmax':
        if softmax_threshold is None:
            softmax_threshold = 0.01
        
        print(f"[PathFilter] Softmax threshold: {softmax_threshold}")
        
        for pair_key, pair_data in pair_paths.items():
            pair_scores = np.array([score for _, score, _, _ in pair_data])
            
            # Negate if reverse (distance-based)
            if reverse:
                pair_scores = -pair_scores
            
            # Softmax normalization
            exp_scores = np.exp(pair_scores - np.max(pair_scores))
            softmax_probs = exp_scores / np.sum(exp_scores)
            
            # Keep paths above threshold
            for i, (idx, _, _, _) in enumerate(pair_data):
                if softmax_probs[i] > softmax_threshold:
                    filtered_indices.append(idx)
        
        print(f"[PathFilter] Kept paths with prob > {softmax_threshold}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Sort and extract
    filtered_indices.sort()
    filtered_paths = [paths[i] for i in filtered_indices]
    filtered_scores = scores[filtered_indices]
    
    print(f"[PathFilter] Filtered to {len(filtered_paths)} paths")
    
    return filtered_paths, filtered_scores, filtered_indices

# %% [code] {"jupyter":{"outputs_hidden":false}}

# ============================================================================
# CELL 15: PATH ANALYSIS & VISUALIZATION
# ============================================================================
def analyze_paths(
    paths: List[Tuple[List[int], List[int]]],
    entity2id: dict,
    relation2id: dict,
    title: str = "Path Analysis"
):
    """Analyze and print path statistics"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if len(paths) == 0:
        print("No paths found!")
        return
    
    # Path length distribution
    length_dist = Counter([len(rels) for _, rels in paths])
    print(f"\nPath Length Distribution:")
    for length in sorted(length_dist.keys()):
        print(f"  Length {length}: {length_dist[length]} paths")
    
    # Paths per pair
    pair_counts = Counter()
    for nodes, _ in paths:
        pair_counts[(nodes[0], nodes[-1])] += 1
    
    print(f"\nTotal unique (source, target) pairs: {len(pair_counts)}")
    if len(pair_counts) > 0:
        print(f"Average paths per pair: {sum(pair_counts.values()) / len(pair_counts):.2f}")
        print(f"Max paths for a pair: {max(pair_counts.values())}")
        print(f"Min paths for a pair: {min(pair_counts.values())}")

def print_sample_paths(
    paths: List[Tuple[List[int], List[int]]],
    entity2id: dict,
    relation2id: dict,
    metadata: dict = None,
    num_samples: int = 5
):
    """Print sample paths with readable names"""
    print(f"\n{'='*60}")
    print(f"Sample Paths (showing {num_samples})")
    print(f"{'='*60}")
    
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    
    for i, (nodes, rels) in enumerate(paths[:num_samples]):
        path_str = str(id2entity.get(nodes[0], f"E{nodes[0]}"))
        for j, rel in enumerate(rels):
            rel_name = id2relation.get(rel, f'R{rel}')
            path_str += f" -[{rel_name}]-> {id2entity.get(nodes[j+1], f'E{nodes[j+1]}')}"
        print(f"  {i+1}. {path_str}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 16: SAVING UTILITIES
# ============================================================================
def save_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_path: str
):
    """Save embeddings to file"""
    print(f"[Save] Saving embeddings to {output_path}")
    torch.save(embeddings, output_path)
    print(f"[Save] Saved {len(embeddings)} node types")

def save_model_checkpoint(
    model,
    entity2id: dict,
    relation2id: dict,
    output_path: str,
    additional_info: dict = None
):
    """Save model checkpoint"""
    print(f"[Save] Saving model checkpoint to {output_path}")
    
    checkpoint = {
        "model_state": model.state_dict(),
        "entity2id": entity2id,
        "relation2id": relation2id
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, output_path)
    print(f"[Save] Checkpoint saved")

def save_filtered_paths_streaming(
    output_dir: str,
    target_node_type: str,
    filtered_paths: List[Tuple[List[int], List[int]]],
    filtered_scores: np.ndarray,
    entity2id: dict,
    relation2id: dict,
    filter_config: dict,
    compress_output: bool = True
):
    """
    Save filtered paths efficiently with compression.
    
    Args:
        output_dir: Directory to save results
        target_node_type: Type of target node
        filtered_paths: List of filtered paths
        filtered_scores: Scores for filtered paths
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        filter_config: Filter configuration dict
        compress_output: Whether to create zip file
    
    Returns:
        Path to saved file
    """
    import pickle
    import json
    import zipfile
    
    print(f"\n{'='*60}")
    print("SAVING FILTERED PATHS")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Save] Total paths: {len(filtered_paths)}")
    print(f"[Save] Target: {target_node_type}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{target_node_type}_metadata.json")
    metadata = {
        "num_paths": len(filtered_paths),
        "target_node_type": target_node_type,
        "filter_config": filter_config,
        "entity_count": len(entity2id),
        "relation_count": len(relation2id)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save mappings
    mappings_path = os.path.join(output_dir, f"{target_node_type}_mappings.pkl")
    with open(mappings_path, 'wb') as f:
        pickle.dump({
            "entity2id": entity2id,
            "relation2id": relation2id
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save scores
    scores_path = os.path.join(output_dir, f"{target_node_type}_scores.npy")
    np.save(scores_path, filtered_scores)
    
    # Save paths in chunks
    chunk_size = 50000
    num_chunks = (len(filtered_paths) + chunk_size - 1) // chunk_size
    
    print(f"[Save] Saving paths in {num_chunks} chunks...")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(filtered_paths))
        
        chunk_data = filtered_paths[start_idx:end_idx]
        chunk_path = os.path.join(output_dir, f"{target_node_type}_paths_chunk_{chunk_idx:04d}.pkl")
        
        with open(chunk_path, 'wb') as f:
            pickle.dump({
                "paths": chunk_data,
                "chunk_id": chunk_idx
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
            print(f"[Save] Progress: {end_idx}/{len(filtered_paths)} paths saved")
    
    # Create manifest
    manifest = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_paths": len(filtered_paths),
        "files": {
            "metadata": f"{target_node_type}_metadata.json",
            "mappings": f"{target_node_type}_mappings.pkl",
            "scores": f"{target_node_type}_scores.npy",
            "path_chunks": [f"{target_node_type}_paths_chunk_{i:04d}.pkl" for i in range(num_chunks)]
        }
    }
    
    manifest_path = os.path.join(output_dir, f"{target_node_type}_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Compress if requested
    if compress_output:
        print(f"\n[Compress] Creating zip archive...")
        zip_path = os.path.join(output_dir, f"{target_node_type}_filtered_paths.zip")
        
        files_to_compress = [f for f in os.listdir(output_dir) 
                            if os.path.isfile(os.path.join(output_dir, f)) and not f.endswith('.zip')]
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for filename in files_to_compress:
                file_path = os.path.join(output_dir, filename)
                zipf.write(file_path, arcname=filename)
        
        compressed_size = os.path.getsize(zip_path)
        print(f"[Compress] Compressed size: {compressed_size / (1024**2):.2f} MB")
        print(f"[Compress] Saved to: {zip_path}")
        
        return zip_path
    
    print(f"[Save] All files saved to: {output_dir}")
    return manifest_path

def load_filtered_paths(results_path: str):
    """Load filtered paths from saved files"""
    import pickle
    import json
    import zipfile
    
    print(f"[Load] Loading from {results_path}")
    
    # Handle zip file
    if results_path.endswith('.zip'):
        extract_dir = results_path.replace('.zip', '_extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(results_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        results_path = extract_dir
    
    # Load manifest
    manifest_files = [f for f in os.listdir(results_path) if f.endswith('_manifest.json')]
    if not manifest_files:
        raise FileNotFoundError("No manifest file found")
    
    manifest_path = os.path.join(results_path, manifest_files[0])
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Load metadata
    metadata_path = os.path.join(results_path, manifest["files"]["metadata"])
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load mappings
    mappings_path = os.path.join(results_path, manifest["files"]["mappings"])
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    # Load scores
    scores_path = os.path.join(results_path, manifest["files"]["scores"])
    scores = np.load(scores_path)
    
    # Load paths
    all_paths = []
    for chunk_file in manifest["files"]["path_chunks"]:
        chunk_path = os.path.join(results_path, chunk_file)
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
            all_paths.extend(chunk_data["paths"])
    
    print(f"[Load] Loaded {len(all_paths)} paths")
    
    return {
        "paths": all_paths,
        "scores": scores,
        "entity2id": mappings["entity2id"],
        "relation2id": mappings["relation2id"],
        "metadata": metadata
    }

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 17: STREAMING SAVE (MEMORY-SAFE)
# ============================================================================

def save_filtered_paths_streaming(
    output_dir: str,
    target_node_type: str,
    filtered_paths: List[Tuple[List[int], List[int]]],
    filtered_scores: np.ndarray,
    entity2id: dict,
    relation2id: dict,
    filter_config: dict,
    compress_output: bool = True,
    chunk_size: int = 10000
):
    """
    Memory-efficient streaming save - processes paths in chunks.
    """
    import pickle
    import json
    import zipfile
    
    print(f"\n{'='*60}")
    print("STREAMING SAVE (Memory-Safe)")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_paths = len(filtered_paths)
    print(f"[Save] Total paths: {num_paths:,}")
    
    # Step 1: Lightweight version (always saved first as safety net)
    print("\n[Save] Step 1/3: Saving lightweight version...")
    
    lightweight_path = os.path.join(output_dir, f"{target_node_type}_lightweight.npz")
    np.savez_compressed(
        lightweight_path,
        scores=filtered_scores.astype(np.float32),
        num_paths=num_paths
    )
    
    print(f"[Save] ✓ Lightweight: {os.path.getsize(lightweight_path) / (1024**2):.2f} MB")
    
    # Step 2: Metadata
    metadata_path = os.path.join(output_dir, f"{target_node_type}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "num_paths": num_paths,
            "filter_config": filter_config,
            "entity_count": len(entity2id),
            "relation_count": len(relation2id)
        }, f, indent=2)
    
    # Step 3: Mappings
    mappings_path = os.path.join(output_dir, f"{target_node_type}_mappings.pkl")
    with open(mappings_path, 'wb') as f:
        pickle.dump({
            "entity2id": entity2id,
            "relation2id": relation2id
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Step 4: Save paths in chunks (streaming)
    print(f"\n[Save] Step 2/3: Saving paths in chunks of {chunk_size:,}...")
    
    num_chunks = (num_paths + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_paths)
        
        chunk_data = filtered_paths[start_idx:end_idx]
        chunk_path = os.path.join(output_dir, f"{target_node_type}_paths_chunk_{chunk_idx:04d}.pkl")
        
        with open(chunk_path, 'wb') as f:
            pickle.dump({
                "paths": chunk_data,
                "chunk_id": chunk_idx
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
            print(f"[Save] Progress: {end_idx:,}/{num_paths:,} paths")
    
    print(f"[Save] ✓ Saved {num_chunks} chunks")
    
    # Step 5: Manifest
    manifest = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_paths": num_paths,
        "files": {
            "lightweight": f"{target_node_type}_lightweight.npz",
            "metadata": f"{target_node_type}_metadata.json",
            "mappings": f"{target_node_type}_mappings.pkl",
            "path_chunks": [f"{target_node_type}_paths_chunk_{i:04d}.pkl" for i in range(num_chunks)]
        }
    }
    
    manifest_path = os.path.join(output_dir, f"{target_node_type}_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Step 6: Compress
    if compress_output:
        print(f"\n[Save] Step 3/3: Compressing...")
        
        zip_path = os.path.join(output_dir, f"{target_node_type}_results.zip")
        
        files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)) and not f.endswith('.zip')]
        
        original_size = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for filename in files:
                file_path = os.path.join(output_dir, filename)
                original_size += os.path.getsize(file_path)
                zipf.write(file_path, arcname=filename)
        
        compressed_size = os.path.getsize(zip_path)
        ratio = (1 - compressed_size / original_size) * 100
        
        print(f"\n[Save] ✓ Compressed!")
        print(f"[Save] Original: {original_size / (1024**2):.1f} MB")
        print(f"[Save] Compressed: {compressed_size / (1024**2):.1f} MB ({ratio:.1f}% reduction)")
        print(f"[Save] File: {zip_path}")
        
        return zip_path
    
    return manifest_path

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 18: COMPLETE PIPELINE FUNCTION
# ============================================================================
def run_gat_transr_path_pipeline(
    train_graph_path: str,
    output_dir: str,
    config: Config
):
    """
    Complete Pipeline: GAT → TransR → Path Filtering → Save
    
    Args:
        train_graph_path: Path to training graph
        output_dir: Output directory
        config: Configuration object
    
    Returns:
        Dictionary with all outputs
    """
    print("\n" + "="*80)
    print("PIPELINE: GAT → TransR → Path Filtering")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    
    results = {}
    
    # ========================================================================
    # STEP 1: Train GAT
    # ========================================================================
    print("\n[STEP 1] Training GAT...")
    gat_output_dir = os.path.join(output_dir, "gat")
    os.makedirs(gat_output_dir, exist_ok=True)
    
    gat_embeddings_path = os.path.join(gat_output_dir, "gat_embeddings.pt")
    
    gat_model, gat_embeddings = train_gat(
        train_graph_path=train_graph_path,
        output_path=gat_embeddings_path,
        hidden_dim=config.GAT_HIDDEN_DIM,
        out_dim=config.GAT_OUT_DIM,
        num_layers=config.GAT_NUM_LAYERS,
        heads=config.GAT_HEADS,
        dropout=config.GAT_DROPOUT,
        lr=config.GAT_LR,
        epochs=config.GAT_EPOCHS,
        batch_size=config.GAT_BATCH_SIZE,
        device=config.DEVICE
    )
    
    results['gat_model'] = gat_model
    results['gat_embeddings'] = gat_embeddings
    results['gat_embeddings_path'] = gat_embeddings_path
    
    print(f"✓ GAT training complete | Embeddings: {gat_embeddings_path}")
    
    # ========================================================================
    # STEP 2: Train TransR (initialized with GAT)
    # ========================================================================
    print("\n[STEP 2] Training TransR with GAT initialization...")
    transr_output_dir = os.path.join(output_dir, "transr")
    os.makedirs(transr_output_dir, exist_ok=True)
    
    transr_model, entity2id, relation2id = train_transr(
        hetero_graph_path=train_graph_path,
        output_dir=transr_output_dir,
        emb_dim=config.TRANSR_EMB_DIM,
        lr=config.TRANSR_LR,
        epochs=config.TRANSR_EPOCHS,
        batch_size=config.TRANSR_BATCH_SIZE,
        num_negatives=config.TRANSR_NUM_NEGATIVES,
        device=config.DEVICE,
        gat_embeddings=gat_embeddings  # Initialize with GAT!
    )
    
    results['transr_model'] = transr_model
    results['entity2id'] = entity2id
    results['relation2id'] = relation2id
    results['transr_model_path'] = os.path.join(transr_output_dir, "transr_final.pt")
    
    print(f"✓ TransR training complete | Model: {results['transr_model_path']}")
    
    # ========================================================================
    # STEP 3: Extract Paths
    # ========================================================================
    print("\n[STEP 3] Extracting paths...")
    data, metadata = load_hetero_graph(train_graph_path)
    
    paths = extract_paths_from_hetero_optimized(
        hetero_graph=data,
        entity2id=entity2id,
        relation2id=relation2id,
        source_node_type="user",
        target_node_type="poi",
        max_length=config.MAX_PATH_LENGTH,
        max_paths=100000,  # NEW
        max_paths_per_source=1000,  # NEW
        enable_cycle_detection=True  # NEW
    )
    
    results['all_paths'] = paths
    print(f"✓ Path extraction complete | Total paths: {len(paths)}")
    
    # ========================================================================
    # STEP 4: Score Paths
    # ========================================================================
    print("\n[STEP 4] Scoring paths with TransR...")
    scores = score_paths_with_transr(
        paths=paths,
        model=transr_model,
        device=config.DEVICE,
        aggregation=config.AGGREGATION
    )
    
    results['all_scores'] = scores
    print(f"✓ Path scoring complete")
    
    # ========================================================================
    # STEP 5: Filter Paths
    # ========================================================================
    print("\n[STEP 5] Filtering paths...")
    filtered_paths, filtered_scores, filtered_indices = filter_paths_per_pair(
        paths=paths,
        scores=scores,
        method=config.FILTER_STRATEGY,
        k=config.FILTER_K,
        reverse=False
    )
    
    results['filtered_paths'] = filtered_paths
    results['filtered_scores'] = filtered_scores
    results['filtered_indices'] = filtered_indices
    
    print(f"✓ Path filtering complete | Filtered: {len(filtered_paths)}/{len(paths)}")

    # ========================================================================
    # STEP 6: Analyze Paths
    # ========================================================================
    print("\n[STEP 6] Analyzing paths...")
    analyze_paths(filtered_paths, entity2id, relation2id, "Filtered Paths Analysis")
    print_sample_paths(filtered_paths, entity2id, relation2id, metadata, num_samples=5)
    
    # ========================================================================
    # STEP 7: Save Everything
    # ========================================================================
    print("\n[STEP 7] Saving filtered paths...")
    paths_output_dir = os.path.join(output_dir, "filtered_paths")
    
    saved_path = save_filtered_paths_streaming(
        output_dir=paths_output_dir,
        target_node_type="poi",
        filtered_paths=filtered_paths,
        filtered_scores=filtered_scores,
        entity2id=entity2id,
        relation2id=relation2id,
        filter_config={
            'strategy': config.FILTER_STRATEGY,
            'k': config.FILTER_K,
            'max_path_length': config.MAX_PATH_LENGTH,
            'aggregation': config.AGGREGATION
        },
        compress_output=True,
        chunk_size=10000  # NEW
    )

    results['saved_paths_file'] = saved_path
    print(f"✓ Paths saved to: {saved_path}")
    
    # ========================================================================
    # COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"GAT embeddings: {results['gat_embeddings_path']}")
    print(f"TransR model: {results['transr_model_path']}")
    print(f"Filtered paths: {results['saved_paths_file']}")
    
    return results

# %% [markdown]
# ### Step 1.6: Complete Pipeline Configuration
# Set paths and verify configuration before running Phase 1

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 19: MAIN EXECUTION
# ============================================================================

# Update config with your actual paths
config.TRAIN_GRAPH = "/kaggle/input/poi-graph-v6/poi_graph_v6.pt"  # UPDATE THIS
config.OUTPUT_DIR = "./pipeline_output"

print("="*80)
print("READY TO RUN PIPELINE")
print("="*80)
print(f"\nConfiguration:")
print(f"  Train Graph: {config.TRAIN_GRAPH}")
print(f"  Output Dir: {config.OUTPUT_DIR}")
print(f"  Device: {config.DEVICE}")
print(f"  GAT Epochs: {config.GAT_EPOCHS}")
print(f"  TransR Epochs: {config.TRANSR_EPOCHS}")
print(f"  Max Path Length: {config.MAX_PATH_LENGTH}")
print(f"  Filter Strategy: {config.FILTER_STRATEGY} (k={config.FILTER_K})")
print("\n" + "="*80)
print("Run the cell below to start the pipeline")
print("="*80)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 20: EXECUTE PIPELINE (Run this to start!)
# ============================================================================
# Run complete pipeline
results = run_gat_transr_path_pipeline(
    train_graph_path=config.TRAIN_GRAPH,
    output_dir=config.OUTPUT_DIR,
    config=config
)

print("\n✓ Pipeline execution completed successfully!")
print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")

# Display summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
for key, value in results.items():
    if isinstance(value, str):
        print(f"  {key}: {value}")
    elif isinstance(value, (list, np.ndarray)):
        print(f"  {key}: {len(value)} items")
    elif hasattr(value, '__class__'):
        print(f"  {key}: {value.__class__.__name__}")

# %% [markdown]
# ## Phase 2: Path Enrichment with Embeddings & Ratings
# 
# ### Step 2.1: Load Phase 1 Results
# Load filtered paths from Phase 1 output

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 21: LOAD PHASE 1 RESULTS FUNCTION
# ============================================================================

def load_phase1_results(results_path: str):
    """Load filtered paths from Phase 1 output"""
    import pickle
    import json
    import zipfile
    from pathlib import Path
    
    print(f"[Load] Loading Phase 1 results from {results_path}")
    
    results_path = Path(results_path)
    
    # Handle zip file
    if results_path.suffix == '.zip':
        extract_dir = results_path.parent / (results_path.stem + '_extracted')
        extract_dir.mkdir(exist_ok=True)
        
        print(f"[Load] Extracting to {extract_dir}")
        with zipfile.ZipFile(results_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        results_path = extract_dir
    
    # Load manifest
    manifest_files = list(results_path.glob('*_manifest.json'))
    if not manifest_files:
        raise FileNotFoundError(f"No manifest file found in {results_path}")
    
    with open(manifest_files[0], 'r') as f:
        manifest = json.load(f)
    
    # Load metadata
    metadata_file = results_path / manifest["files"]["metadata"]
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load mappings
    mappings_file = results_path / manifest["files"]["mappings"]
    with open(mappings_file, 'rb') as f:
        mappings = pickle.load(f)
    
    # Load all path chunks
    all_paths = []
    print(f"[Load] Loading {len(manifest['files']['path_chunks'])} path chunks...")
    for chunk_file in manifest["files"]["path_chunks"]:
        chunk_path = results_path / chunk_file
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
            all_paths.extend(chunk_data["paths"])
    
    print(f"[Load] ✓ Loaded {len(all_paths):,} filtered paths from Phase 1")
    print(f"[Load] ✓ Entity mappings: {len(mappings['entity2id']):,} entities")
    print(f"[Load] ✓ Relation mappings: {len(mappings['relation2id']):,} relations")
    
    return {
        'paths': all_paths,
        'entity2id': mappings['entity2id'],
        'relation2id': mappings['relation2id'],
        'metadata': metadata
    }

print("✓ Phase 1 results loader ready")

# %% [markdown]
# ### Step 2.2: PathEnricher Class
# Enriches paths with GAT embeddings and actual user ratings

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 22: ENHANCED PATH ENRICHER FOR PHASE 1 PATHS
# ============================================================================

import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

class PathEnricher:
    def __init__(self, 
                 hetero_data_path: str,
                 gat_embeddings_path: str,
                 phase1_results_path: str,
                 output_dir: str):
        """
        Initialize the PathEnricher with Phase 1 results.
        
        Args:
            hetero_data_path: Path to the HeteroData object (.pt file)
            gat_embeddings_path: Path to GAT embeddings (.pt file)
            phase1_results_path: Path to Phase 1 filtered paths (zip or directory)
            output_dir: Directory to save enriched path chunks
        """
        self.hetero_data_path = hetero_data_path
        self.gat_embeddings_path = gat_embeddings_path
        self.phase1_results_path = phase1_results_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Phase 1 results
        print("="*60)
        print("Loading Phase 1 filtered paths...")
        print("="*60)
        self.phase1_data = load_phase1_results(phase1_results_path)
        self.entity2id = self.phase1_data['entity2id']
        self.relation2id = self.phase1_data['relation2id']
        
        # Load data
        print("\n" + "="*60)
        print("Loading HeteroData...")
        print("="*60)
        self.hetero_data = self._load_hetero_data()
        
        print("\n" + "="*60)
        print("Loading GAT embeddings...")
        print("="*60)
        self.gat_embeddings = self._load_gat_embeddings()
        
        print("\n" + "="*60)
        print("Building rating lookup...")
        print("="*60)
        self.rating_lookup = self._build_rating_lookup()
        
        print("\n" + "="*60)
        print("Building node type mappings...")
        print("="*60)
        self.node_mappings = self._build_node_mappings()
        
    def _load_hetero_data(self):
        """Load the HeteroData object from .pt file."""
        data = torch.load(self.hetero_data_path, weights_only=False)
        if isinstance(data, dict) and 'graph' in data:
            return data['graph']
        return data
    
    def _load_gat_embeddings(self):
        """Load GAT embeddings from .pt file."""
        embeddings = torch.load(self.gat_embeddings_path, weights_only=False)
        return embeddings
    
    def _build_rating_lookup(self):
        """Build a dictionary for quick rating lookup."""
        edge_index = self.hetero_data[('user', 'rates', 'poi')]['edge_index']
        edge_attr = self.hetero_data[('user', 'rates', 'poi')]['edge_attr']
        
        print(f"  Total rating edges: {edge_index.shape[1]:,}")
        
        rating_dict = {}
        for i in range(edge_index.shape[1]):
            user_idx = edge_index[0, i].item()
            poi_idx = edge_index[1, i].item()
            rating = edge_attr[i].item() if edge_attr.dim() > 1 else edge_attr[i].item()
            rating_dict[(user_idx, poi_idx)] = rating
        
        print(f"  Built rating lookup with {len(rating_dict):,} unique user-POI pairs")
        return rating_dict
    
    def _build_node_mappings(self):
        """Build mappings between Phase 1 entity IDs and local node indices."""
        mappings = {
            'global_to_local': {},
            'type_offsets': {}
        }
        
        # Reverse entity2id mapping
        for key, global_id in self.entity2id.items():
            if isinstance(key, tuple) and len(key) == 2:
                node_type, local_idx = key
                mappings['global_to_local'][global_id] = (node_type, local_idx)
            elif isinstance(key, str) and '_' in key:
                # Handle string format: "user_0" -> ("user", 0)
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    node_type, local_idx = parts[0], int(parts[1])
                    mappings['global_to_local'][global_id] = (node_type, local_idx)
        
        # Build type offsets
        cumulative = 0
        for node_type in ['user', 'poi', 'sensory_attr', 'other_attr', 'category']:
            if node_type in self.hetero_data.node_types:
                num_nodes = self.hetero_data[node_type].num_nodes
                mappings['type_offsets'][node_type] = {
                    'start': cumulative,
                    'end': cumulative + num_nodes,
                    'size': num_nodes
                }
                cumulative += num_nodes
                print(f"  {node_type}: {num_nodes:,} nodes")
        
        print(f"  Total global->local mappings: {len(mappings['global_to_local']):,}")
        return mappings
    
    def _global_to_local(self, global_idx: int) -> Tuple[str, int]:
        """Convert global entity ID to (node_type, local_index)"""
        if global_idx in self.node_mappings['global_to_local']:
            return self.node_mappings['global_to_local'][global_idx]
        raise ValueError(f"Global index {global_idx} not found in mappings")
    
    def _get_node_embedding(self, node_type: str, local_idx: int):
        """Get embedding for a specific node using local index."""
        if isinstance(self.gat_embeddings, dict):
            if node_type in self.gat_embeddings:
                emb_tensor = self.gat_embeddings[node_type]
                if isinstance(emb_tensor, torch.Tensor):
                    if local_idx >= emb_tensor.shape[0]:
                        return None
                    return emb_tensor[local_idx].cpu().numpy()
        return None
    
    def enrich_path(self, path: Tuple[List[int], List[int]]) -> Dict:
        """Enrich a single path with GAT embeddings and rating."""
        node_indices, edge_types = path
        
        # Convert global indices to (node_type, local_idx) pairs
        node_info = []
        for global_idx in node_indices:
            try:
                node_type, local_idx = self._global_to_local(global_idx)
                node_info.append((global_idx, node_type, local_idx))
            except ValueError:
                node_info.append((global_idx, 'unknown', -1))
        
        # Extract user and POI (first and last nodes)
        user_global_idx = node_indices[0]
        poi_global_idx = node_indices[-1]
        
        user_type, user_local_idx = node_info[0][1], node_info[0][2]
        poi_type, poi_local_idx = node_info[-1][1], node_info[-1][2]
        
        # Get rating
        rating = self.rating_lookup.get((user_local_idx, poi_local_idx), None)
        has_actual_rating = (user_local_idx, poi_local_idx) in self.rating_lookup
        
        # Get embeddings for all nodes in path
        node_embeddings = []
        node_types = []
        local_indices = []
        
        for global_idx, node_type, local_idx in node_info:
            if node_type != 'unknown' and local_idx >= 0:
                emb = self._get_node_embedding(node_type, local_idx)
                if emb is not None:
                    node_embeddings.append(emb)
                    node_types.append(node_type)
                    local_indices.append(local_idx)
        
        return {
            'path': {
                'node_indices': node_indices,
                'edge_types': edge_types
            },
            'node_types': node_types,
            'local_indices': local_indices,
            'embeddings': node_embeddings,
            'rating': rating,
            'has_actual_rating': has_actual_rating,
            'user_idx': user_local_idx,
            'poi_idx': poi_local_idx,
            'user_global_idx': user_global_idx,
            'poi_global_idx': poi_global_idx,
            'user_type': user_type,
            'poi_type': poi_type
        }
    
    def process_all_chunks(self, chunk_size: int = 10000):
        """Process Phase 1 filtered paths and enrich them."""
        print(f"\n{'='*60}")
        print(f"Processing {len(self.phase1_data['paths']):,} Phase 1 filtered paths")
        print(f"{'='*60}")
        
        all_phase1_paths = self.phase1_data['paths']
        num_chunks = (len(all_phase1_paths) + chunk_size - 1) // chunk_size
        
        stats = {
            'total_paths': 0,
            'paths_with_actual_ratings': 0,
            'paths_with_default_ratings': 0,
            'chunks_processed': 0,
            'failed_paths': 0
        }
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(all_phase1_paths))
            chunk_paths = all_phase1_paths[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"Processing chunk {chunk_idx+1}/{num_chunks}")
            print(f"Paths: {start_idx:,} to {end_idx:,}")
            print(f"{'='*60}")
            
            enriched_paths = []
            for path in tqdm(chunk_paths, desc=f"Enriching chunk {chunk_idx+1}"):
                try:
                    enriched_path = self.enrich_path(path)
                    enriched_paths.append(enriched_path)
                except Exception as e:
                    stats['failed_paths'] += 1
                    if stats['failed_paths'] <= 5:
                        print(f"Warning: Failed to enrich path: {e}")
                    continue
            
            # Update statistics
            stats['total_paths'] += len(enriched_paths)
            paths_with_actual = sum(1 for p in enriched_paths if p.get('has_actual_rating', False))
            paths_with_default = sum(1 for p in enriched_paths if not p.get('has_actual_rating', False))
            
            stats['paths_with_actual_ratings'] += paths_with_actual
            stats['paths_with_default_ratings'] += paths_with_default
            stats['chunks_processed'] += 1
            
            # Save enriched chunk
            output_file = self.output_dir / f"enriched_poi_paths_chunk_{chunk_idx:04d}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(enriched_paths, f)
            
            print(f"Saved: {output_file}")
            print(f"Paths in chunk: {len(enriched_paths):,}")
            print(f"Actual ratings: {paths_with_actual:,} ({paths_with_actual/len(enriched_paths)*100:.1f}%)")
        
        # Save statistics
        stats_file = self.output_dir / 'enrichment_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ENRICHMENT COMPLETE!")
        print(f"{'='*60}")
        print(f"Total paths processed: {stats['total_paths']:,}")
        print(f"Paths with actual ratings: {stats['paths_with_actual_ratings']:,}")
        print(f"Hit rate: {stats['paths_with_actual_ratings']/stats['total_paths']*100:.1f}%")
        print(f"Failed paths: {stats['failed_paths']:,}")
        
        return stats

print("✓ PathEnricher class ready")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# CELL 23: RUN PATH ENRICHMENT ON PHASE 1 RESULTS
# ============================================================================

# Configure paths
HETERO_DATA_PATH = "/kaggle/input/poi-data-v5/poi_graph_v5.pt"
GAT_EMBEDDINGS_PATH = "./pipeline_output/gat/gat_embeddings.pt"  # From Phase 1

# Phase 1 output (from the previous pipeline run)
PHASE1_RESULTS_PATH = "./pipeline_output/filtered_paths/poi_results.zip"

# Output directory for enriched paths
ENRICHED_OUTPUT_DIR = "/kaggle/working/results"

print("="*80)
print("PHASE 2: ENRICHING PHASE 1 FILTERED PATHS")
print("="*80)
print(f"\nConfiguration:")
print(f"  HeteroData: {HETERO_DATA_PATH}")
print(f"  GAT Embeddings: {GAT_EMBEDDINGS_PATH}")
print(f"  Phase 1 Results: {PHASE1_RESULTS_PATH}")
print(f"  Output Directory: {ENRICHED_OUTPUT_DIR}")
print()

# Create enricher
enricher = PathEnricher(
    hetero_data_path=HETERO_DATA_PATH,
    gat_embeddings_path=GAT_EMBEDDINGS_PATH,
    phase1_results_path=PHASE1_RESULTS_PATH,
    output_dir=ENRICHED_OUTPUT_DIR
)

# Process all Phase 1 paths and enrich them
print("\n" + "="*80)
print("STARTING ENRICHMENT PROCESS")
print("="*80)

stats = enricher.process_all_chunks(chunk_size=10000)

print("\n" + "="*80)
print("✓ ENRICHMENT PHASE COMPLETE!")
print("="*80)
print(f"\nEnriched paths saved to: {ENRICHED_OUTPUT_DIR}")
print(f"Ready for model training!")

# %% [code]
# ============================================================================
# COLD START RESILIENCE EVALUATION
# Uses ONLY: regularized_model_best.pt + enriched path chunks
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# MODEL ARCHITECTURE (must match training exactly)
# ============================================================================

class RegularizedPathLSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            num_layers=2, dropout=dropout)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward_single(self, path_embeddings):
        seq_len   = path_embeddings.size(0)
        embed_dim = path_embeddings.size(1)
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                pad = torch.zeros(seq_len, self.embedding_dim - embed_dim,
                                  device=path_embeddings.device,
                                  dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, pad], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        x = path_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out.squeeze(0))
        lstm_out = self.dropout(lstm_out)
        if seq_len == 1:
            return lstm_out.squeeze(0), torch.ones(1, device=path_embeddings.device)
        attn_logits  = self.attention_score(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_logits, dim=0)
        path_repr    = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=0)
        return path_repr, attn_weights

    def forward_batch(self, path_embeddings_batch, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            path_embeddings_batch, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _   = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        max_len      = lstm_out.size(1)
        attn_logits  = self.attention_score(lstm_out).squeeze(-1)
        mask = (torch.arange(max_len, device=lengths.device).unsqueeze(0)
                < lengths.unsqueeze(1))
        attn_logits  = attn_logits.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        path_reprs   = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        return path_reprs, attn_weights

    def forward(self, path_embeddings, lengths=None):
        if lengths is not None:
            return self.forward_batch(path_embeddings, lengths)
        return self.forward_single(path_embeddings)


class RegularizedPathRecommendationModel(nn.Module):
    def __init__(self, embedding_dim=64, lstm_hidden_dim=128,
                 attention_dim=64, mlp_dims=[128, 64], dropout=0.4):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_dim   = attention_dim

        self.path_lstm   = RegularizedPathLSTM(embedding_dim, lstm_hidden_dim, dropout)
        self.query_proj  = nn.Linear(lstm_hidden_dim, attention_dim)
        self.key_proj    = nn.Linear(lstm_hidden_dim, attention_dim)
        self.value_proj  = nn.Linear(lstm_hidden_dim, attention_dim)

        mlp_layers, in_dim = [], attention_dim
        for h in mlp_dims:
            mlp_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h),
                           nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def encode_paths_batch(self, all_paths):
        device = next(self.parameters()).device
        flat_paths, path_map = [], []
        for si, paths in enumerate(all_paths):
            for pi, emb in enumerate(paths):
                if isinstance(emb, np.ndarray) and len(emb) > 0:
                    flat_paths.append(emb)
                    path_map.append((si, pi))
        if not flat_paths:
            return [[] for _ in all_paths]
        groups = defaultdict(list)
        for idx, path in enumerate(flat_paths):
            groups[len(path)].append((idx, path))
        representations = [None] * len(flat_paths)
        for length, group in groups.items():
            indices = [g[0] for g in group]
            paths_g = [g[1] for g in group]
            batch   = torch.from_numpy(np.stack(paths_g)).float().to(device)
            lens    = torch.full((len(paths_g),), length, dtype=torch.long, device=device)
            reprs, _ = self.path_lstm(batch, lens)
            for i, r in enumerate(reprs):
                representations[indices[i]] = r
        result = [[] for _ in all_paths]
        for fi, (si, _) in enumerate(path_map):
            if representations[fi] is not None:
                result[si].append(representations[fi])
        return result

    def aggregate_path_sets_batch(self, all_path_reprs):
        device     = next(self.parameters()).device
        batch_size = len(all_path_reprs)
        valid      = [r for r in all_path_reprs if r]
        if not valid:
            return torch.zeros(batch_size, self.attention_dim, device=device), [0] * batch_size
        max_paths = max(len(r) for r in all_path_reprs)
        padded    = torch.zeros(batch_size, max_paths, self.lstm_hidden_dim, device=device)
        counts    = []
        for i, reprs in enumerate(all_path_reprs):
            if reprs:
                padded[i, :len(reprs)] = torch.stack(reprs)
                counts.append(len(reprs))
            else:
                counts.append(1)
        counts = torch.tensor(counts, device=device)
        Q = self.query_proj(padded)
        K = self.key_proj(padded)
        V = self.value_proj(padded)
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
        mask   = (torch.arange(max_paths, device=device).unsqueeze(0)
                  < counts.unsqueeze(1))
        mask   = mask.unsqueeze(1).expand(-1, max_paths, -1)
        scores = scores.masked_fill(~mask, -1e9)
        attn_w = F.softmax(scores, dim=2)
        L_mp   = torch.max(torch.bmm(attn_w, V), dim=1)[0]
        best   = torch.argmax(attn_w.sum(dim=1), dim=1).tolist()
        return L_mp, best

    def forward(self, paths_batch):
        reprs       = self.encode_paths_batch(paths_batch)
        L_mp, best  = self.aggregate_path_sets_batch(reprs)
        predictions = torch.sigmoid(self.mlp(L_mp).squeeze(-1))
        return predictions, best


# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    @staticmethod
    def hit_rate(ranked, gt, k):
        return 1.0 if set(ranked[:k]) & gt else 0.0

    @staticmethod
    def precision(ranked, gt, k):
        return len(set(ranked[:k]) & gt) / k if k else 0.0

    @staticmethod
    def recall(ranked, gt, k):
        return len(set(ranked[:k]) & gt) / len(gt) if gt else 0.0

    @staticmethod
    def f1(ranked, gt, k):
        p = Metrics.precision(ranked, gt, k)
        r = Metrics.recall(ranked, gt, k)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @staticmethod
    def ndcg(ranked, gt, k):
        dcg  = sum(1.0 / np.log2(i + 2)
                   for i, item in enumerate(ranked[:k]) if item in gt)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gt), k)))
        return dcg / idcg if idcg else 0.0

    @staticmethod
    def mrr(ranked, gt):
        for i, item in enumerate(ranked):
            if item in gt:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def map_at_k(ranked, gt, k):
        if not gt:
            return 0.0
        hits = score = 0.0
        for i, item in enumerate(ranked[:k]):
            if item in gt:
                hits  += 1
                score += hits / (i + 1)
        return score / min(len(gt), k)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_chunks(chunks_dir: str) -> List[Dict]:
    p     = Path(chunks_dir)
    files = sorted(p.glob('enriched_poi_paths_chunk_*.pkl'))
    if not files:
        files = sorted(p.glob('*chunk*.pkl'))
    paths = []
    for f in tqdm(files, desc="Loading chunks"):
        with open(f, 'rb') as fh:
            paths.extend(pickle.load(fh))
    print(f"Loaded {len(paths):,} paths")
    return paths


def prepare_data(all_paths):
    """Build user->item->paths map, positive sets, interaction counts."""
    user_item_paths         = defaultdict(lambda: defaultdict(list))
    user_positive_items     = defaultdict(set)
    user_interaction_counts = defaultdict(int)
    pois_with_ratings       = set()

    for path in all_paths:
        if not path.get('embeddings') or len(path['embeddings']) == 0:
            continue
        user_id = (path.get('user_idx') or path.get('user_global_idx') or
                   path.get('user_id')  or path.get('user'))
        item_id = (path.get('poi_idx')  or path.get('poi_global_idx') or
                   path.get('poi_id')   or path.get('item_id'))
        if user_id is None or item_id is None:
            continue

        user_item_paths[user_id][item_id].append(path)

        rating     = path.get('rating')
        has_actual = path.get('has_actual_rating', False)
        if has_actual and rating is not None:
            pois_with_ratings.add(item_id)
            if rating >= 3.0:
                user_positive_items[user_id].add(item_id)
                user_interaction_counts[user_id] += 1

    # Prune to rated POIs only (matches your existing eval logic)
    pruned = {
        u: {i: p for i, p in items.items() if i in pois_with_ratings}
        for u, items in user_item_paths.items()
    }

    print(f"Users with >=1 positive : {len(user_positive_items):,}")
    print(f"Rated POIs              : {len(pois_with_ratings):,}")
    return pruned, dict(user_positive_items), dict(user_interaction_counts)


# ============================================================================
# RANKING + EVALUATION HELPERS
# ============================================================================

def rank_items(model, user_id, user_item_paths, device, batch_size=64):
    model.eval()
    all_items = list(user_item_paths[user_id].keys())
    rankings  = []
    for i in range(0, len(all_items), batch_size):
        batch_items  = all_items[i:i + batch_size]
        paths_batch, valid_items = [], []
        for item_id in batch_items:
            embs = []
            for p in user_item_paths[user_id][item_id]:
                e = p.get('embeddings')
                if e is not None and len(e) > 0:
                    embs.append(np.array(e) if isinstance(e, list) else e)
            if embs:
                paths_batch.append(embs)
                valid_items.append(item_id)
        if not paths_batch:
            continue
        with torch.no_grad():
            scores, _ = model(paths_batch)
        for item_id, score in zip(valid_items, scores.cpu().numpy()):
            rankings.append((item_id, float(score)))
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def evaluate_users(model, user_ids, user_item_paths,
                   user_positive_items, device, k_values):
    """Evaluate a list of users and return averaged metrics dict."""
    collector = defaultdict(list)
    n_eval = n_skip = 0

    for user_id in tqdm(user_ids, desc="  Evaluating", leave=False):
        gt = user_positive_items.get(user_id, set())
        if not gt or user_id not in user_item_paths:
            n_skip += 1
            continue
        rankings = rank_items(model, user_id, user_item_paths, device)
        if not rankings:
            n_skip += 1
            continue
        ranked = [item for item, _ in rankings]
        if len(gt) == len(ranked) or not gt:
            n_skip += 1
            continue
        n_eval += 1
        for k in k_values:
            collector[f'HR@{k}'].append(Metrics.hit_rate(ranked, gt, k))
            collector[f'P@{k}'].append(Metrics.precision(ranked, gt, k))
            collector[f'R@{k}'].append(Metrics.recall(ranked, gt, k))
            collector[f'F1@{k}'].append(Metrics.f1(ranked, gt, k))
            collector[f'NDCG@{k}'].append(Metrics.ndcg(ranked, gt, k))
            collector[f'MAP@{k}'].append(Metrics.map_at_k(ranked, gt, k))
        collector['MRR'].append(Metrics.mrr(ranked, gt))

    avg = {k: float(np.mean(v)) for k, v in collector.items()}
    return avg, n_eval, n_skip


# ============================================================================
# TEST 1 — Interaction-count cold start
# ============================================================================

def test_interaction_cold_start(model, user_item_paths, user_positive_items,
                                 user_interaction_counts, device,
                                 k_values, thresholds=[1, 2, 5]):
    print("\n" + "=" * 65)
    print("TEST 1: COLD START BY INTERACTION COUNT")
    print("=" * 65)

    all_results = {}
    for thresh in thresholds:
        cold = [u for u, c in user_interaction_counts.items()
                if c <= thresh and u in user_positive_items]
        warm = [u for u, c in user_interaction_counts.items()
                if c >  thresh and u in user_positive_items]

        print(f"\n  Threshold <=  {thresh}  |  cold={len(cold):,}  warm={len(warm):,}")
        if not cold:
            print("  No cold users at this threshold -- skipping.")
            continue

        cold_paths = {u: user_item_paths[u] for u in cold if u in user_item_paths}
        warm_paths = {u: user_item_paths[u] for u in warm if u in user_item_paths}

        print("  Evaluating cold users...")
        cold_m, cn, cs = evaluate_users(
            model, cold, cold_paths, user_positive_items, device, k_values)
        print("  Evaluating warm users...")
        warm_m, wn, ws = evaluate_users(
            model, warm, warm_paths, user_positive_items, device, k_values)

        degradation = {
            m: cold_m[m] / warm_m[m]
            for m in cold_m if m in warm_m and warm_m[m] > 0
        }

        print(f"\n  {'Metric':<12} {'Cold':>8} {'Warm':>8} {'Ratio':>8}  Quality")
        print(f"  {'-'*50}")
        for k in k_values:
            for m in [f'HR@{k}', f'NDCG@{k}']:
                cv = cold_m.get(m, 0)
                wv = warm_m.get(m, 0)
                r  = degradation.get(m, float('nan'))
                q  = ('strong' if r > 0.85 else 'good' if r > 0.70
                       else 'fair' if r > 0.50 else 'poor')
                print(f"  {m:<12} {cv:>8.4f} {wv:>8.4f} {r:>8.3f}  {q}")
        mrr_r = degradation.get('MRR', float('nan'))
        print(f"  {'MRR':<12} {cold_m.get('MRR',0):>8.4f} "
              f"{warm_m.get('MRR',0):>8.4f} {mrr_r:>8.3f}")

        all_results[thresh] = {
            'cold': cold_m, 'warm': warm_m,
            'degradation': degradation,
            'cold_n': cn, 'warm_n': wn
        }

    return all_results


# ============================================================================
# TEST 2 — Path coverage
# ============================================================================

def test_path_coverage(model, user_item_paths, user_positive_items,
                        device, k_values):
    print("\n" + "=" * 65)
    print("TEST 2: PATH COVERAGE (CL under sparsity)")
    print("=" * 65)

    user_avg_paths = {
        u: np.mean([len(paths) for paths in items.values()])
        for u, items in user_item_paths.items() if items
    }

    buckets = {
        'very_sparse (avg<=1)': lambda x: x <= 1.0,
        'sparse (1<avg<=3)':    lambda x: 1.0 < x <= 3.0,
        'moderate (3<avg<=10)': lambda x: 3.0 < x <= 10.0,
        'dense (avg>10)':       lambda x: x > 10.0,
    }

    results = {}
    for name, cond in buckets.items():
        users = [u for u, avg in user_avg_paths.items()
                 if cond(avg) and u in user_positive_items]
        paths = {u: user_item_paths[u] for u in users if u in user_item_paths}
        print(f"\n  {name}  ({len(users):,} users)")
        if len(users) < 3:
            print("  Too few users -- skipping.")
            continue
        m, n_eval, n_skip = evaluate_users(
            model, users, paths, user_positive_items, device, k_values)
        results[name] = m
        for k in k_values:
            print(f"    K={k:2d}  HR={m.get(f'HR@{k}',0):.4f}  "
                  f"NDCG={m.get(f'NDCG@{k}',0):.4f}  "
                  f"Recall={m.get(f'R@{k}',0):.4f}")
        print(f"    MRR={m.get('MRR',0):.4f}  (evaluated {n_eval}, skipped {n_skip})")

    return results


# ============================================================================
# TEST 3 — CL embedding quality for cold users
# ============================================================================

def test_embedding_quality(model, user_item_paths, user_positive_items,
                            user_interaction_counts, device,
                            cold_threshold=2, n_users=200):
    print("\n" + "=" * 65)
    print("TEST 3: CL EMBEDDING QUALITY (cold vs warm)")
    print("=" * 65)

    cold = [u for u, c in user_interaction_counts.items()
            if c <= cold_threshold and u in user_item_paths][:n_users]
    warm = [u for u, c in user_interaction_counts.items()
            if c >  cold_threshold and u in user_item_paths][:n_users]

    model.eval()

    def extract_reprs(users):
        out = {}
        for uid in tqdm(users, desc="  Extracting", leave=False):
            reprs = []
            for paths in user_item_paths[uid].values():
                for path in paths:
                    e = path.get('embeddings')
                    if e is None or len(e) == 0:
                        continue
                    e = np.array(e) if isinstance(e, list) else e
                    t = torch.from_numpy(e).float().to(device)
                    with torch.no_grad():
                        r, _ = model.path_lstm.forward_single(t)
                    reprs.append(F.normalize(r, dim=0).cpu().numpy())
            if reprs:
                out[uid] = reprs
        return out

    cold_reprs = extract_reprs(cold)
    warm_reprs = extract_reprs(warm)

    def intra_sim(reprs_dict):
        sims = []
        for reprs in reprs_dict.values():
            if len(reprs) < 2:
                continue
            arr = np.stack(reprs)
            mat = arr @ arr.T
            n   = len(reprs)
            sims.extend(mat[~np.eye(n, dtype=bool)].tolist())
        return float(np.mean(sims)) if sims else 0.0

    def inter_dist(reprs_dict):
        centroids = []
        for reprs in reprs_dict.values():
            c = np.mean(np.stack(reprs), axis=0)
            centroids.append(c / (np.linalg.norm(c) + 1e-8))
        if len(centroids) < 2:
            return 0.0
        arr = np.stack(centroids)
        idx = np.random.choice(len(arr), min(100, len(arr)), replace=False)
        sub = arr[idx]
        dists = [np.linalg.norm(sub[i] - sub[j])
                 for i in range(len(sub)) for j in range(i + 1, len(sub))]
        return float(np.mean(dists)) if dists else 0.0

    ci = intra_sim(cold_reprs)
    wi = intra_sim(warm_reprs)
    cd = inter_dist(cold_reprs)
    wd = inter_dist(warm_reprs)

    print(f"\n  Cold threshold : <= {cold_threshold} interactions")
    print(f"  Users analysed : cold={len(cold_reprs):,}  warm={len(warm_reprs):,}")
    print(f"\n  {'Metric':<38} {'Cold':>8} {'Warm':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Intra-user path cosine similarity':<38} {ci:>8.4f} {wi:>8.4f}")
    print(f"  {'Inter-user centroid L2 distance':<38} {cd:>8.4f} {wd:>8.4f}")
    print()
    print("  Interpretation:")
    print(f"  {'OK' if ci >= 0.7 * wi else '!!'} Intra-user consistency "
          f"({'good -- CL embeddings are stable for cold users' if ci >= 0.7 * wi else 'lower than warm -- CL may need more epochs'})")
    print(f"  {'OK' if cd >= 0.8 * wd else '!!'} Inter-user separability "
          f"({'good -- cold users are distinguishable' if cd >= 0.8 * wd else 'cold users cluster together'})")

    return {'cold_intra': ci, 'warm_intra': wi,
            'cold_inter': cd, 'warm_inter': wd}


# ============================================================================
# TEST 4 — Resilience score
# ============================================================================

def compute_resilience_score(interaction_results, primary_metric='NDCG@20',
                              threshold=2):
    print("\n" + "=" * 65)
    print("TEST 4: RESILIENCE SCORE")
    print("=" * 65)

    if threshold not in interaction_results:
        print(f"  Threshold {threshold} not found in results.")
        return None

    r  = interaction_results[threshold]
    cv = r['cold'].get(primary_metric, 0)
    wv = r['warm'].get(primary_metric, 0)

    if wv == 0:
        print("  Warm metric is 0 -- cannot compute ratio.")
        return None

    score = cv / wv
    label = ('excellent' if score > 0.85 else 'good'     if score > 0.70
             else 'moderate' if score > 0.50 else 'poor')

    print(f"\n  Metric     : {primary_metric}")
    print(f"  Threshold  : <= {threshold} interaction(s)")
    print(f"  Cold value : {cv:.4f}")
    print(f"  Warm value : {wv:.4f}")
    print(f"  Score      : {score:.4f}  [{label}]")
    print(f"\n  Thesis quote:")
    print(f"  'With CL pretraining, the model achieves a cold-start resilience")
    print(f"   score of {score:.3f} on {primary_metric}, retaining")
    print(f"   {score*100:.1f}% of warm-user performance'")

    return score


# ============================================================================
# MAIN
# ============================================================================

def main():

    ENRICHED_CHUNKS_DIR = "/kaggle/working/results"
    MODEL_PATH          = "/kaggle/input/datasets/lavivasudev/full-model/full_model_model (4).pt"
    K_VALUES            = [5, 10, 20, 50]
    PRIMARY_METRIC      = "NDCG@20"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_paths = load_chunks(ENRICHED_CHUNKS_DIR)
    pruned_paths, user_positive_items, user_interaction_counts = prepare_data(all_paths)

    # ── Auto-detect thresholds from actual interaction distribution ──────
    counts = sorted(user_interaction_counts.values())
    print(f"\nInteraction count distribution across {len(counts)} users:")
    print(f"  Min    : {min(counts)}")
    print(f"  Median : {int(np.median(counts))}")
    print(f"  Mean   : {np.mean(counts):.1f}")
    print(f"  Max    : {max(counts)}")
    print(f"  Percentiles — 10th: {int(np.percentile(counts,10))}  "
          f"25th: {int(np.percentile(counts,25))}  "
          f"50th: {int(np.percentile(counts,50))}")

    # Set thresholds at 10th, 25th, 50th percentile of interaction counts
    p10 = int(np.percentile(counts, 10))
    p25 = int(np.percentile(counts, 25))
    p50 = int(np.percentile(counts, 50))
    # Ensure at least 3 distinct values and no duplicates
    thresholds = sorted(set([p10, p25, p50]))
    if len(thresholds) < 2:
        thresholds = sorted(set([max(1, p10), p25, p50, p50 + 1]))
    print(f"\n  Auto thresholds: {thresholds}")

    # Also set COLD_THRESHOLD for embedding test to 25th percentile
    COLD_THRESHOLD = p25

    # Load model
    print(f"\nLoading model from {MODEL_PATH} ...")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    cfg  = ckpt.get('model_config', {
        'embedding_dim': 64, 'lstm_hidden_dim': 128,
        'attention_dim': 64, 'mlp_dims': [128, 64], 'dropout': 0.4
    })
    model = RegularizedPathRecommendationModel(
        embedding_dim   = cfg.get('embedding_dim',   64),
        lstm_hidden_dim = cfg.get('lstm_hidden_dim', 128),
        attention_dim   = cfg.get('attention_dim',   64),
        mlp_dims        = cfg.get('mlp_dims',        [128, 64]),
        dropout         = cfg.get('dropout',         0.4)
    )
    state_dict = ckpt['model_state_dict']
    remapped = {
        k.replace('path_encoder.', 'path_lstm.'): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(remapped)
    model.to(device).eval()
    print("Model loaded successfully")

    # Run tests with auto-detected thresholds
    interaction_results = test_interaction_cold_start(
        model, pruned_paths, user_positive_items,
        user_interaction_counts, device, K_VALUES,
        thresholds=thresholds
    )

    coverage_results = test_path_coverage(
        model, pruned_paths, user_positive_items,
        device, k_values=[10, 20]
    )

    embedding_results = test_embedding_quality(
        model, pruned_paths, user_positive_items,
        user_interaction_counts, device,
        cold_threshold=COLD_THRESHOLD
    )

    score = compute_resilience_score(
        interaction_results,
        primary_metric=PRIMARY_METRIC,
        threshold=p25
    )

    output = {
        'interaction_results': {str(k): v for k, v in interaction_results.items()},
        'coverage_results'   : coverage_results,
        'embedding_results'  : embedding_results,
        'resilience_score'   : score,
        'config': {
            'cold_threshold'  : COLD_THRESHOLD,
            'auto_thresholds' : thresholds,
            'primary_metric'  : PRIMARY_METRIC,
            'k_values'        : K_VALUES,
        }
    }
    out = "/kaggle/working/cold_start_results.json"
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
    

# %% [code]
# ============================================================================
# CELL 23.5: CONTRASTIVE LEARNING IMPLEMENTATION
# ============================================================================
# INSERT THIS AS A NEW CELL BETWEEN CELL 23 AND CELL 24

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
import random

# ============================================================================
# 1. PATH AUGMENTATION STRATEGIES
# ============================================================================

class PathAugmenter:
    """Generate augmented views of paths for contrastive learning."""
    
    @staticmethod
    def node_dropout(path_embeddings: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
        """Randomly drop nodes from path (except start/end)."""
        if len(path_embeddings) <= 2:
            return path_embeddings
        
        mask = np.random.random(len(path_embeddings)) > drop_prob
        mask[0] = True  # Keep start
        mask[-1] = True  # Keep end
        
        return path_embeddings[mask]
    
    @staticmethod
    def embedding_noise(path_embeddings: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to embeddings."""
        noise = np.random.normal(0, noise_std, path_embeddings.shape)
        return path_embeddings + noise.astype(np.float32)
    
    @staticmethod
    def subpath_sampling(path_embeddings: np.ndarray, min_length: int = 2) -> np.ndarray:
        """Sample a contiguous subpath."""
        path_len = len(path_embeddings)
        if path_len <= min_length:
            return path_embeddings
        
        max_start = path_len - min_length
        start = np.random.randint(0, max_start + 1)
        end = np.random.randint(start + min_length, path_len + 1)
        
        return path_embeddings[start:end]
    
    @staticmethod
    def get_two_views(path_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two augmented views of the same path."""
        # View 1: Node dropout + noise
        view1 = PathAugmenter.node_dropout(path_embeddings, drop_prob=0.1)
        view1 = PathAugmenter.embedding_noise(view1, noise_std=0.05)
        
        # View 2: Different dropout + noise
        view2 = PathAugmenter.node_dropout(path_embeddings, drop_prob=0.15)
        view2 = PathAugmenter.embedding_noise(view2, noise_std=0.08)
        
        return view1, view2


# ============================================================================
# 2. CONTRASTIVE DATASET
# ============================================================================

class ContrastivePathDataset(Dataset):
    """Dataset for contrastive learning on paths."""
    
    def __init__(self, enriched_paths: List[Dict], augment: bool = True):
        self.paths = []
        
        for path in enriched_paths:
            emb = path.get('embeddings')
            if emb is not None and len(emb) > 0:
                if isinstance(emb, list):
                    emb = np.stack(emb).astype(np.float32)
                elif isinstance(emb, np.ndarray):
                    emb = emb.astype(np.float32)
                else:
                    continue
                
                self.paths.append(emb)
        
        self.augment = augment
        print(f"Contrastive dataset: {len(self.paths)} paths")
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path_emb = self.paths[idx]
        
        if self.augment:
            view1, view2 = PathAugmenter.get_two_views(path_emb)
            return view1, view2
        else:
            return path_emb


# ============================================================================
# 3. CONTRASTIVE LOSS (InfoNCE)
# ============================================================================

class InfoNCELoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: [batch_size, dim] - embeddings of view 1
            z_j: [batch_size, dim] - embeddings of view 2
        
        Returns:
            loss: scalar
        """
        batch_size = z_i.size(0)
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate views
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create mask to exclude self-similarities
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        
        # Mask out self-similarities
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        # Get positive pairs
        pos_sim = exp_sim[torch.arange(2 * batch_size, device=z.device), positives]
        
        # Sum of all similarities (excluding self)
        sum_sim = exp_sim.sum(dim=1)
        
        # Compute loss
        loss = -torch.log(pos_sim / sum_sim + 1e-8).mean()
        
        return loss


# ============================================================================
# 4. CONTRASTIVE TRAINING FUNCTION
# ============================================================================

def pretrain_with_contrastive_learning(
    enriched_paths: List[Dict],
    path_lstm: nn.Module,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    temperature: float = 0.07
) -> nn.Module:
    """
    Pretrain path encoder using contrastive learning.
    
    Args:
        enriched_paths: All paths (with embeddings)
        path_lstm: PathLSTM model to pretrain
        device: cuda/cpu
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        temperature: Temperature for InfoNCE loss
    
    Returns:
        Pretrained path_lstm
    """
    print(f"\n{'='*70}")
    print("CONTRASTIVE PRETRAINING")
    print(f"{'='*70}")
    print(f"Paths: {len(enriched_paths)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    
    # Create dataset
    dataset = ContrastivePathDataset(enriched_paths, augment=True)
    
    if len(dataset) == 0:
        print("⚠️ No valid paths for contrastive learning!")
        return path_lstm
    
    # Collate function for variable-length paths
    def collate_fn(batch):
        views1 = [item[0] for item in batch]
        views2 = [item[1] for item in batch]
        return views1, views2
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True  # For stable batch size
    )
    
    # Setup
    path_lstm = path_lstm.to(device)
    path_lstm.train()
    
    optimizer = torch.optim.Adam(path_lstm.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=temperature)
    
    # Training loop
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Add progress bar
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for views1, views2 in pbar:
            # Skip empty batches
            if len(views1) == 0 or len(views2) == 0:
                continue
                
            optimizer.zero_grad()
            
            # Encode view 1
            z_i_list = []
            for path_emb in views1:
                path_tensor = torch.from_numpy(path_emb).float().to(device)
                z, _ = path_lstm(path_tensor)
                z_i_list.append(z)
            
            # Skip if no valid encodings
            if len(z_i_list) == 0:
                continue
                
            z_i = torch.stack(z_i_list)  # [batch_size, hidden_dim]
            
            # Encode view 2
            z_j_list = []
            for path_emb in views2:
                path_tensor = torch.from_numpy(path_emb).float().to(device)
                z, _ = path_lstm(path_tensor)
                z_j_list.append(z)
            
            # Skip if no valid encodings
            if len(z_j_list) == 0:
                continue
                
            z_j = torch.stack(z_j_list)  # [batch_size, hidden_dim]
            
            # Compute loss
            loss = criterion(z_i, z_j)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(path_lstm.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Batches: {num_batches}")
 
    print(f"✓ Contrastive pretraining complete!")
    return path_lstm

print("✓ Contrastive learning components loaded")

# %% [markdown]
# ## Phase 3: Recommendation Model Training
# 
# ### Step 3.1: Model Architecture & Training Components
# - **EarlyStopping**: Prevents overfitting with patience mechanism
# - **ImprovedDataset**: Balanced positive/negative sampling with minimal augmentation
# - **RegularizedPathLSTM**: LSTM with LayerNorm and increased dropout (0.4)
# - **BPRLoss**: Bayesian Personalized Ranking with L2 regularization

# %% [code]
# ============================================================
# CELL 24: KEY FIXES FOR OVERFITTING
# ============================================================
# 1. Early stopping with patience
# 2. Stronger regularization (dropout, weight decay, L2)
# 3. Reduced model complexity
# 4. Better negative sampling strategy
# 5. Learning rate scheduling
# 6. Gradient clipping
# 7. Better train/val split validation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Tuple
import copy


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        
        return self.early_stop


# ============================================================
# IMPROVED NEGATIVE SAMPLING
# ============================================================

class ImprovedPathFinetuningDataset(Dataset):
    """Improved dataset with better negative sampling."""
    
    def __init__(self, enriched_paths: List[Dict], negative_ratio: int = 3):
        self.negative_ratio = negative_ratio
        self.positive_paths = []
        self.negative_paths = []
        
        for path in enriched_paths:
            if not path.get('embeddings') or len(path['embeddings']) == 0:
                continue
                
            rating = path.get('rating')
            has_actual = path.get('has_actual_rating', False)
            
            if has_actual:
                path_copy = path.copy()
                path_copy['embeddings'] = np.stack(path['embeddings']).astype(np.float32)
                
                if rating >= 3.0:
                    self.positive_paths.append(path_copy)
                else:
                    self.negative_paths.append(path_copy)
        
        print(f"Improved Fine-tuning Dataset:")
        print(f"  Positive: {len(self.positive_paths)}")
        print(f"  Negative: {len(self.negative_paths)}")
        
        # ONLY augment if we have very few real negatives
        if len(self.negative_paths) < len(self.positive_paths):
            self._minimal_augmentation()
    
    def _minimal_augmentation(self):
        """Only augment to match positive count, not multiply by negative_ratio."""
        target = len(self.positive_paths)
        current = len(self.negative_paths)
        shortage = target - current
        
        if shortage > 0:
            print(f"  Minimal augmentation: {shortage} negatives")
            for _ in range(shortage):
                # Strategy 1: Add noise to a random positive (80% of time)
                if random.random() < 0.8:
                    pos_path = random.choice(self.positive_paths)
                    
                    neg_path = pos_path.copy()
                    neg_path['rating'] = random.uniform(1.0, 2.9)
                    neg_path['has_actual_rating'] = False
                    neg_path['is_synthetic'] = True
                    
                    # Add controlled noise
                    noise_scale = random.uniform(0.05, 0.15)
                    noise = np.random.normal(0, noise_scale, pos_path['embeddings'].shape).astype(np.float32)
                    neg_path['embeddings'] = pos_path['embeddings'] + noise
                
                # Strategy 2: Shuffle path order (20% of time)
                else:
                    pos_path = random.choice(self.positive_paths)
                    
                    neg_path = pos_path.copy()
                    neg_path['rating'] = random.uniform(1.0, 2.9)
                    neg_path['has_actual_rating'] = False
                    neg_path['is_synthetic'] = True
                    
                    # Shuffle embeddings to create invalid path ordering
                    shuffled_embeddings = pos_path['embeddings'].copy()
                    if len(shuffled_embeddings) > 2:
                        # Keep start and end, shuffle middle
                        middle = shuffled_embeddings[1:-1].copy()
                        np.random.shuffle(middle)
                        shuffled_embeddings[1:-1] = middle
                    
                    neg_path['embeddings'] = shuffled_embeddings
                
                self.negative_paths.append(neg_path)
    
    def __len__(self):
        return len(self.positive_paths)
    
    def __getitem__(self, idx):
        pos_path = self.positive_paths[idx]
        
        # Sample from ALL negatives, not just a fixed ratio
        num_negatives = min(self.negative_ratio, len(self.negative_paths))
        neg_paths = random.sample(self.negative_paths, num_negatives)
        
        return {
            'positive': pos_path,
            'negatives': neg_paths
        }


# ============================================================
# REGULARIZED MODEL (REDUCED COMPLEXITY)
# ============================================================

class RegularizedPathLSTM(nn.Module):
    """LSTM with stronger regularization."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Add dropout between LSTM layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Simplified attention
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward_single(self, path_embeddings):
        seq_len = path_embeddings.size(0)
        embed_dim = path_embeddings.size(1)
        
        # Handle dimension mismatch
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                padding = torch.zeros(seq_len, self.embedding_dim - embed_dim, 
                                     device=path_embeddings.device, dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, padding], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        
        x = path_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        if seq_len == 1:
            return lstm_out.squeeze(0), torch.ones(1, device=path_embeddings.device)
        
        # Attention
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=0)
        path_repr = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=0)
        
        return path_repr, attention_weights
    
    def forward_batch(self, path_embeddings_batch, lengths):
        batch_size, max_len, embed_dim = path_embeddings_batch.shape
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            path_embeddings_batch, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        path_reprs = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return path_reprs, attention_weights
    
    def forward(self, path_embeddings, lengths=None):
        if lengths is not None:
            return self.forward_batch(path_embeddings, lengths)
        else:
            return self.forward_single(path_embeddings)


class RegularizedPathRecommendationModel(nn.Module):
    """Model with reduced complexity and stronger regularization."""
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 lstm_hidden_dim: int = 128,  # REDUCED from 256
                 attention_dim: int = 64,      # REDUCED from 128
                 mlp_dims: List[int] = [128, 64],  # REDUCED layers
                 dropout: float = 0.4):        # INCREASED from 0.3
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_dim = attention_dim
        
        self.path_lstm = RegularizedPathLSTM(embedding_dim, lstm_hidden_dim, dropout)
        
        # Path set aggregation
        self.query_proj = nn.Linear(lstm_hidden_dim, attention_dim)
        self.key_proj = nn.Linear(lstm_hidden_dim, attention_dim)
        self.value_proj = nn.Linear(lstm_hidden_dim, attention_dim)
        
        # Simplified MLP with stronger regularization
        mlp_layers = []
        input_dim = attention_dim
        for hidden_dim in mlp_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def encode_paths_batch(self, all_paths: List[List[np.ndarray]]) -> List[List[torch.Tensor]]:
        device = next(self.parameters()).device
        
        flat_paths = []
        path_map = []
        
        for sample_idx, paths in enumerate(all_paths):
            for path_idx, path_emb in enumerate(paths):
                if isinstance(path_emb, np.ndarray) and len(path_emb) > 0:
                    flat_paths.append(path_emb)
                    path_map.append((sample_idx, path_idx))
        
        if len(flat_paths) == 0:
            return [[] for _ in all_paths]
        
        length_groups = {}
        for idx, path in enumerate(flat_paths):
            length = len(path)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((idx, path))
        
        path_representations = [None] * len(flat_paths)
        
        for length, group in length_groups.items():
            indices = [item[0] for item in group]
            paths = [item[1] for item in group]
            
            batch_tensor = torch.from_numpy(np.stack(paths)).float().to(device)
            lengths = torch.full((len(paths),), length, dtype=torch.long, device=device)
            
            batch_reprs, _ = self.path_lstm(batch_tensor, lengths)
            
            for i, repr_tensor in enumerate(batch_reprs):
                path_representations[indices[i]] = repr_tensor
        
        result = [[] for _ in all_paths]
        for flat_idx, (sample_idx, path_idx) in enumerate(path_map):
            if path_representations[flat_idx] is not None:
                result[sample_idx].append(path_representations[flat_idx])
        
        return result
    
    def aggregate_path_sets_batch(self, all_path_reprs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[int]]:
        device = next(self.parameters()).device
        batch_size = len(all_path_reprs)
        
        max_paths = max(len(reprs) for reprs in all_path_reprs if len(reprs) > 0)
        if max_paths == 0:
            return torch.zeros(batch_size, self.attention_dim, device=device), [0] * batch_size
        
        padded_reprs = torch.zeros(batch_size, max_paths, self.lstm_hidden_dim, device=device)
        path_counts = []
        
        for i, reprs in enumerate(all_path_reprs):
            if len(reprs) > 0:
                stacked = torch.stack(reprs)
                padded_reprs[i, :len(reprs)] = stacked
                path_counts.append(len(reprs))
            else:
                path_counts.append(1)
        
        path_counts = torch.tensor(path_counts, device=device)
        
        Q = self.query_proj(padded_reprs)
        K = self.key_proj(padded_reprs)
        V = self.value_proj(padded_reprs)
        
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
        
        mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(-1, max_paths, -1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=2)
        L_prime = torch.bmm(attention_weights, V)
        L_mp = torch.max(L_prime, dim=1)[0]
        
        best_indices = torch.argmax(attention_weights.sum(dim=1), dim=1).tolist()
        
        return L_mp, best_indices
    
    def forward(self, paths_batch: List[List[np.ndarray]]) -> Tuple[torch.Tensor, List[int]]:
        all_path_reprs = self.encode_paths_batch(paths_batch)
        L_mp_batch, best_indices = self.aggregate_path_sets_batch(all_path_reprs)
        
        scores = self.mlp(L_mp_batch).squeeze(-1)
        predictions = torch.sigmoid(scores)
        
        return predictions, best_indices


# ============================================================
# IMPROVED TRAINING WITH EARLY STOPPING
# ============================================================

def finetune_model_with_early_stopping(model, train_loader, val_loader, optimizer, 
                                       criterion, device, scheduler=None, num_epochs=50, 
                                       patience=7):
    """Fine-tuning with early stopping and learning rate scheduling."""
    model.to(device)
    print("\nPHASE 2: FINE-TUNING (with Early Stopping)")
    
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for pos_paths, neg_paths_list in train_loader:
            optimizer.zero_grad()
            
            pos_path_sets = [[path['embeddings']] for path in pos_paths]
            pos_scores, _ = model(pos_path_sets)
            
            all_neg_paths = []
            neg_path_counts = []
            
            for neg_paths in neg_paths_list:
                neg_path_counts.append(len(neg_paths))
                for neg_path in neg_paths:
                    all_neg_paths.append([neg_path['embeddings']])
            
            if len(all_neg_paths) > 0:
                all_neg_scores, _ = model(all_neg_paths)
                
                neg_scores_list = []
                start_idx = 0
                for count in neg_path_counts:
                    end_idx = start_idx + count
                    neg_scores_list.append(all_neg_scores[start_idx:end_idx])
                    start_idx = end_idx
                
                max_negs = max(len(scores) for scores in neg_scores_list)
                padded_neg_scores = []
                for scores in neg_scores_list:
                    if len(scores) < max_negs:
                        padding = torch.full((max_negs - len(scores),), 0.3, device=device)
                        padded_scores = torch.cat([scores, padding])
                    else:
                        padded_scores = scores
                    padded_neg_scores.append(padded_scores)
                
                neg_scores = torch.stack(padded_neg_scores)
            else:
                neg_scores = torch.zeros(len(pos_scores), 1, device=device) + 0.3
            
            loss, bpr_loss, reg_loss = criterion(pos_scores, neg_scores, model.parameters())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Stricter clipping
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                try:
                    pos_paths, neg_paths_list = batch_data
                    
                    if len(pos_paths) == 0:
                        continue
                    
                    pos_path_sets = [[path['embeddings']] for path in pos_paths]
                    pos_scores, _ = model(pos_path_sets)
                    
                    all_neg_paths = []
                    neg_path_counts = []
                    
                    for neg_paths in neg_paths_list:
                        neg_path_counts.append(len(neg_paths))
                        for neg_path in neg_paths:
                            all_neg_paths.append([neg_path['embeddings']])
                    
                    if len(all_neg_paths) > 0:
                        all_neg_scores, _ = model(all_neg_paths)
                        
                        neg_scores_list = []
                        start_idx = 0
                        for count in neg_path_counts:
                            end_idx = start_idx + count
                            neg_scores_list.append(all_neg_scores[start_idx:end_idx])
                            start_idx = end_idx
                        
                        max_negs = max(len(scores) for scores in neg_scores_list)
                        padded_neg_scores = []
                        for scores in neg_scores_list:
                            if len(scores) < max_negs:
                                padding = torch.full((max_negs - len(scores),), 0.3, device=device)
                                padded_scores = torch.cat([scores, padding])
                            else:
                                padded_scores = scores
                            padded_neg_scores.append(padded_scores)
                        
                        neg_scores = torch.stack(padded_neg_scores)
                    else:
                        neg_scores = torch.zeros(len(pos_scores), 1, device=device) + 0.3
                    
                    loss, _, _ = criterion(pos_scores, neg_scores, model.parameters())
                    total_val_loss += loss.item()
                    num_val_batches += 1
                    
                except Exception as e:
                    continue
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if num_val_batches == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: NO VALID BATCHES, LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        if avg_val_loss < best_val_loss and num_val_batches > 0:
            best_val_loss = avg_val_loss
        
        # Early stopping check
        if num_val_batches > 0:
            if early_stopping(avg_val_loss, model):
                print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                # Restore best model
                model.load_state_dict(early_stopping.best_model_state)
                break
        
        torch.cuda.empty_cache()
    
    return best_val_loss


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def load_enriched_chunks(chunks_dir: str) -> List[Dict]:
    """Load enriched path chunks from directory."""
    from pathlib import Path
    import pickle
    
    chunks_path = Path(chunks_dir)
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {chunks_path}")
    
    chunk_files = sorted(chunks_path.glob('enriched_poi_paths_chunk_*.pkl'))
    
    if len(chunk_files) == 0:
        chunk_files = sorted(chunks_path.glob('*chunk*.pkl'))
        
    if len(chunk_files) == 0:
        raise FileNotFoundError(f"No enriched chunk files found in {chunks_path}")
    
    all_paths = []
    print(f"Loading {len(chunk_files)} files...")
    
    for i, chunk_file in enumerate(chunk_files):
        try:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
                all_paths.extend(chunk_data)
            
            if (i + 1) % max(1, len(chunk_files) // 5) == 0:
                print(f"  {((i+1)/len(chunk_files))*100:.0f}% complete")
        except Exception as e:
            print(f"  Warning: Failed to load {chunk_file.name}")
            continue
    
    print(f"Loaded {len(all_paths):,} paths\n")
    
    if len(all_paths) == 0:
        raise ValueError("No paths were loaded!")
    
    return all_paths


def split_paths_by_user_poi(all_paths: List[Dict], val_ratio: float = 0.2, seed: int = 42):
    """Split paths into train/val by user-poi pairs."""
    random.seed(seed)
    np.random.seed(seed)
    
    user_poi_paths = defaultdict(list)
    
    for path in all_paths:
        user_id = path.get('user_global_idx', path.get('user_idx', path.get('user_id', 'unknown')))
        poi_id = path.get('poi_global_idx', path.get('poi_idx', path.get('poi_id', 'unknown')))
        
        key = (user_id, poi_id)
        user_poi_paths[key].append(path)
    
    print(f"\nTrain/Val Split Statistics:")
    print(f"  Total paths: {len(all_paths):,}")
    print(f"  Unique user-poi pairs: {len(user_poi_paths):,}")
    
    all_pairs = list(user_poi_paths.keys())
    random.shuffle(all_pairs)
    
    val_size = int(len(all_pairs) * val_ratio)
    val_pairs = set(all_pairs[:val_size])
    train_pairs = set(all_pairs[val_size:])
    
    print(f"  Train user-poi pairs: {len(train_pairs):,}")
    print(f"  Val user-poi pairs: {len(val_pairs):,}")
    
    train_paths = []
    val_paths = []
    
    for pair, paths in user_poi_paths.items():
        if pair in val_pairs:
            val_paths.extend(paths)
        else:
            train_paths.extend(paths)
    
    print(f"  Train paths: {len(train_paths):,}")
    print(f"  Val paths: {len(val_paths):,}")
    print()
    
    return train_paths, val_paths


# ============================================================
# USAGE EXAMPLE
# ============================================================

def train_with_fixes(enriched_paths, finetune_train_paths, finetune_val_paths, device):
    """Main training function with all fixes applied."""
    
    # Hyperparameters (ADJUSTED for less overfitting)
    EMBEDDING_DIM = 64
    LSTM_HIDDEN_DIM = 128      # REDUCED
    ATTENTION_DIM = 64         # REDUCED
    MLP_DIMS = [128, 64]       # REDUCED
    DROPOUT = 0.4              # INCREASED
    
    FINETUNE_EPOCHS = 50       # More epochs but with early stopping
    FINETUNE_BATCH_SIZE = 128  # REDUCED for better generalization
    FINETUNE_LR = 0.0001       # REDUCED initial LR
    NEGATIVE_RATIO = 3         # REDUCED
    LAMBDA_REG = 1e-4          # INCREASED regularization
    
    PATIENCE = 7               # Early stopping patience

    
    print("\n" + "="*70)
    print("STEP 1: CONTRASTIVE PRETRAINING")
    print("="*70)
    
    # Create path encoder
    path_lstm = RegularizedPathLSTM(
        embedding_dim=64,
        hidden_dim=128,
        dropout=0.4
    )
    
    # Pretrain with contrastive learning on ALL paths
    path_lstm = pretrain_with_contrastive_learning(
        enriched_paths=enriched_paths,  # ✓ FIXED
        path_lstm=path_lstm,
        device=device,
        epochs=5,
        batch_size=128,
        lr=1e-3,
        temperature=0.07
    )
        
    print("\n" + "="*70)
    print("STEP 2: SUPERVISED FINE-TUNING")
    print("="*70)

    # Initialize model
    model = RegularizedPathRecommendationModel(
        embedding_dim=EMBEDDING_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        mlp_dims=MLP_DIMS,
        dropout=DROPOUT
    )

    model.path_lstm = path_lstm
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Datasets with improved negative sampling
    train_dataset = ImprovedPathFinetuningDataset(finetune_train_paths, negative_ratio=NEGATIVE_RATIO)
    val_dataset = ImprovedPathFinetuningDataset(finetune_val_paths, negative_ratio=NEGATIVE_RATIO)
    
    train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, 
                              shuffle=True, collate_fn=finetune_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, 
                            shuffle=False, collate_fn=finetune_collate_fn)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, 
                                  weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    # Loss with stronger regularization
    criterion = BPRLoss(lambda_reg=LAMBDA_REG)
    
    # Train with early stopping
    best_val_loss = finetune_model_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        num_epochs=FINETUNE_EPOCHS,
        patience=PATIENCE
    )
    
    return model, best_val_loss


def finetune_collate_fn(batch):
    positive_paths = []
    negative_paths = []
    
    for sample in batch:
        positive_paths.append(sample['positive'])
        negative_paths.append(sample['negatives'])
    
    return positive_paths, negative_paths


class BPRLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, pos_scores, neg_scores, model_params):
        pos_scores_expanded = pos_scores.unsqueeze(1)
        score_diff = pos_scores_expanded - neg_scores
        bpr_loss = -torch.log(torch.sigmoid(score_diff) + 1e-10).mean()
        
        l2_reg = 0
        for param in model_params:
            l2_reg += torch.norm(param, p=2)
        
        total_loss = bpr_loss + self.lambda_reg * l2_reg
        
        return total_loss, bpr_loss, l2_reg


# ============================================================
# MAIN FUNCTION
# ============================================================

if __name__ == "__main__":
    import pickle
    import json
    from pathlib import Path
    import gc
    
    # Configuration
    ENRICHED_CHUNKS_DIR = "/kaggle/working/results"
    VAL_RATIO = 0.2
    RANDOM_SEED = 42
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"{'='*70}\n")
    
    # Load data
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    all_paths = load_enriched_chunks(ENRICHED_CHUNKS_DIR)
    
    # Extract labeled paths for fine-tuning
    print("\nExtracting labeled paths for fine-tuning split...")
    labeled_paths = []
    for path in all_paths:
        if path.get('embeddings') and len(path['embeddings']) > 0:
            rating = path.get('rating')
            has_actual = path.get('has_actual_rating', False)
            if has_actual:
                labeled_paths.append(path)
    
    print(f"Found {len(labeled_paths):,} labeled paths")
    
    if len(labeled_paths) == 0:
        raise ValueError("No labeled paths found! Cannot perform fine-tuning.")
    
    # DIAGNOSTIC: Analyze user-poi pair distribution
    print("\n" + "="*70)
    print("DIAGNOSTIC: Analyzing labeled paths structure")
    print("="*70)
    
    user_poi_pairs = defaultdict(list)
    user_ids_seen = set()
    poi_ids_seen = set()
    
    for path in labeled_paths:
        user_id = path.get('user_global_idx', path.get('user_idx', 'unknown'))
        poi_id = path.get('poi_global_idx', path.get('poi_idx', 'unknown'))
        
        user_ids_seen.add(user_id)
        poi_ids_seen.add(poi_id)
        
        key = (user_id, poi_id)
        user_poi_pairs[key].append(path)
    
    print(f"Unique users: {len(user_ids_seen)}")
    print(f"Unique POIs: {len(poi_ids_seen)}")
    print(f"Unique user-poi pairs: {len(user_poi_pairs)}")
    
    if len(user_poi_pairs) > 0:
        paths_per_pair = [len(paths) for paths in user_poi_pairs.values()]
        print(f"Average paths per pair: {np.mean(paths_per_pair):.2f}")
        print(f"Min paths per pair: {np.min(paths_per_pair)}")
        print(f"Max paths per pair: {np.max(paths_per_pair)}")
        
        print("\nSample user-poi pairs (first 5):")
        for i, (key, paths) in enumerate(list(user_poi_pairs.items())[:5]):
            print(f"  {i+1}. User: {key[0]}, POI: {key[1]}, Paths: {len(paths)}")
    
    print("="*70)
    
    # Split strategy
    if len(user_poi_pairs) <= 1:
        print("\n⚠️ WARNING: Using RANDOM SPLIT (not enough unique user-poi pairs)")
        random.seed(RANDOM_SEED)
        random.shuffle(labeled_paths)
        
        val_size = int(len(labeled_paths) * VAL_RATIO)
        finetune_val_paths = labeled_paths[:val_size]
        finetune_train_paths = labeled_paths[val_size:]
        
        print(f"\nRandom Split Results:")
        print(f"  Train paths: {len(finetune_train_paths):,}")
        print(f"  Val paths: {len(finetune_val_paths):,}")
    else:
        print("\n✓ Using USER-POI SPLIT")
        finetune_train_paths, finetune_val_paths = split_paths_by_user_poi(
            labeled_paths, 
            val_ratio=VAL_RATIO, 
            seed=RANDOM_SEED
        )
    
    # Train model with fixes
    print("\n" + "="*70)
    print("TRAINING MODEL WITH ANTI-OVERFITTING FIXES")
    print("="*70)
    
    try:
        model, best_val_loss = train_with_fixes(
            enriched_paths=all_paths,
            finetune_train_paths=finetune_train_paths,
            finetune_val_paths=finetune_val_paths,
            device=device
        )
        
        # Save model
        model_path = "/kaggle/working/regularized_model_best.pt"
        torch.save({
            'config_name': 'regularized_full_model',
            'config': {
                'use_pretraining': False,  # We skipped pretraining in this script
                'use_attention_aggregation': True,
                'use_mlp': True,
                'description': 'Regularized model with early stopping'
            },
            'model_state_dict': model.state_dict(),
            'model_config': {
                'embedding_dim': 64,
                'lstm_hidden_dim': 128,
                'attention_dim': 64,
                'mlp_dims': [128, 64],
                'dropout': 0.4,
                'use_attention_aggregation': True,
                'use_mlp': True,
                'model_type': 'regularized'  # Flag to indicate this is the regularized version
            },
            'pretrain_val_loss': None,
            'finetune_val_loss': best_val_loss,
            'total_params': sum(p.numel() for p in model.parameters())
        }, model_path)
        
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETE!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Model saved: {model_path}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"{'='*70}\n")
        
        # Save training summary
        summary = {
            'best_val_loss': float(best_val_loss),
            'num_train_paths': len(finetune_train_paths),
            'num_val_paths': len(finetune_val_paths),
            'unique_users': len(user_ids_seen),
            'unique_pois': len(poi_ids_seen),
            'unique_user_poi_pairs': len(user_poi_pairs),
            'model_params': sum(p.numel() for p in model.parameters())
        }
        
        summary_path = "/kaggle/working/training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Training summary saved: {summary_path}")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n✓ Cleanup complete")

# %% [markdown]
# ## Phase 4: Model Evaluation
# 
# ### Step 4.1: Unified Evaluation Framework
# **Supports both architectures**:
# - `RegularizedPathLSTM` (reduced complexity, LayerNorm, dropout=0.4)
# - `OriginalPathLSTM` (baseline, BatchNorm, dropout=0.2)
# 
# **Metrics**: HR@K, Precision@K, Recall@K, F1@K, NDCG@K, MAP@K, MRR

# %% [code]
# ============================================================================
# CELL 25: REGULARIZATION
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import json


# ============================================================
# REGULARIZED MODEL ARCHITECTURE (From Training)
# ============================================================

class RegularizedPathLSTM(nn.Module):
    """LSTM with stronger regularization."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward_single(self, path_embeddings):
        seq_len = path_embeddings.size(0)
        embed_dim = path_embeddings.size(1)
        
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                padding = torch.zeros(seq_len, self.embedding_dim - embed_dim, 
                                     device=path_embeddings.device, dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, padding], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        
        x = path_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        if seq_len == 1:
            return lstm_out.squeeze(0), torch.ones(1, device=path_embeddings.device)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=0)
        path_repr = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=0)
        
        return path_repr, attention_weights
    
    def forward_batch(self, path_embeddings_batch, lengths):
        batch_size, max_len, embed_dim = path_embeddings_batch.shape
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            path_embeddings_batch, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        path_reprs = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return path_reprs, attention_weights
    
    def forward(self, path_embeddings, lengths=None):
        if lengths is not None:
            return self.forward_batch(path_embeddings, lengths)
        else:
            return self.forward_single(path_embeddings)


# ============================================================
# ORIGINAL MODEL ARCHITECTURE (From Ablation Study)
# ============================================================

class OriginalPathLSTM(nn.Module):
    """Original LSTM with two-layer attention mechanism."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           num_layers=1, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        
        self.attention_w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_b = nn.Parameter(torch.zeros(hidden_dim))
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward_single(self, path_embeddings):
        seq_len = path_embeddings.size(0)
        embed_dim = path_embeddings.size(1)
        
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                padding = torch.zeros(seq_len, self.embedding_dim - embed_dim, 
                                     device=path_embeddings.device, dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, padding], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        
        if seq_len == 1:
            x = path_embeddings.unsqueeze(0)
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            path_repr = lstm_out.squeeze(0).squeeze(0)
            return path_repr, torch.ones(1, device=path_embeddings.device)
        
        x = path_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.dropout(lstm_out)
        
        attention_input = self.attention_w(lstm_out) + self.attention_b
        attention_activated = F.relu(attention_input)
        attention_logits = self.attention_score(attention_activated).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=0)
        
        path_repr = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=0)
        return path_repr, attention_weights
    
    def forward_batch(self, path_embeddings_batch, lengths):
        batch_size, max_len, embed_dim = path_embeddings_batch.shape
        
        packed_input = nn.utils.rnn.pack_padded_sequence(
            path_embeddings_batch, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        
        attention_input = self.attention_w(lstm_out) + self.attention_b
        attention_activated = F.relu(attention_input)
        attention_logits = self.attention_score(attention_activated).squeeze(-1)
        
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        path_reprs = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        return path_reprs, attention_weights
    
    def forward(self, path_embeddings, lengths=None):
        if lengths is not None:
            return self.forward_batch(path_embeddings, lengths)
        else:
            return self.forward_single(path_embeddings)


# ============================================================
# UNIFIED MODEL WRAPPER
# ============================================================

class PathRecommendationModel(nn.Module):
    """Unified model that can use either architecture."""
    
    def __init__(self, 
                 embedding_dim: int = 64,
                 lstm_hidden_dim: int = 128,
                 attention_dim: int = 64,
                 mlp_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.2,
                 use_attention_aggregation: bool = True,
                 use_mlp: bool = True,
                 model_type: str = 'original'):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_dim = attention_dim
        self.use_attention_aggregation = use_attention_aggregation
        self.use_mlp = use_mlp
        self.model_type = model_type
        
        # Choose LSTM architecture based on model type
        if model_type == 'regularized':
            self.path_lstm = RegularizedPathLSTM(embedding_dim, lstm_hidden_dim, dropout)
        else:
            self.path_lstm = OriginalPathLSTM(embedding_dim, lstm_hidden_dim, dropout)
        
        if use_attention_aggregation:
            self.query_proj = nn.Linear(lstm_hidden_dim, attention_dim)
            self.key_proj = nn.Linear(lstm_hidden_dim, attention_dim)
            self.value_proj = nn.Linear(lstm_hidden_dim, attention_dim)
            mlp_input_dim = attention_dim
        else:
            mlp_input_dim = lstm_hidden_dim
        
        if use_mlp:
            mlp_layers = []
            input_dim = mlp_input_dim
            
            if model_type == 'regularized':
                # Regularized model uses LayerNorm
                for hidden_dim in mlp_dims:
                    mlp_layers.extend([
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    input_dim = hidden_dim
            else:
                # Original model uses BatchNorm
                for hidden_dim in mlp_dims:
                    mlp_layers.append(nn.Linear(input_dim, hidden_dim))
                    mlp_layers.append(nn.BatchNorm1d(hidden_dim))
                    mlp_layers.append(nn.ReLU())
                    mlp_layers.append(nn.Dropout(dropout))
                    input_dim = hidden_dim
            
            mlp_layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.score_layer = nn.Linear(mlp_input_dim, 1)
    
    def encode_paths_batch(self, all_paths: List[List[np.ndarray]]) -> List[List[torch.Tensor]]:
        device = next(self.parameters()).device
        
        flat_paths = []
        path_map = []
        
        for sample_idx, paths in enumerate(all_paths):
            for path_idx, path_emb in enumerate(paths):
                if isinstance(path_emb, np.ndarray) and len(path_emb) > 0:
                    flat_paths.append(path_emb)
                    path_map.append((sample_idx, path_idx))
        
        if len(flat_paths) == 0:
            return [[] for _ in all_paths]
        
        length_groups = {}
        for idx, path in enumerate(flat_paths):
            length = len(path)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append((idx, path))
        
        path_representations = [None] * len(flat_paths)
        
        for length, group in length_groups.items():
            indices = [item[0] for item in group]
            paths = [item[1] for item in group]
            
            batch_tensor = torch.from_numpy(np.stack(paths)).float().to(device)
            lengths = torch.full((len(paths),), length, dtype=torch.long, device=device)
            
            batch_reprs, _ = self.path_lstm(batch_tensor, lengths)
            
            for i, repr_tensor in enumerate(batch_reprs):
                path_representations[indices[i]] = repr_tensor
        
        result = [[] for _ in all_paths]
        for flat_idx, (sample_idx, path_idx) in enumerate(path_map):
            if path_representations[flat_idx] is not None:
                result[sample_idx].append(path_representations[flat_idx])
        
        return result
    
    def aggregate_path_sets_batch(self, all_path_reprs: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[int]]:
        device = next(self.parameters()).device
        batch_size = len(all_path_reprs)
        
        max_paths = max(len(reprs) for reprs in all_path_reprs if len(reprs) > 0)
        if max_paths == 0:
            output_dim = self.attention_dim if self.use_attention_aggregation else self.lstm_hidden_dim
            return torch.zeros(batch_size, output_dim, device=device), [0] * batch_size
        
        padded_reprs = torch.zeros(batch_size, max_paths, self.lstm_hidden_dim, device=device)
        path_counts = []
        
        for i, reprs in enumerate(all_path_reprs):
            if len(reprs) > 0:
                stacked = torch.stack(reprs)
                padded_reprs[i, :len(reprs)] = stacked
                path_counts.append(len(reprs))
            else:
                path_counts.append(1)
        
        path_counts = torch.tensor(path_counts, device=device)
        
        if self.use_attention_aggregation:
            Q = self.query_proj(padded_reprs)
            K = self.key_proj(padded_reprs)
            V = self.value_proj(padded_reprs)
            
            attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
            
            mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, max_paths, -1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
            
            attention_weights = F.softmax(attention_scores, dim=2)
            L_prime = torch.bmm(attention_weights, V)
            
            L_mp = torch.max(L_prime, dim=1)[0]
            best_indices = torch.argmax(attention_weights.sum(dim=1), dim=1).tolist()
        else:
            mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
            mask_expanded = mask.unsqueeze(2).expand_as(padded_reprs)
            
            masked_reprs = padded_reprs * mask_expanded
            L_mp = masked_reprs.sum(dim=1) / path_counts.unsqueeze(1)
            
            best_indices = [0] * batch_size
        
        return L_mp, best_indices
    
    def forward(self, paths_batch: List[List[np.ndarray]]) -> Tuple[torch.Tensor, List[int]]:
        all_path_reprs = self.encode_paths_batch(paths_batch)
        L_mp_batch, best_indices = self.aggregate_path_sets_batch(all_path_reprs)
        
        if self.use_mlp:
            scores = self.mlp(L_mp_batch).squeeze(-1)
        else:
            scores = self.score_layer(L_mp_batch).squeeze(-1)
        
        predictions = torch.sigmoid(scores)
        
        return predictions, best_indices


# ============================================================
# EVALUATION METRICS
# ============================================================

class RecommendationMetrics:
    """Compute recommendation metrics at different K values."""
    
    @staticmethod
    def hit_rate(recommended: List, ground_truth: Set, k: int) -> float:
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & ground_truth) > 0 else 0.0
    
    @staticmethod
    def precision(recommended: List, ground_truth: Set, k: int) -> float:
        top_k = set(recommended[:k])
        if len(top_k) == 0:
            return 0.0
        return len(top_k & ground_truth) / k
    
    @staticmethod
    def recall(recommended: List, ground_truth: Set, k: int) -> float:
        top_k = set(recommended[:k])
        if len(ground_truth) == 0:
            return 0.0
        return len(top_k & ground_truth) / len(ground_truth)
    
    @staticmethod
    def f1_score(recommended: List, ground_truth: Set, k: int) -> float:
        prec = RecommendationMetrics.precision(recommended, ground_truth, k)
        rec = RecommendationMetrics.recall(recommended, ground_truth, k)
        
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def ndcg(recommended: List, ground_truth: Set, k: int) -> float:
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mrr(recommended: List, ground_truth: Set) -> float:
        for i, item in enumerate(recommended):
            if item in ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def map_at_k(recommended: List, ground_truth: Set, k: int) -> float:
        if len(ground_truth) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(ground_truth), k)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_evaluation_data(enriched_paths: List[Dict]) -> Tuple[Dict, Dict, Set]:
    """Organize ALL paths by user-item pairs."""
    print("\n" + "="*60)
    print("Preparing evaluation data")
    print("="*60)
    
    user_item_paths = defaultdict(lambda: defaultdict(list))
    user_positive_items = defaultdict(set)
    pois_with_any_rating = set()
    
    stats = {
        'total_paths': 0,
        'valid_paths': 0,
        'with_ratings': 0,
        'positive_ratings': 0
    }
    
    for path in enriched_paths:
        stats['total_paths'] += 1
        
        if not path.get('embeddings') or len(path['embeddings']) == 0:
            continue
        
        stats['valid_paths'] += 1
        
        user_id = (path.get('user_idx') or path.get('user_global_idx') or 
                   path.get('user_id') or path.get('user'))
        item_id = (path.get('poi_idx') or path.get('poi_global_idx') or 
                   path.get('poi_id') or path.get('item_id'))
        
        rating = path.get('rating')
        has_actual = path.get('has_actual_rating', False)
        
        if user_id is None or item_id is None:
            continue
        
        user_item_paths[user_id][item_id].append(path)
        
        if has_actual and rating is not None:
            stats['with_ratings'] += 1
            pois_with_any_rating.add(item_id)
            if rating >= 3.0:
                stats['positive_ratings'] += 1
                user_positive_items[user_id].add(item_id)
    
    print(f"\nData Statistics:")
    print(f"  Total paths: {stats['total_paths']:,}")
    print(f"  Valid paths: {stats['valid_paths']:,}")
    print(f"  Paths with ratings: {stats['with_ratings']:,}")
    print(f"  Positive interactions: {stats['positive_ratings']:,}")
    print(f"  Unique users: {len(user_item_paths):,}")
    print(f"  Unique POIs with ratings: {len(pois_with_any_rating):,}")
    
    return dict(user_item_paths), dict(user_positive_items), pois_with_any_rating


# ============================================================
# RANKING GENERATION
# ============================================================

def generate_rankings_for_user(model: nn.Module,
                               user_id: int,
                               user_item_paths: Dict,
                               device: torch.device,
                               batch_size: int = 64) -> List[Tuple[int, float]]:
    """Rank ALL items this user has paths to."""
    model.eval()
    
    all_items = list(user_item_paths[user_id].keys())
    
    if len(all_items) == 0:
        return []
    
    rankings = []
    
    for i in range(0, len(all_items), batch_size):
        batch_items = all_items[i:i+batch_size]
        
        paths_batch = []
        valid_items = []
        
        for item_id in batch_items:
            paths = user_item_paths[user_id][item_id]
            
            path_embeddings = []
            for p in paths:
                emb = p.get('embeddings')
                if emb is not None and len(emb) > 0:
                    if isinstance(emb, list):
                        emb = np.array(emb)
                    path_embeddings.append(emb)
            
            if len(path_embeddings) > 0:
                paths_batch.append(path_embeddings)
                valid_items.append(item_id)
        
        if len(paths_batch) == 0:
            continue
        
        try:
            with torch.no_grad():
                scores, _ = model(paths_batch)
                scores = scores.cpu().numpy()
            
            for item_id, score in zip(valid_items, scores):
                rankings.append((item_id, float(score)))
        except Exception as e:
            continue
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings


# ============================================================
# EVALUATION PIPELINE
# ============================================================

def evaluate_ranking_model(model: nn.Module,
                          user_item_paths: Dict,
                          user_positive_items: Dict,
                          device: torch.device,
                          k_values: List[int] = [5, 10, 20, 50]) -> Dict:
    """Evaluate ranking quality."""
    print("\n" + "="*60)
    print("Evaluating ranking performance...")
    print("="*60)
    
    model.to(device)
    model.eval()
    
    metrics_collector = defaultdict(list)
    
    eval_users = [u for u in user_positive_items.keys() if len(user_positive_items[u]) > 0]
    
    print(f"Evaluating on {len(eval_users)} users with positive ratings\n")
    
    users_evaluated = 0
    users_skipped = 0
    
    for user_id in tqdm(eval_users, desc="Evaluating users"):
        positive_items = user_positive_items[user_id]
        
        if user_id not in user_item_paths:
            users_skipped += 1
            continue
        
        rankings = generate_rankings_for_user(
            model, user_id, user_item_paths, device
        )
        
        if len(rankings) == 0:
            users_skipped += 1
            continue
        
        ranked_items = [item for item, _ in rankings]
        num_positives = len(positive_items)
        num_total = len(ranked_items)
        
        if num_positives == num_total or num_positives == 0:
            users_skipped += 1
            continue
        
        users_evaluated += 1
        
        for k in k_values:
            metrics_collector[f'HR@{k}'].append(
                RecommendationMetrics.hit_rate(ranked_items, positive_items, k)
            )
            metrics_collector[f'Precision@{k}'].append(
                RecommendationMetrics.precision(ranked_items, positive_items, k)
            )
            metrics_collector[f'Recall@{k}'].append(
                RecommendationMetrics.recall(ranked_items, positive_items, k)
            )
            metrics_collector[f'F1@{k}'].append(
                RecommendationMetrics.f1_score(ranked_items, positive_items, k)
            )
            metrics_collector[f'NDCG@{k}'].append(
                RecommendationMetrics.ndcg(ranked_items, positive_items, k)
            )
            metrics_collector[f'MAP@{k}'].append(
                RecommendationMetrics.map_at_k(ranked_items, positive_items, k)
            )
        
        metrics_collector['MRR'].append(
            RecommendationMetrics.mrr(ranked_items, positive_items)
        )
    
    print(f"\nEvaluation complete:")
    print(f"  Users evaluated: {users_evaluated}")
    print(f"  Users skipped: {users_skipped}")
    
    if users_evaluated == 0:
        return {}
    
    results = {}
    for metric_name, values in metrics_collector.items():
        results[metric_name] = np.mean(values) if len(values) > 0 else 0.0
    
    return results


# ============================================================
# MAIN - UNIFIED EVALUATION
# ============================================================

def main():
    # Configuration
    MODEL_PATHS = [
        "/kaggle/working/regularized_model_best.pt"
    ]
    
    ENRICHED_CHUNKS_DIR = "/kaggle/working/results"
    K_VALUES = [5, 10, 20, 50]
    BATCH_SIZE = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data once
    print("\n" + "="*60)
    print("Loading enriched paths...")
    print("="*60)
    
    chunks_path = Path(ENRICHED_CHUNKS_DIR)
    chunk_files = sorted(chunks_path.glob('enriched_poi_paths_chunk_*.pkl'))
    
    if len(chunk_files) == 0:
        chunk_files = sorted(chunks_path.glob('*chunk*.pkl'))
    
    all_paths = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            all_paths.extend(chunk_data)
    
    print(f"Total paths loaded: {len(all_paths):,}")
    
    # Prepare data
    user_item_paths, user_positive_items, pois_with_any_rating = prepare_evaluation_data(all_paths)
    
    # Prune items without ratings
    print("\n" + "="*60)
    print("Pruning items without ratings...")
    print("="*60)
    
    pruned_user_item_paths = {}
    for user_id, items in user_item_paths.items():
        filtered = {item_id: paths for item_id, paths in items.items() if item_id in pois_with_any_rating}
        pruned_user_item_paths[user_id] = filtered
    
    print(f"Unique POIs with ratings: {len(pois_with_any_rating):,}")
    
    # Evaluate all models
    all_results = []
    
    print("\n" + "="*70)
    print("EVALUATING ALL MODELS")
    print("="*70)
    
    for model_path in MODEL_PATHS:
        model_name = Path(model_path).stem
        
        print("\n" + "#"*70)
        print(f"MODEL: {model_name}")
        print("#"*70)
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract configuration
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
            else:
                # Default config for old models
                config = {
                    'embedding_dim': 64,
                    'lstm_hidden_dim': 256,
                    'attention_dim': 128,
                    'mlp_dims': [256, 128, 64, 32],
                    'dropout': 0.3,
                    'use_attention_aggregation': True,
                    'use_mlp': True,
                    'model_type': 'original'
                }
            
            # Determine model type
            model_type = config.get('model_type', 'original')
            
            print(f"Model type: {model_type}")
            print(f"Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
            
            # Create model with correct architecture
            model = PathRecommendationModel(
                embedding_dim=config['embedding_dim'],
                lstm_hidden_dim=config['lstm_hidden_dim'],
                attention_dim=config['attention_dim'],
                mlp_dims=config['mlp_dims'],
                dropout=config['dropout'],
                use_attention_aggregation=config.get('use_attention_aggregation', True),
                use_mlp=config.get('use_mlp', True),
                model_type=model_type
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Model loaded successfully")
            
            # Evaluate
            results = evaluate_ranking_model(
                model=model,
                user_item_paths=pruned_user_item_paths,
                user_positive_items=user_positive_items,
                device=device,
                k_values=K_VALUES
            )
            
            if results:
                results['model_name'] = model_name
                results['model_path'] = model_path
                results['model_type'] = model_type
                
                # Add config info
                if 'config' in checkpoint:
                    ablation_config = checkpoint['config']
                    results['use_pretraining'] = ablation_config.get('use_pretraining', 'Unknown')
                    results['use_attention_aggregation'] = ablation_config.get('use_attention_aggregation', 'Unknown')
                    results['use_mlp'] = ablation_config.get('use_mlp', 'Unknown')
                    results['description'] = ablation_config.get('description', 'N/A')
                
                all_results.append(results)
                
                print(f"\n✓ Evaluation complete for {model_name}")
            else:
                print(f"\n⚠️  No results for {model_name}")
            
            # Clean up
            del model, checkpoint
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    if len(all_results) == 0:
        print("\n❌ No models were successfully evaluated!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    metric_cols = [col for col in results_df.columns if '@' in col or col == 'MRR']
    info_cols = ['model_name', 'model_type', 'description', 'use_pretraining', 'use_attention_aggregation', 'use_mlp']
    info_cols = [col for col in info_cols if col in results_df.columns]
    other_cols = [col for col in results_df.columns if col not in metric_cols and col not in info_cols]
    
    results_df = results_df[info_cols + metric_cols + other_cols]
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    # Print compact summary
    print("\n" + "-"*130)
    print(f"{'Model':<30} {'Type':<12} {'Pretrain':<10} {'Attention':<12} {'MLP':<8} {'HR@10':<8} {'NDCG@10':<10} {'MRR':<8}")
    print("-"*130)
    
    for _, row in results_df.iterrows():
        model_name = row['model_name'].replace('ablation_', '')[:28]
        model_type = str(row.get('model_type', 'original'))[:10]
        pretrain = str(row.get('use_pretraining', 'N/A'))[:8]
        attention = str(row.get('use_attention_aggregation', 'N/A'))[:10]
        mlp = str(row.get('use_mlp', 'N/A'))[:6]
        hr10 = f"{row.get('HR@10', 0):.4f}"
        ndcg10 = f"{row.get('NDCG@10', 0):.4f}"
        mrr = f"{row.get('MRR', 0):.4f}"
        
        print(f"{model_name:<30} {model_type:<12} {pretrain:<10} {attention:<12} {mlp:<8} {hr10:<8} {ndcg10:<10} {mrr:<8}")
    
    print("-"*130)
    
    # Print detailed metrics for each model
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['model_name']}")
        print("-" * 70)
        print(f"Model Type: {row.get('model_type', 'original')}")
        if 'description' in row:
            print(f"Description: {row['description']}")
        print(f"Configuration:")
        print(f"  Pretraining: {row.get('use_pretraining', 'N/A')}")
        print(f"  Attention Aggregation: {row.get('use_attention_aggregation', 'N/A')}")
        print(f"  MLP: {row.get('use_mlp', 'N/A')}")
        print(f"\nMetrics:")
        for k in K_VALUES:
            print(f"  K={k}:")
            print(f"    HR@{k}: {row.get(f'HR@{k}', 0):.4f}")
            print(f"    Precision@{k}: {row.get(f'Precision@{k}', 0):.4f}")
            print(f"    Recall@{k}: {row.get(f'Recall@{k}', 0):.4f}")
            print(f"    F1@{k}: {row.get(f'F1@{k}', 0):.4f}")
            print(f"    NDCG@{k}: {row.get(f'NDCG@{k}', 0):.4f}")
            print(f"    MAP@{k}: {row.get(f'MAP@{k}', 0):.4f}")
        print(f"  MRR: {row.get('MRR', 0):.4f}")
    
    # Save results to CSV
    output_file = 'unified_evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Save detailed JSON report
    json_output = {
        'summary': {
            'total_models_evaluated': len(all_results),
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'k_values': K_VALUES,
            'device': str(device)
        },
        'results': all_results
    }
    
    json_file = 'unified_evaluation_results.json'
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"✓ Detailed results saved to: {json_file}")
    
    # Find best model for each metric
    print("\n" + "="*70)
    print("BEST MODELS BY METRIC")
    print("="*70)
    
    for metric in metric_cols:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'model_name']
            best_score = results_df.loc[best_idx, metric]
            print(f"{metric:<20} {best_model:<30} {best_score:.4f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total models evaluated: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print(f"JSON report saved to: {json_file}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ### Step 4.2: Ablation Study Evaluation
# Evaluates all ablation models to analyze component contributions:
# - Full model (pretraining + attention + MLP)
# - No pretraining variations
# - Average aggregation vs attention
# - With/without MLP configurations
# 

# %% [code]
# ============================================================
# OPTIMIZED ABLATION STUDY WITH CONTRASTIVE LEARNING
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
import copy
import json
from pathlib import Path
import pickle
import time
from tqdm import tqdm
import gc


# ============================================================
# CONTRASTIVE LEARNING COMPONENTS
# ============================================================

class PathAugmenter:
    """Generate augmented views of paths for contrastive learning."""
    
    @staticmethod
    def node_dropout(path_embeddings: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
        if len(path_embeddings) <= 2:
            return path_embeddings
        mask = np.random.random(len(path_embeddings)) > drop_prob
        mask[0] = True
        mask[-1] = True
        return path_embeddings[mask]
    
    @staticmethod
    def embedding_noise(path_embeddings: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        noise = np.random.normal(0, noise_std, path_embeddings.shape)
        return path_embeddings + noise.astype(np.float32)
    
    @staticmethod
    def get_two_views(path_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        view1 = PathAugmenter.node_dropout(path_embeddings, drop_prob=0.1)
        view1 = PathAugmenter.embedding_noise(view1, noise_std=0.05)
        view2 = PathAugmenter.node_dropout(path_embeddings, drop_prob=0.15)
        view2 = PathAugmenter.embedding_noise(view2, noise_std=0.08)
        return view1, view2


class ContrastivePathDataset(Dataset):
    """Dataset for contrastive learning on paths."""
    
    def __init__(self, enriched_paths: List[Dict], augment: bool = True):
        self.paths = []
        for path in enriched_paths:
            emb = path.get('embeddings')
            if emb is not None and len(emb) > 0:
                if isinstance(emb, list):
                    emb = np.stack(emb).astype(np.float32)
                elif isinstance(emb, np.ndarray):
                    emb = emb.astype(np.float32)
                else:
                    continue
                self.paths.append(emb)
        self.augment = augment
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path_emb = self.paths[idx]
        if self.augment:
            view1, view2 = PathAugmenter.get_two_views(path_emb)
            return view1, view2
        else:
            return path_emb


class InfoNCELoss(nn.Module):
    """NT-Xent Loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(mask, 0)
        pos_sim = exp_sim[torch.arange(2 * batch_size, device=z.device), positives]
        sum_sim = exp_sim.sum(dim=1)
        loss = -torch.log(pos_sim / sum_sim + 1e-8).mean()
        return loss


def pretrain_path_lstm_contrastive(
    enriched_paths: List[Dict],
    path_lstm: nn.Module,
    device: torch.device,
    epochs: int = 3,  # Reduced for speed
    batch_size: int = 256,  # Larger batches
    lr: float = 1e-3,
    temperature: float = 0.07,
    verbose: bool = False
) -> nn.Module:
    """Fast contrastive pretraining."""
    
    if verbose:
        print(f"  Contrastive pretraining: {epochs} epochs")
    
    dataset = ContrastivePathDataset(enriched_paths, augment=True)
    
    if len(dataset) == 0:
        if verbose:
            print("  ⚠️ No valid paths!")
        return path_lstm
    
    def collate_fn(batch):
        views1 = [item[0] for item in batch]
        views2 = [item[1] for item in batch]
        return views1, views2
    
    # Use num_workers for faster loading
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    
    path_lstm = path_lstm.to(device)
    path_lstm.train()
    optimizer = torch.optim.Adam(path_lstm.parameters(), lr=lr)
    criterion = InfoNCELoss(temperature=temperature)
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        iterator = tqdm(loader, desc=f"  CL {epoch+1}/{epochs}", leave=False, ncols=80) if verbose else loader
        
        for views1, views2 in iterator:
            if len(views1) == 0 or len(views2) == 0:
                continue
            
            optimizer.zero_grad()
            
            # Batch encode views
            z_i_list = []
            for path_emb in views1:
                path_tensor = torch.from_numpy(path_emb).float().to(device)
                z, _ = path_lstm(path_tensor)
                z_i_list.append(z)
            
            if len(z_i_list) == 0:
                continue
            z_i = torch.stack(z_i_list)
            
            z_j_list = []
            for path_emb in views2:
                path_tensor = torch.from_numpy(path_emb).float().to(device)
                z, _ = path_lstm(path_tensor)
                z_j_list.append(z)
            
            if len(z_j_list) == 0:
                continue
            z_j = torch.stack(z_j_list)
            
            loss = criterion(z_i, z_j)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(path_lstm.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if verbose and num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"  CL Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    return path_lstm


# ============================================================
# CORE COMPONENTS
# ============================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = -val_loss if self.mode == 'min' else val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        return self.early_stop


class ImprovedPathFinetuningDataset(Dataset):
    def __init__(self, enriched_paths: List[Dict], negative_ratio: int = 3):
        self.negative_ratio = negative_ratio
        self.positive_paths = []
        self.negative_paths = []
        
        for path in enriched_paths:
            if not path.get('embeddings') or len(path['embeddings']) == 0:
                continue
            rating = path.get('rating')
            has_actual = path.get('has_actual_rating', False)
            if has_actual:
                path_copy = path.copy()
                path_copy['embeddings'] = np.stack(path['embeddings']).astype(np.float32)
                if rating >= 3.0:
                    self.positive_paths.append(path_copy)
                else:
                    self.negative_paths.append(path_copy)
        
        if len(self.negative_paths) < len(self.positive_paths):
            self._minimal_augmentation()
    
    def _minimal_augmentation(self):
        target = len(self.positive_paths)
        current = len(self.negative_paths)
        shortage = target - current
        
        if shortage > 0:
            for _ in range(shortage):
                pos_path = random.choice(self.positive_paths)
                neg_path = pos_path.copy()
                neg_path['rating'] = random.uniform(1.0, 2.9)
                neg_path['has_actual_rating'] = False
                neg_path['is_synthetic'] = True
                
                if random.random() < 0.8:
                    noise_scale = random.uniform(0.05, 0.15)
                    noise = np.random.normal(0, noise_scale, pos_path['embeddings'].shape).astype(np.float32)
                    neg_path['embeddings'] = pos_path['embeddings'] + noise
                else:
                    shuffled_embeddings = pos_path['embeddings'].copy()
                    if len(shuffled_embeddings) > 2:
                        middle = shuffled_embeddings[1:-1].copy()
                        np.random.shuffle(middle)
                        shuffled_embeddings[1:-1] = middle
                    neg_path['embeddings'] = shuffled_embeddings
                
                self.negative_paths.append(neg_path)
    
    def __len__(self):
        return len(self.positive_paths)
    
    def __getitem__(self, idx):
        pos_path = self.positive_paths[idx]
        num_negatives = min(self.negative_ratio, len(self.negative_paths))
        neg_paths = random.sample(self.negative_paths, num_negatives)
        return {'positive': pos_path, 'negatives': neg_paths}


class BPRLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1e-4):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(self, pos_scores, neg_scores, model_params):
        pos_scores_expanded = pos_scores.unsqueeze(1)
        score_diff = pos_scores_expanded - neg_scores
        bpr_loss = -torch.log(torch.sigmoid(score_diff) + 1e-10).mean()
        l2_reg = sum(torch.norm(param, p=2) for param in model_params)
        total_loss = bpr_loss + self.lambda_reg * l2_reg
        return total_loss, bpr_loss, l2_reg


def finetune_collate_fn(batch):
    positive_paths = [sample['positive'] for sample in batch]
    negative_paths = [sample['negatives'] for sample in batch]
    return positive_paths, negative_paths


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class PathLSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.4, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward_single(self, path_embeddings):
        seq_len, embed_dim = path_embeddings.size(0), path_embeddings.size(1)
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                padding = torch.zeros(seq_len, self.embedding_dim - embed_dim, 
                                     device=path_embeddings.device, dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, padding], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        
        x = path_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out.squeeze(0))
        lstm_out = self.dropout(lstm_out)
        
        if seq_len == 1:
            return lstm_out.squeeze(0), torch.ones(1, device=path_embeddings.device)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=0)
        path_repr = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=0)
        return path_repr, attention_weights
    
    def forward_batch(self, path_embeddings_batch, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            path_embeddings_batch, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        path_reprs = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return path_reprs, attention_weights
    
    def forward(self, path_embeddings, lengths=None):
        return self.forward_batch(path_embeddings, lengths) if lengths is not None else self.forward_single(path_embeddings)


class AblationPathRecommendationModel(nn.Module):
    """Configurable model for ablation study."""
    
    def __init__(self, embedding_dim=64, lstm_hidden_dim=128, attention_dim=64,
                 mlp_dims=[128, 64], dropout=0.4, use_lstm=True,
                 use_attention_aggregation=True, use_mlp=True, lstm_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_dim = attention_dim
        self.use_lstm = use_lstm
        self.use_attention_aggregation = use_attention_aggregation
        self.use_mlp = use_mlp
        
        if use_lstm:
            self.path_encoder = PathLSTM(embedding_dim, lstm_hidden_dim, dropout, lstm_layers)
            encoder_output_dim = lstm_hidden_dim
        else:
            self.path_encoder = None
            encoder_output_dim = embedding_dim
        
        if use_attention_aggregation:
            self.query_proj = nn.Linear(encoder_output_dim, attention_dim)
            self.key_proj = nn.Linear(encoder_output_dim, attention_dim)
            self.value_proj = nn.Linear(encoder_output_dim, attention_dim)
            aggregation_output_dim = attention_dim
        else:
            aggregation_output_dim = encoder_output_dim
        
        if use_mlp and len(mlp_dims) > 0:
            mlp_layers = []
            input_dim = aggregation_output_dim
            for hidden_dim in mlp_dims:
                mlp_layers.extend([
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                    nn.ReLU(), nn.Dropout(dropout)
                ])
                input_dim = hidden_dim
            mlp_layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = nn.Linear(aggregation_output_dim, 1)
    
    def encode_paths_batch(self, all_paths):
        device = next(self.parameters()).device
        flat_paths, path_map = [], []
        
        for sample_idx, paths in enumerate(all_paths):
            for path_idx, path_emb in enumerate(paths):
                if isinstance(path_emb, np.ndarray) and len(path_emb) > 0:
                    flat_paths.append(path_emb)
                    path_map.append((sample_idx, path_idx))
        
        if len(flat_paths) == 0:
            return [[] for _ in all_paths]
        
        if self.use_lstm:
            length_groups = {}
            for idx, path in enumerate(flat_paths):
                length = len(path)
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append((idx, path))
            
            path_representations = [None] * len(flat_paths)
            for length, group in length_groups.items():
                indices = [item[0] for item in group]
                paths = [item[1] for item in group]
                batch_tensor = torch.from_numpy(np.stack(paths)).float().to(device)
                lengths_tensor = torch.full((len(paths),), length, dtype=torch.long, device=device)
                batch_reprs, _ = self.path_encoder(batch_tensor, lengths_tensor)
                for i, repr_tensor in enumerate(batch_reprs):
                    path_representations[indices[i]] = repr_tensor
        else:
            path_representations = []
            for path in flat_paths:
                path_tensor = torch.from_numpy(path).float().to(device)
                if path_tensor.size(1) != self.embedding_dim:
                    if path_tensor.size(1) < self.embedding_dim:
                        padding = torch.zeros(path_tensor.size(0), 
                                            self.embedding_dim - path_tensor.size(1),
                                            device=device, dtype=path_tensor.dtype)
                        path_tensor = torch.cat([path_tensor, padding], dim=1)
                    else:
                        path_tensor = path_tensor[:, :self.embedding_dim]
                path_repr = path_tensor.mean(dim=0)
                path_representations.append(path_repr)
        
        result = [[] for _ in all_paths]
        for flat_idx, (sample_idx, path_idx) in enumerate(path_map):
            if path_representations[flat_idx] is not None:
                result[sample_idx].append(path_representations[flat_idx])
        return result
    
    def aggregate_path_sets_batch(self, all_path_reprs):
        device = next(self.parameters()).device
        batch_size = len(all_path_reprs)
        max_paths = max(len(reprs) for reprs in all_path_reprs if len(reprs) > 0)
        
        if max_paths == 0:
            output_dim = self.attention_dim if self.use_attention_aggregation else (
                self.lstm_hidden_dim if self.use_lstm else self.embedding_dim
            )
            return torch.zeros(batch_size, output_dim, device=device), [0] * batch_size
        
        repr_dim = all_path_reprs[0][0].size(0) if len(all_path_reprs[0]) > 0 else (
            self.lstm_hidden_dim if self.use_lstm else self.embedding_dim
        )
        padded_reprs = torch.zeros(batch_size, max_paths, repr_dim, device=device)
        path_counts = []
        
        for i, reprs in enumerate(all_path_reprs):
            if len(reprs) > 0:
                stacked = torch.stack(reprs)
                padded_reprs[i, :len(reprs)] = stacked
                path_counts.append(len(reprs))
            else:
                path_counts.append(1)
        
        path_counts = torch.tensor(path_counts, device=device)
        
        if self.use_attention_aggregation:
            Q = self.query_proj(padded_reprs)
            K = self.key_proj(padded_reprs)
            V = self.value_proj(padded_reprs)
            attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
            mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, max_paths, -1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
            attention_weights = F.softmax(attention_scores, dim=2)
            L_prime = torch.bmm(attention_weights, V)
            L_mp = torch.max(L_prime, dim=1)[0]
            best_indices = torch.argmax(attention_weights.sum(dim=1), dim=1).tolist()
        else:
            mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
            masked_reprs = padded_reprs * mask.unsqueeze(-1).float()
            L_mp = masked_reprs.sum(dim=1) / path_counts.unsqueeze(-1).float()
            best_indices = [0] * batch_size
        
        return L_mp, best_indices
    
    def forward(self, paths_batch):
        all_path_reprs = self.encode_paths_batch(paths_batch)
        L_mp_batch, best_indices = self.aggregate_path_sets_batch(all_path_reprs)
        scores = self.mlp(L_mp_batch).squeeze(-1)
        predictions = torch.sigmoid(scores)
        return predictions, best_indices


# ============================================================
# OPTIMIZED TRAINING FUNCTION
# ============================================================

def train_ablation_model(model, train_loader, val_loader, optimizer, criterion, 
                         device, scheduler=None, num_epochs=50, patience=5,
                         model_name="model", verbose=True):
    """Optimized training with timeout protection."""
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'epochs_trained': 0}
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss, num_train_batches = 0, 0
        
        # Add timeout per epoch
        epoch_start = time.time()
        max_epoch_time = 600  # 10 minutes max per epoch
        
        try:
            for pos_paths, neg_paths_list in train_loader:
                if time.time() - epoch_start > max_epoch_time:
                    print(f"  ⚠️ Epoch timeout at {num_train_batches} batches")
                    break
                
                optimizer.zero_grad()
                pos_path_sets = [[path['embeddings']] for path in pos_paths]
                pos_scores, _ = model(pos_path_sets)
                
                all_neg_paths, neg_path_counts = [], []
                for neg_paths in neg_paths_list:
                    neg_path_counts.append(len(neg_paths))
                    for neg_path in neg_paths:
                        all_neg_paths.append([neg_path['embeddings']])
                
                if len(all_neg_paths) > 0:
                    all_neg_scores, _ = model(all_neg_paths)
                    neg_scores_list = []
                    start_idx = 0
                    for count in neg_path_counts:
                        end_idx = start_idx + count
                        neg_scores_list.append(all_neg_scores[start_idx:end_idx])
                        start_idx = end_idx
                    max_negs = max(len(scores) for scores in neg_scores_list)
                    padded_neg_scores = []
                    for scores in neg_scores_list:
                        if len(scores) < max_negs:
                            padding = torch.full((max_negs - len(scores),), 0.3, device=device)
                            padded_scores = torch.cat([scores, padding])
                        else:
                            padded_scores = scores
                        padded_neg_scores.append(padded_scores)
                    neg_scores = torch.stack(padded_neg_scores)
                else:
                    neg_scores = torch.zeros(len(pos_scores), 1, device=device) + 0.3
                
                loss, bpr_loss, reg_loss = criterion(pos_scores, neg_scores, model.parameters())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_train_loss += loss.item()
                num_train_batches += 1
        except Exception as e:
            print(f"  ⚠️ Training error at epoch {epoch+1}: {e}")
            if num_train_batches == 0:
                break
        
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        
        # Validation
        model.eval()
        total_val_loss, num_val_batches = 0, 0
        with torch.no_grad():
            for batch_data in val_loader:
                try:
                    pos_paths, neg_paths_list = batch_data
                    if len(pos_paths) == 0:
                        continue
                    
                    pos_path_sets = [[path['embeddings']] for path in pos_paths]
                    pos_scores, _ = model(pos_path_sets)
                    
                    all_neg_paths, neg_path_counts = [], []
                    for neg_paths in neg_paths_list:
                        neg_path_counts.append(len(neg_paths))
                        for neg_path in neg_paths:
                            all_neg_paths.append([neg_path['embeddings']])
                    
                    if len(all_neg_paths) > 0:
                        all_neg_scores, _ = model(all_neg_paths)
                        neg_scores_list = []
                        start_idx = 0
                        for count in neg_path_counts:
                            end_idx = start_idx + count
                            neg_scores_list.append(all_neg_scores[start_idx:end_idx])
                            start_idx = end_idx
                        max_negs = max(len(scores) for scores in neg_scores_list)
                        padded_neg_scores = []
                        for scores in neg_scores_list:
                            if len(scores) < max_negs:
                                padding = torch.full((max_negs - len(scores),), 0.3, device=device)
                                padded_scores = torch.cat([scores, padding])
                            else:
                                padded_scores = scores
                            padded_neg_scores.append(padded_scores)
                        neg_scores = torch.stack(padded_neg_scores)
                    else:
                        neg_scores = torch.zeros(len(pos_scores), 1, device=device) + 0.3
                    
                    loss, _, _ = criterion(pos_scores, neg_scores, model.parameters())
                    total_val_loss += loss.item()
                    num_val_batches += 1
                except:
                    continue
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epochs_trained'] = epoch + 1
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss and num_val_batches > 0:
            best_val_loss = avg_val_loss
        
        if num_val_batches > 0:
            if early_stopping(avg_val_loss, model):
                if verbose:
                    print(f"  Early stop at epoch {epoch+1}")
                model.load_state_dict(early_stopping.best_model_state)
                break
        
        torch.cuda.empty_cache()
    
    return best_val_loss, history


# ============================================================
# CONFIGS FOR YOUR EVALUATION SET
# ============================================================

def get_evaluation_configs():
    """Your exact evaluation configurations with CL support."""
    base_config = {
        'embedding_dim': 64,
        'lstm_hidden_dim': 128,
        'attention_dim': 64,
        'mlp_dims': [128, 64],
        'dropout': 0.4,
        'use_lstm': True,
        'use_attention_aggregation': True,
        'use_mlp': True,
        'lstm_layers': 2,
        'use_contrastive_pretrain': True
    }

    configs = {
        'full_model': {
            **base_config,
            'description': 'Full model with all components + CL pretraining'
        },
        
        'no_pretraining': {
            **base_config,
            'use_contrastive_pretrain': False,
            'description': 'Full model WITHOUT contrastive pretraining'
        },
        
        'avg_aggregation': {
            **base_config,
            'use_attention_aggregation': False,
            'description': 'Mean pooling aggregation + CL'
        },
        
        'no_mlp': {
            **base_config,
            'mlp_dims': [],
            'description': 'Direct linear projection + CL'
        },
        
        'avg_no_mlp': {
            **base_config,
            'use_attention_aggregation': False,
            'mlp_dims': [],
            'description': 'Mean aggregation + direct projection + CL'
        },
        
        'no_pretrain_avg': {
            **base_config,
            'use_attention_aggregation': False,
            'use_contrastive_pretrain': False,
            'description': 'Mean aggregation WITHOUT pretraining'
        },
        
        'no_pretrain_no_mlp': {
            **base_config,
            'mlp_dims': [],
            'use_contrastive_pretrain': False,
            'description': 'No MLP WITHOUT pretraining'
        },
        
        'minimal_model': {
            **base_config,
            'use_lstm': False,
            'use_attention_aggregation': False,
            'mlp_dims': [],
            'description': 'Minimal: mean pooling everywhere + CL'
        },
    }

    return configs


# ============================================================
# MAIN ABLATION STUDY
# ============================================================

def run_ablation_study(all_paths, train_paths, val_paths, device, 
                       output_dir="./ablation_results", 
                       num_epochs=50, batch_size=128, patience=5):
    """Run complete ablation study with all optimizations."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("PREPARING DATASETS")
    print("="*70)
    
    train_dataset = ImprovedPathFinetuningDataset(train_paths, negative_ratio=3)
    val_dataset = ImprovedPathFinetuningDataset(val_paths, negative_ratio=3)
    
    # Use num_workers for speed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=finetune_collate_fn,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=finetune_collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    configs = get_evaluation_configs()
    results = {}
    
    print("\n" + "="*70)
    print(f"RUNNING ABLATION STUDY - {len(configs)} CONFIGURATIONS")
    print("="*70)
    
    for i, (config_name, config) in enumerate(configs.items(), 1):
        print(f"\n[{i}/{len(configs)}] {config_name}")
        print(f"{config['description']}")
        print("-" * 70)
        
        start_time = time.time()
        
        try:
            # Create model
            model = AblationPathRecommendationModel(
                embedding_dim=config['embedding_dim'],
                lstm_hidden_dim=config['lstm_hidden_dim'],
                attention_dim=config['attention_dim'],
                mlp_dims=config['mlp_dims'],
                dropout=config['dropout'],
                use_lstm=config['use_lstm'],
                use_attention_aggregation=config['use_attention_aggregation'],
                use_mlp=config['use_mlp'],
                lstm_layers=config['lstm_layers']
            )
        
            # Contrastive pretraining
            if config.get('use_contrastive_pretrain', False) and config['use_lstm']:
                print("  Applying contrastive pretraining...")
                model.path_encoder = pretrain_path_lstm_contrastive(
                    enriched_paths=all_paths,
                    path_lstm=model.path_encoder,
                    device=device,
                    epochs=3,  # Fast pretraining
                    batch_size=256,
                    lr=1e-3,
                    temperature=0.07,
                    verbose=True
                )
                print("  ✓ Pretraining complete")
        
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {num_params:,}")
            
            # Setup training
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, 
                                          weight_decay=1e-4, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=False, min_lr=1e-6
            )
            criterion = BPRLoss(lambda_reg=1e-4)
            
            # Train
            best_val_loss, history = train_ablation_model(
                model=model, train_loader=train_loader, val_loader=val_loader,
                optimizer=optimizer, criterion=criterion, device=device,
                scheduler=scheduler, num_epochs=num_epochs, patience=patience,
                model_name=config_name, verbose=True
            )
            
            training_time = time.time() - start_time
            
            # Save model
            model_path = output_path / f"{config_name}_model.pt"
            torch.save({
                'config_name': config_name,
                'config': config,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'num_params': num_params,
                'history': history,
                'training_time': training_time
            }, model_path)
            
            # Store results
            results[config_name] = {
                'config': config,
                'best_val_loss': float(best_val_loss),
                'num_params': num_params,
                'epochs_trained': history['epochs_trained'],
                'training_time': training_time,
                'final_train_loss': float(history['train_loss'][-1]) if history['train_loss'] else None,
                'history': {
                    'train_loss': [float(x) for x in history['train_loss']],
                    'val_loss': [float(x) for x in history['val_loss']]
                }
            }
            
            print(f"✓ Done in {training_time:.1f}s | Val loss: {best_val_loss:.4f} | Epochs: {history['epochs_trained']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Aggressive cleanup
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save summary
    print("\n" + "="*70)
    print("ABLATION COMPLETE")
    print("="*70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_val_loss'])
    
    print("\nRESULTS (sorted by val loss):")
    print("-" * 70)
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"{rank}. {name:<25} Val={res['best_val_loss']:.4f} | "
              f"Epochs={res['epochs_trained']} | Time={res['training_time']:.1f}s")
    
    summary_path = output_path / "ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'results': results,
            'sorted_rankings': [(name, res['best_val_loss']) for name, res in sorted_results]
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {summary_path}")
    
    return results, sorted_results


# ============================================================
# UTILITIES
# ============================================================

def load_enriched_chunks(chunks_dir: str) -> List[Dict]:
    chunks_path = Path(chunks_dir)
    chunk_files = sorted(chunks_path.glob('enriched_poi_paths_chunk_*.pkl'))
    if len(chunk_files) == 0:
        chunk_files = sorted(chunks_path.glob('*chunk*.pkl'))
    
    all_paths = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        with open(chunk_file, 'rb') as f:
            all_paths.extend(pickle.load(f))
    
    print(f"Loaded {len(all_paths):,} paths")
    return all_paths


def split_paths_by_user_poi(all_paths: List[Dict], val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    user_poi_paths = defaultdict(list)
    for path in all_paths:
        user_id = path.get('user_global_idx', path.get('user_idx', 'unknown'))
        poi_id = path.get('poi_global_idx', path.get('poi_idx', 'unknown'))
        user_poi_paths[(user_id, poi_id)].append(path)
    
    all_pairs = list(user_poi_paths.keys())
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * val_ratio)
    val_pairs = set(all_pairs[:val_size])
    
    train_paths, val_paths = [], []
    for pair, paths in user_poi_paths.items():
        if pair in val_pairs:
            val_paths.extend(paths)
        else:
            train_paths.extend(paths)
    
    print(f"Train: {len(train_paths):,} | Val: {len(val_paths):,}")
    return train_paths, val_paths


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ENRICHED_CHUNKS_DIR = "/kaggle/working/results"
    OUTPUT_DIR = "/kaggle/working/ablation_results"
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    PATIENCE = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    all_paths = load_enriched_chunks(ENRICHED_CHUNKS_DIR)
    
    # Extract labeled paths
    labeled_paths = [p for p in all_paths 
                     if p.get('embeddings') and p.get('has_actual_rating', False)]
    print(f"Labeled paths: {len(labeled_paths):,}")
    
    # Split
    train_paths, val_paths = split_paths_by_user_poi(labeled_paths, val_ratio=0.2, seed=42)
    
    # Run study
    results, sorted_results = run_ablation_study(
        all_paths=all_paths,
        train_paths=train_paths,
        val_paths=val_paths,
        device=device,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE
    )
    
    print("\n✓ Complete!")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ============================================================================
# FIXED ABLATION EVALUATION
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import json


# ============================================================
# CORRECT MODEL ARCHITECTURE (FROM ABLATION STUDY)
# ============================================================

class PathLSTM(nn.Module):
    """PathLSTM from ablation study - matches saved checkpoints."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.4, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward_single(self, path_embeddings):
        seq_len, embed_dim = path_embeddings.size(0), path_embeddings.size(1)
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                padding = torch.zeros(seq_len, self.embedding_dim - embed_dim, 
                                     device=path_embeddings.device, dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, padding], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        
        x = path_embeddings.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out.squeeze(0))
        lstm_out = self.dropout(lstm_out)
        
        if seq_len == 1:
            return lstm_out.squeeze(0), torch.ones(1, device=path_embeddings.device)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=0)
        path_repr = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=0)
        return path_repr, attention_weights
    
    def forward_batch(self, path_embeddings_batch, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            path_embeddings_batch, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        attention_logits = self.attention_score(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_logits = attention_logits.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        path_reprs = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return path_reprs, attention_weights
    
    def forward(self, path_embeddings, lengths=None):
        return self.forward_batch(path_embeddings, lengths) if lengths is not None else self.forward_single(path_embeddings)


class AblationPathRecommendationModel(nn.Module):
    """Model from ablation study - matches saved checkpoints."""
    
    def __init__(self, embedding_dim=64, lstm_hidden_dim=128, attention_dim=64,
                 mlp_dims=[128, 64], dropout=0.4, use_lstm=True,
                 use_attention_aggregation=True, use_mlp=True, lstm_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.attention_dim = attention_dim
        self.use_lstm = use_lstm
        self.use_attention_aggregation = use_attention_aggregation
        self.use_mlp = use_mlp
        
        if use_lstm:
            self.path_encoder = PathLSTM(embedding_dim, lstm_hidden_dim, dropout, lstm_layers)
            encoder_output_dim = lstm_hidden_dim
        else:
            self.path_encoder = None
            encoder_output_dim = embedding_dim
        
        if use_attention_aggregation:
            self.query_proj = nn.Linear(encoder_output_dim, attention_dim)
            self.key_proj = nn.Linear(encoder_output_dim, attention_dim)
            self.value_proj = nn.Linear(encoder_output_dim, attention_dim)
            aggregation_output_dim = attention_dim
        else:
            aggregation_output_dim = encoder_output_dim
        
        if use_mlp and len(mlp_dims) > 0:
            mlp_layers = []
            input_dim = aggregation_output_dim
            for hidden_dim in mlp_dims:
                mlp_layers.extend([
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                    nn.ReLU(), nn.Dropout(dropout)
                ])
                input_dim = hidden_dim
            mlp_layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*mlp_layers)
        else:
            self.mlp = nn.Linear(aggregation_output_dim, 1)
    
    def encode_paths_batch(self, all_paths):
        device = next(self.parameters()).device
        flat_paths, path_map = [], []
        
        for sample_idx, paths in enumerate(all_paths):
            for path_idx, path_emb in enumerate(paths):
                if isinstance(path_emb, np.ndarray) and len(path_emb) > 0:
                    flat_paths.append(path_emb)
                    path_map.append((sample_idx, path_idx))
        
        if len(flat_paths) == 0:
            return [[] for _ in all_paths]
        
        if self.use_lstm:
            length_groups = {}
            for idx, path in enumerate(flat_paths):
                length = len(path)
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append((idx, path))
            
            path_representations = [None] * len(flat_paths)
            for length, group in length_groups.items():
                indices = [item[0] for item in group]
                paths = [item[1] for item in group]
                batch_tensor = torch.from_numpy(np.stack(paths)).float().to(device)
                lengths_tensor = torch.full((len(paths),), length, dtype=torch.long, device=device)
                batch_reprs, _ = self.path_encoder(batch_tensor, lengths_tensor)
                for i, repr_tensor in enumerate(batch_reprs):
                    path_representations[indices[i]] = repr_tensor
        else:
            path_representations = []
            for path in flat_paths:
                path_tensor = torch.from_numpy(path).float().to(device)
                if path_tensor.size(1) != self.embedding_dim:
                    if path_tensor.size(1) < self.embedding_dim:
                        padding = torch.zeros(path_tensor.size(0), 
                                            self.embedding_dim - path_tensor.size(1),
                                            device=device, dtype=path_tensor.dtype)
                        path_tensor = torch.cat([path_tensor, padding], dim=1)
                    else:
                        path_tensor = path_tensor[:, :self.embedding_dim]
                path_repr = path_tensor.mean(dim=0)
                path_representations.append(path_repr)
        
        result = [[] for _ in all_paths]
        for flat_idx, (sample_idx, path_idx) in enumerate(path_map):
            if path_representations[flat_idx] is not None:
                result[sample_idx].append(path_representations[flat_idx])
        return result
    
    def aggregate_path_sets_batch(self, all_path_reprs):
        device = next(self.parameters()).device
        batch_size = len(all_path_reprs)
        max_paths = max(len(reprs) for reprs in all_path_reprs if len(reprs) > 0)
        
        if max_paths == 0:
            output_dim = self.attention_dim if self.use_attention_aggregation else (
                self.lstm_hidden_dim if self.use_lstm else self.embedding_dim
            )
            return torch.zeros(batch_size, output_dim, device=device), [0] * batch_size
        
        repr_dim = all_path_reprs[0][0].size(0) if len(all_path_reprs[0]) > 0 else (
            self.lstm_hidden_dim if self.use_lstm else self.embedding_dim
        )
        padded_reprs = torch.zeros(batch_size, max_paths, repr_dim, device=device)
        path_counts = []
        
        for i, reprs in enumerate(all_path_reprs):
            if len(reprs) > 0:
                stacked = torch.stack(reprs)
                padded_reprs[i, :len(reprs)] = stacked
                path_counts.append(len(reprs))
            else:
                path_counts.append(1)
        
        path_counts = torch.tensor(path_counts, device=device)
        
        if self.use_attention_aggregation:
            Q = self.query_proj(padded_reprs)
            K = self.key_proj(padded_reprs)
            V = self.value_proj(padded_reprs)
            attention_scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.attention_dim)
            mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, max_paths, -1)
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
            attention_weights = F.softmax(attention_scores, dim=2)
            L_prime = torch.bmm(attention_weights, V)
            L_mp = torch.max(L_prime, dim=1)[0]
            best_indices = torch.argmax(attention_weights.sum(dim=1), dim=1).tolist()
        else:
            mask = torch.arange(max_paths, device=device).unsqueeze(0) < path_counts.unsqueeze(1)
            masked_reprs = padded_reprs * mask.unsqueeze(-1).float()
            L_mp = masked_reprs.sum(dim=1) / path_counts.unsqueeze(-1).float()
            best_indices = [0] * batch_size
        
        return L_mp, best_indices
    
    def forward(self, paths_batch):
        all_path_reprs = self.encode_paths_batch(paths_batch)
        L_mp_batch, best_indices = self.aggregate_path_sets_batch(all_path_reprs)
        scores = self.mlp(L_mp_batch).squeeze(-1)
        predictions = torch.sigmoid(scores)
        return predictions, best_indices


# ============================================================
# EVALUATION METRICS
# ============================================================

class RecommendationMetrics:
    """Compute recommendation metrics at different K values."""
    
    @staticmethod
    def hit_rate(recommended: List, ground_truth: Set, k: int) -> float:
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & ground_truth) > 0 else 0.0
    
    @staticmethod
    def precision(recommended: List, ground_truth: Set, k: int) -> float:
        top_k = set(recommended[:k])
        if len(top_k) == 0:
            return 0.0
        return len(top_k & ground_truth) / k
    
    @staticmethod
    def recall(recommended: List, ground_truth: Set, k: int) -> float:
        top_k = set(recommended[:k])
        if len(ground_truth) == 0:
            return 0.0
        return len(top_k & ground_truth) / len(ground_truth)
    
    @staticmethod
    def f1_score(recommended: List, ground_truth: Set, k: int) -> float:
        prec = RecommendationMetrics.precision(recommended, ground_truth, k)
        rec = RecommendationMetrics.recall(recommended, ground_truth, k)
        
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def ndcg(recommended: List, ground_truth: Set, k: int) -> float:
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mrr(recommended: List, ground_truth: Set) -> float:
        for i, item in enumerate(recommended):
            if item in ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def map_at_k(recommended: List, ground_truth: Set, k: int) -> float:
        if len(ground_truth) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended[:k]):
            if item in ground_truth:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(ground_truth), k)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_evaluation_data(enriched_paths: List[Dict]) -> Tuple[Dict, Dict, Set]:
    """Organize ALL paths by user-item pairs."""
    print("\n" + "="*60)
    print("Preparing evaluation data")
    print("="*60)
    
    user_item_paths = defaultdict(lambda: defaultdict(list))
    user_positive_items = defaultdict(set)
    pois_with_any_rating = set()
    
    stats = {
        'total_paths': 0,
        'valid_paths': 0,
        'with_ratings': 0,
        'positive_ratings': 0
    }
    
    for path in enriched_paths:
        stats['total_paths'] += 1
        
        if not path.get('embeddings') or len(path['embeddings']) == 0:
            continue
        
        stats['valid_paths'] += 1
        
        user_id = (path.get('user_idx') or path.get('user_global_idx') or 
                   path.get('user_id') or path.get('user'))
        item_id = (path.get('poi_idx') or path.get('poi_global_idx') or 
                   path.get('poi_id') or path.get('item_id'))
        
        rating = path.get('rating')
        has_actual = path.get('has_actual_rating', False)
        
        if user_id is None or item_id is None:
            continue
        
        user_item_paths[user_id][item_id].append(path)
        
        if has_actual and rating is not None:
            stats['with_ratings'] += 1
            pois_with_any_rating.add(item_id)
            if rating >= 3.0:
                stats['positive_ratings'] += 1
                user_positive_items[user_id].add(item_id)
    
    print(f"\nData Statistics:")
    print(f"  Total paths: {stats['total_paths']:,}")
    print(f"  Valid paths: {stats['valid_paths']:,}")
    print(f"  Paths with ratings: {stats['with_ratings']:,}")
    print(f"  Positive interactions: {stats['positive_ratings']:,}")
    print(f"  Unique users: {len(user_item_paths):,}")
    print(f"  Unique POIs with ratings: {len(pois_with_any_rating):,}")
    
    return dict(user_item_paths), dict(user_positive_items), pois_with_any_rating


# ============================================================
# RANKING GENERATION
# ============================================================

def generate_rankings_for_user(model: nn.Module,
                               user_id: int,
                               user_item_paths: Dict,
                               device: torch.device,
                               batch_size: int = 64) -> List[Tuple[int, float]]:
    """Rank ALL items this user has paths to."""
    model.eval()
    
    all_items = list(user_item_paths[user_id].keys())
    
    if len(all_items) == 0:
        return []
    
    rankings = []
    
    for i in range(0, len(all_items), batch_size):
        batch_items = all_items[i:i+batch_size]
        
        paths_batch = []
        valid_items = []
        
        for item_id in batch_items:
            paths = user_item_paths[user_id][item_id]
            
            path_embeddings = []
            for p in paths:
                emb = p.get('embeddings')
                if emb is not None and len(emb) > 0:
                    if isinstance(emb, list):
                        emb = np.array(emb)
                    path_embeddings.append(emb)
            
            if len(path_embeddings) > 0:
                paths_batch.append(path_embeddings)
                valid_items.append(item_id)
        
        if len(paths_batch) == 0:
            continue
        
        try:
            with torch.no_grad():
                scores, _ = model(paths_batch)
                scores = scores.cpu().numpy()
            
            for item_id, score in zip(valid_items, scores):
                rankings.append((item_id, float(score)))
        except Exception as e:
            continue
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings


# ============================================================
# EVALUATION PIPELINE
# ============================================================

def evaluate_ranking_model(model: nn.Module,
                          user_item_paths: Dict,
                          user_positive_items: Dict,
                          device: torch.device,
                          k_values: List[int] = [5, 10, 20, 50]) -> Dict:
    """Evaluate ranking quality."""
    print("\n" + "="*60)
    print("Evaluating ranking performance...")
    print("="*60)
    
    model.to(device)
    model.eval()
    
    metrics_collector = defaultdict(list)
    
    eval_users = [u for u in user_positive_items.keys() if len(user_positive_items[u]) > 0]
    
    print(f"Evaluating on {len(eval_users)} users with positive ratings\n")
    
    users_evaluated = 0
    users_skipped = 0
    
    for user_id in tqdm(eval_users, desc="Evaluating users"):
        positive_items = user_positive_items[user_id]
        
        if user_id not in user_item_paths:
            users_skipped += 1
            continue
        
        rankings = generate_rankings_for_user(
            model, user_id, user_item_paths, device
        )
        
        if len(rankings) == 0:
            users_skipped += 1
            continue
        
        ranked_items = [item for item, _ in rankings]
        num_positives = len(positive_items)
        num_total = len(ranked_items)
        
        if num_positives == num_total or num_positives == 0:
            users_skipped += 1
            continue
        
        users_evaluated += 1
        
        for k in k_values:
            metrics_collector[f'HR@{k}'].append(
                RecommendationMetrics.hit_rate(ranked_items, positive_items, k)
            )
            metrics_collector[f'Precision@{k}'].append(
                RecommendationMetrics.precision(ranked_items, positive_items, k)
            )
            metrics_collector[f'Recall@{k}'].append(
                RecommendationMetrics.recall(ranked_items, positive_items, k)
            )
            metrics_collector[f'F1@{k}'].append(
                RecommendationMetrics.f1_score(ranked_items, positive_items, k)
            )
            metrics_collector[f'NDCG@{k}'].append(
                RecommendationMetrics.ndcg(ranked_items, positive_items, k)
            )
            metrics_collector[f'MAP@{k}'].append(
                RecommendationMetrics.map_at_k(ranked_items, positive_items, k)
            )
        
        metrics_collector['MRR'].append(
            RecommendationMetrics.mrr(ranked_items, positive_items)
        )
    
    print(f"\nEvaluation complete:")
    print(f"  Users evaluated: {users_evaluated}")
    print(f"  Users skipped: {users_skipped}")
    
    if users_evaluated == 0:
        return {}
    
    results = {}
    for metric_name, values in metrics_collector.items():
        results[metric_name] = np.mean(values) if len(values) > 0 else 0.0
    
    return results


# ============================================================
# MAIN - ABLATION EVALUATION
# ============================================================

def main():
    # Configuration
    MODEL_PATHS = [
        "/kaggle/working/ablation_results/avg_aggregation_model.pt",
        "/kaggle/working/ablation_results/avg_no_mlp_model.pt",
        "/kaggle/working/ablation_results/full_model_model.pt",
        "/kaggle/working/ablation_results/minimal_model_model.pt",
        "/kaggle/working/ablation_results/no_mlp_model.pt",
        "/kaggle/working/ablation_results/no_pretrain_avg_model.pt",
        "/kaggle/working/ablation_results/no_pretrain_no_mlp_model.pt",
        "/kaggle/working/ablation_results/no_pretraining_model.pt"
    ]
    
    ENRICHED_CHUNKS_DIR = "/kaggle/working/results"
    K_VALUES = [5, 10, 20, 50]
    BATCH_SIZE = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data once
    print("\n" + "="*60)
    print("Loading enriched paths...")
    print("="*60)
    
    chunks_path = Path(ENRICHED_CHUNKS_DIR)
    chunk_files = sorted(chunks_path.glob('enriched_poi_paths_chunk_*.pkl'))
    
    if len(chunk_files) == 0:
        chunk_files = sorted(chunks_path.glob('*chunk*.pkl'))
    
    all_paths = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            all_paths.extend(chunk_data)
    
    print(f"Total paths loaded: {len(all_paths):,}")
    
    # Prepare data
    user_item_paths, user_positive_items, pois_with_any_rating = prepare_evaluation_data(all_paths)
    
    # Prune items without ratings
    print("\n" + "="*60)
    print("Pruning items without ratings...")
    print("="*60)
    
    pruned_user_item_paths = {}
    for user_id, items in user_item_paths.items():
        filtered = {item_id: paths for item_id, paths in items.items() if item_id in pois_with_any_rating}
        pruned_user_item_paths[user_id] = filtered
    
    print(f"Unique POIs with ratings: {len(pois_with_any_rating):,}")
    
    # Evaluate all models
    all_results = []
    
    print("\n" + "="*70)
    print("EVALUATING ALL ABLATION MODELS")
    print("="*70)
    
    for model_path in MODEL_PATHS:
        model_name = Path(model_path).stem
        
        print("\n" + "#"*70)
        print(f"MODEL: {model_name}")
        print("#"*70)

        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Extract config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("✓ Loaded config from checkpoint")
            else:
                # Default ablation config
                config = {
                    'embedding_dim': 64,
                    'lstm_hidden_dim': 128,
                    'attention_dim': 64,
                    'mlp_dims': [128, 64],
                    'dropout': 0.4,
                    'use_lstm': True,
                    'use_attention_aggregation': True,
                    'use_mlp': True,
                    'lstm_layers': 2
                }
                print("⚠ Using default config")
            
            print("Configuration:")
            for key, value in config.items():
                if not key.startswith('use_') and key != 'description':
                    print(f"  {key}: {value}")
            
            # Create model with CORRECT architecture
            model = AblationPathRecommendationModel(
                embedding_dim=config.get('embedding_dim', 64),
                lstm_hidden_dim=config.get('lstm_hidden_dim', 128),
                attention_dim=config.get('attention_dim', 64),
                mlp_dims=config.get('mlp_dims', [128, 64]),
                dropout=config.get('dropout', 0.4),
                use_lstm=config.get('use_lstm', True),
                use_attention_aggregation=config.get('use_attention_aggregation', True),
                use_mlp=config.get('use_mlp', True),
                lstm_layers=config.get('lstm_layers', 2)
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded successfully")
            
            # Evaluate
            results = evaluate_ranking_model(
                model=model,
                user_item_paths=pruned_user_item_paths,
                user_positive_items=user_positive_items,
                device=device,
                k_values=K_VALUES
            )
            
            if results:
                results['model_name'] = model_name
                results['model_path'] = model_path
                
                # Add config info
                results['use_contrastive_pretrain'] = config.get('use_contrastive_pretrain', 'Unknown')
                results['use_attention_aggregation'] = config.get('use_attention_aggregation', 'Unknown')
                results['use_mlp'] = config.get('use_mlp', 'Unknown')
                results['description'] = config.get('description', 'N/A')
                
                all_results.append(results)
                
                print(f"\n✓ Evaluation complete for {model_name}")
            else:
                print(f"\n⚠️  No results for {model_name}")
            
            # Clean up
            del model, checkpoint
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    if len(all_results) == 0:
        print("\n❌ No models were successfully evaluated!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns
    metric_cols = [col for col in results_df.columns if '@' in col or col == 'MRR']
    info_cols = ['model_name', 'description', 'use_contrastive_pretrain', 'use_attention_aggregation', 'use_mlp']
    info_cols = [col for col in info_cols if col in results_df.columns]
    other_cols = [col for col in results_df.columns if col not in metric_cols and col not in info_cols]
    
    results_df = results_df[info_cols + metric_cols + other_cols]
    
    # Display results
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    
    # Print compact summary
    print("\n" + "-"*120)
    print(f"{'Model':<30} {'Pretrain':<10} {'Attention':<12} {'MLP':<8} {'HR@10':<8} {'NDCG@10':<10} {'MRR':<8}")
    print("-"*120)
    
    for _, row in results_df.iterrows():
        model_name = row['model_name']
        pretrain = str(row.get('use_contrastive_pretrain', 'N/A'))[:8]
        attention = str(row.get('use_attention_aggregation', 'N/A'))[:10]
        mlp = str(row.get('use_mlp', 'N/A'))[:6]
        hr10 = f"{row.get('HR@10', 0):.4f}"
        ndcg10 = f"{row.get('NDCG@10', 0):.4f}"
        mrr = f"{row.get('MRR', 0):.4f}"
        
        print(f"{model_name:<30} {pretrain:<10} {attention:<12} {mlp:<8} {hr10:<8} {ndcg10:<10} {mrr:<8}")
    
    print("-"*120)
    
    # Print detailed metrics for each model
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['model_name']}")
        print("-" * 70)
        if 'description' in row:
            print(f"Description: {row['description']}")
        print(f"Configuration:")
        print(f"  Pretraining: {row.get('use_contrastive_pretrain', 'N/A')}")
        print(f"  Attention Aggregation: {row.get('use_attention_aggregation', 'N/A')}")
        print(f"  MLP: {row.get('use_mlp', 'N/A')}")
        print(f"\nMetrics:")
        for k in K_VALUES:
            print(f"  K={k}:")
            print(f"    HR@{k}: {row.get(f'HR@{k}', 0):.4f}")
            print(f"    Precision@{k}: {row.get(f'Precision@{k}', 0):.4f}")
            print(f"    Recall@{k}: {row.get(f'Recall@{k}', 0):.4f}")
            print(f"    F1@{k}: {row.get(f'F1@{k}', 0):.4f}")
            print(f"    NDCG@{k}: {row.get(f'NDCG@{k}', 0):.4f}")
            print(f"    MAP@{k}: {row.get(f'MAP@{k}', 0):.4f}")
        print(f"  MRR: {row.get('MRR', 0):.4f}")
    
    # Save results to CSV
    output_file = 'ablation_evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Save detailed JSON report
    json_output = {
        'summary': {
            'total_models_evaluated': len(all_results),
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'k_values': K_VALUES,
            'device': str(device)
        },
        'results': all_results
    }
    
    json_file = 'ablation_evaluation_results.json'
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"✓ Detailed results saved to: {json_file}")
    
    # Find best model for each metric
    print("\n" + "="*70)
    print("BEST MODELS BY METRIC")
    print("="*70)
    
    for metric in metric_cols:
        if metric in results_df.columns:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'model_name']
            best_score = results_df.loc[best_idx, metric]
            print(f"{metric:<20} {best_model:<30} {best_score:.4f}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Total models evaluated: {len(all_results)}")
    print(f"Results saved to: {output_file}")
    print(f"JSON report saved to: {json_file}")


if __name__ == "__main__":
    main()