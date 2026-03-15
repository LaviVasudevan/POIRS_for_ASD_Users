"""
POI Recommendation Flask App
Pipeline : GAT → TransR → PathLSTM (AblationPathRecommendationModel)
"""

import os, csv, json, pickle
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
    print(f"[OpenRouter] openai package imported OK, _OpenAI={_OpenAI}")
except Exception as _import_err:
    print(f"[OpenRouter] Failed to import openai: {_import_err}")
    _OpenAI = None
    _OPENAI_AVAILABLE = False

def _get_openrouter_client():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    print(f"[OpenRouter] key present={bool(key)}, _OPENAI_AVAILABLE={_OPENAI_AVAILABLE}, _OpenAI={_OpenAI}")
    if not key:
        print("[OpenRouter] Key is empty")
        return None, key
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )
        print(f"[OpenRouter] Client created OK: {client}")
        return client, key
    except Exception as e:
        print(f"[OpenRouter] Failed to create client: {e}")
        return None, key

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE          = Path(os.environ.get("BASE_DIR",       "data"))
GRAPH_PATH    = Path(os.environ.get("GRAPH_PATH",     BASE / "poi_graph_v6.pt"))
GAT_EMB_PATH  = Path(os.environ.get("GAT_EMB_PATH",  BASE / "gat_embeddings.pt"))
TRANSR_PATH   = Path(os.environ.get("TRANSR_PATH",   BASE / "transr_final.pt"))
MODEL_PATH    = Path(os.environ.get("MODEL_PATH",    BASE / "model_checkpoint.pt"))
ENRICHED_DIR  = Path(os.environ.get("ENRICHED_DIR",  BASE / "enriched_paths"))
POIS_CSV      = Path(os.environ.get("POIS_CSV",      BASE / "POIRS.pois.csv"))
PROFILES_JSON = Path(os.environ.get("PROFILES_JSON", BASE / "POIRS.questionnaires.json"))
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (mirrors notebook Cell 25 exactly)
# ─────────────────────────────────────────────────────────────────────────────
class PathLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.4, num_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim    = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))

    def forward_single(self, path_embeddings):
        seq_len, embed_dim = path_embeddings.size(0), path_embeddings.size(1)
        if embed_dim != self.embedding_dim:
            if embed_dim < self.embedding_dim:
                pad = torch.zeros(seq_len, self.embedding_dim - embed_dim,
                                  device=path_embeddings.device, dtype=path_embeddings.dtype)
                path_embeddings = torch.cat([path_embeddings, pad], dim=1)
            else:
                path_embeddings = path_embeddings[:, :self.embedding_dim]
        lstm_out, _ = self.lstm(path_embeddings.unsqueeze(0))
        lstm_out = self.layer_norm(lstm_out.squeeze(0))
        lstm_out = self.dropout(lstm_out)
        if lstm_out.size(0) == 1:
            return lstm_out.squeeze(0), torch.ones(1, device=path_embeddings.device)
        attn = F.softmax(self.attention_score(lstm_out).squeeze(-1), dim=0)
        return (lstm_out * attn.unsqueeze(-1)).sum(0), attn

    def forward(self, path_embeddings, lengths=None):
        return self.forward_single(path_embeddings)


class AblationPathRecommendationModel(nn.Module):
    def __init__(self, embedding_dim=64, lstm_hidden_dim=128, attention_dim=64,
                 mlp_dims=None, dropout=0.4, use_lstm=True,
                 use_attention_aggregation=True, use_mlp=True, lstm_layers=2):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64]
        self.use_lstm = use_lstm
        self.use_attention_aggregation = use_attention_aggregation
        self.use_mlp = use_mlp

        if use_lstm:
            self.path_encoder = PathLSTM(embedding_dim, lstm_hidden_dim, dropout, lstm_layers)
            enc_dim = lstm_hidden_dim
        else:
            self.path_encoder = None
            enc_dim = embedding_dim

        if use_attention_aggregation:
            self.query_proj = nn.Linear(enc_dim, attention_dim)
            self.key_proj   = nn.Linear(enc_dim, attention_dim)
            self.value_proj = nn.Linear(enc_dim, attention_dim)
            agg_dim = attention_dim
        else:
            agg_dim = enc_dim

        if use_mlp and mlp_dims:
            layers, d = [], agg_dim
            for o in mlp_dims:
                layers += [nn.Linear(d, o), nn.LayerNorm(o), nn.ReLU(), nn.Dropout(dropout)]
                d = o
            layers.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Linear(agg_dim, 1)

    def score_candidate(self, path_list: list) -> float:
        if not path_list:
            return 0.0
        reprs = []
        for emb in path_list:
            t = torch.tensor(emb, dtype=torch.float32).to(DEVICE)
            r, _ = self.path_encoder.forward_single(t) if self.path_encoder else (t.mean(0), None)
            reprs.append(r)
        stack = torch.stack(reprs)
        if self.use_attention_aggregation:
            q = self.query_proj(stack)
            k = self.key_proj(stack)
            v = self.value_proj(stack)
            w = F.softmax((q * k).sum(-1) / (q.size(-1) ** .5), dim=0)
            agg = (v * w.unsqueeze(-1)).sum(0)
        else:
            agg = stack.mean(0)
        return self.mlp(agg.unsqueeze(0)).squeeze().item()


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class Engine:
    def __init__(self):
        self.graph = None; self.metadata: Dict = {}
        self.user_to_idx: Dict[str,int]={};  self.idx_to_user: Dict[int,str]={}
        self.poi_to_idx:  Dict[str,int]={};  self.idx_to_poi:  Dict[int,str]={}
        self.category_to_idx:Dict[str,int]={}; self.idx_to_category:Dict[int,str]={}
        self.sensory_to_idx: Dict[str,int]={};  self.idx_to_sensory: Dict[int,str]={}
        self.other_to_idx:   Dict[str,int]={};  self.idx_to_other:   Dict[int,str]={}
        self.poi_names: Dict[str,str]={}
        self.profiles:  Dict={}
        self.gat_embeddings: Optional[Dict]=None
        self.model: Optional[AblationPathRecommendationModel]=None
        self.user_poi_paths: Dict[int,Dict[int,List]]={}
        self.num_users=0; self.num_pois=0

    def load_all(self):
        self._load_graph(); self._load_poi_names(); self._load_profiles()
        self._load_gat(); self._load_enriched_paths(); self._generate_paths_from_graph(); self._load_model()
        print(f"[Engine] Ready  u={self.num_users} p={self.num_pois} "
              f"prof={len([k for k in self.profiles if isinstance(k,str)])} names={len(self.poi_names)}")

    def _load_graph(self):
        if not GRAPH_PATH.exists(): print(f"[Engine] graph not found: {GRAPH_PATH}"); return
        raw = torch.load(GRAPH_PATH, weights_only=False, map_location="cpu")
        self.graph    = raw.get("graph", raw) if isinstance(raw, dict) else raw
        self.metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
        for mk, attr in [("user_to_idx","user"),("poi_to_idx","poi"),
                          ("category_to_idx","category"),("sensory_attr_to_idx","sensory"),
                          ("other_attr_to_idx","other")]:
            fwd = self.metadata.get(mk, {})
            setattr(self, f"{attr}_to_idx", fwd)
            setattr(self, f"idx_to_{attr}", {v:k for k,v in fwd.items()})
        if self.graph:
            self.num_users = int(self.graph["user"].num_nodes)
            self.num_pois  = int(self.graph["poi"].num_nodes)

    def _load_poi_names(self):
        if not POIS_CSV.exists(): return
        with open(POIS_CSV, encoding="utf-8") as f:
            reader = csv.reader(f); next(reader, None)
            for row in reader:
                if len(row) >= 3: self.poi_names[row[1].strip()] = row[2].strip()

    def _load_profiles(self):
        if not PROFILES_JSON.exists(): return
        with open(PROFILES_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for p in data:
            uid = p.get("userId");
            if uid is None: continue
            self.profiles[uid] = p
            idx = self.user_to_idx.get(uid)
            if idx is not None: self.profiles[idx] = p
            if isinstance(uid, str) and uid.startswith("user_"):
                try: self.profiles.setdefault(int(uid.split("_",1)[1]), p)
                except: pass

    def _load_gat(self):
        if GAT_EMB_PATH.exists():
            self.gat_embeddings = torch.load(GAT_EMB_PATH, weights_only=False, map_location="cpu")

    def _load_entity_mapping(self):
        """
        Load poi_mappings.pkl.
        entity2id keys are tuples: ('user', local_idx), ('poi', local_idx), etc.
        Returns id2entity: {int -> (node_type, local_idx)}
        """
        candidates = []
        if ENRICHED_DIR.exists():
            candidates += sorted(ENRICHED_DIR.glob("*mapping*.pkl"))
            candidates += sorted(ENRICHED_DIR.glob("*mappings*.pkl"))
        if BASE.exists():
            candidates += sorted(BASE.glob("*mapping*.pkl"))
            candidates += sorted(BASE.glob("*mappings*.pkl"))

        for path in candidates:
            try:
                with open(path, "rb") as fh:
                    m = pickle.load(fh)
                if isinstance(m, dict) and "entity2id" in m:
                    id2entity = {v: k for k, v in m["entity2id"].items()}
                    print(f"[Engine] Entity mapping loaded from {path.name}: "
                          f"{len(id2entity)} entities, {len(m.get('relation2id', {}))} relations")
                    return id2entity
            except Exception as exc:
                print(f"[Engine] Could not load mapping {path.name}: {exc}")

        print("[Engine] WARNING: No entity mapping file found")
        return {}

    def _path_to_embedding(self, entity_ids: list, id2entity: dict) -> Optional[np.ndarray]:
        """
        Convert a list of entity IDs into a (path_len, embed_dim) float32 array
        by looking up each entity in id2entity -> (node_type, local_idx)
        then fetching that row from gat_embeddings[node_type].
        """
        if self.gat_embeddings is None:
            return None

        vectors = []
        for eid in entity_ids:
            entity = id2entity.get(int(eid))
            if entity is None:
                return None

            node_type, local_idx = entity  # e.g. ('user', 184) or ('poi', 6)

            emb_tensor = self.gat_embeddings.get(node_type)
            if emb_tensor is None or local_idx >= emb_tensor.size(0):
                return None

            vectors.append(emb_tensor[local_idx].cpu().numpy().astype(np.float32))

        if not vectors:
            return None

        return np.stack(vectors)  # shape: (path_len, embed_dim)

    def _load_enriched_paths(self):
        if not ENRICHED_DIR.exists():
            print(f"[Engine] ENRICHED_DIR not found: {ENRICHED_DIR}")
            return

        # Exclude mapping files from chunk list
        all_pkl = sorted(ENRICHED_DIR.glob("*.pkl"))
        files   = [f for f in all_pkl
                   if not any(x in f.name.lower() for x in ("mapping", "mappings"))]
        if not files:
            files = all_pkl

        if not files:
            print(f"[Engine] No .pkl files found in {ENRICHED_DIR}")
            return

        id2entity = self._load_entity_mapping()
        if not id2entity:
            print("[Engine] Cannot load enriched paths without entity mapping")
            return

        print(f"[Engine] Loading {len(files)} enriched-path chunk(s) from {ENRICHED_DIR}")
        total_ok = total_skip = total_err = 0

        for f in files:
            try:
                with open(f, "rb") as fh:
                    chunk = pickle.load(fh)
            except Exception as exc:
                print(f"[Engine]   ERROR reading {f.name}: {exc}")
                continue

            if not isinstance(chunk, dict):
                print(f"[Engine]   Unexpected chunk type {type(chunk)} in {f.name}, skipping")
                continue

            paths = chunk.get("paths", [])
            if not paths:
                print(f"[Engine]   {f.name}: empty paths, skipping")
                continue

            file_ok = file_skip = file_err = 0

            for path_tuple in paths:
                try:
                    # path_tuple = ([entity_ids], [relation_ids])
                    # The path always starts at the user node and ends at the target poi.
                    # e.g. decoded: [('user', 184), ('poi', 6), ('sensory_attr', 12), ('poi', 118)]
                    # u = first entity, p = last entity (must be a poi)
                    if not isinstance(path_tuple, (tuple, list)) or len(path_tuple) < 1:
                        file_skip += 1; continue

                    entity_ids = path_tuple[0]
                    if not entity_ids or len(entity_ids) < 2:
                        file_skip += 1; continue

                    # Decode first entity -> user
                    first = id2entity.get(int(entity_ids[0]))
                    if first is None or first[0] != 'user':
                        file_skip += 1; continue
                    u = first[1]

                    # Decode last entity -> poi (target POI)
                    last = id2entity.get(int(entity_ids[-1]))
                    if last is None or last[0] != 'poi':
                        file_skip += 1; continue
                    p = last[1]

                    # Convert full path entity IDs -> GAT embedding matrix
                    arr = self._path_to_embedding(entity_ids, id2entity)
                    if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
                        file_skip += 1; continue

                    self.user_poi_paths.setdefault(u, {}).setdefault(p, []).append(arr)
                    file_ok += 1

                except Exception as exc:
                    file_err += 1
                    if file_err <= 3:
                        print(f"[Engine]   error in {f.name}: {exc}")

            total_ok   += file_ok
            total_skip += file_skip
            total_err  += file_err
            print(f"[Engine]   {f.name}: loaded={file_ok}  skipped={file_skip}  errors={file_err}")

        print(f"[Engine] Enriched paths total: loaded={total_ok}  skipped={total_skip}  errors={total_err}")
        print(f"[Engine] Users with paths: {len(self.user_poi_paths)}")

    def _generate_paths_from_graph(self, walk_length: int = 4, walks_per_pair: int = 5):
        if self.user_poi_paths:
            return

        if self.gat_embeddings is None:
            print("[Engine] Cannot generate paths: GAT embeddings not loaded")
            return
        if self.graph is None:
            print("[Engine] Cannot generate paths: graph not loaded")
            return

        print(f"[Engine] No enriched paths found — generating on-the-fly from graph "
              f"(walk_length={walk_length}, walks_per_pair={walks_per_pair}) …")

        gat = self.gat_embeddings
        ed  = self.graph.edge_index_dict

        def get_emb(node_type: str, idx: int):
            t = gat.get(node_type)
            if t is not None and idx < t.size(0):
                return t[idx].cpu().numpy().astype(np.float32)
            return None

        adj: Dict = defaultdict(lambda: defaultdict(list))
        for (st, _r, dt), ei in ed.items():
            src, dst = ei[0].tolist(), ei[1].tolist()
            for s, d in zip(src, dst):
                adj[(st, dt)][s].append(d)
            for s, d in zip(src, dst):
                adj[(dt, st)][d].append(s)

        generated_pairs = 0
        for u_idx in range(self.num_users):
            u_emb = get_emb("user", u_idx)
            if u_emb is None:
                continue

            poi_paths: Dict[int, List[np.ndarray]] = defaultdict(list)
            frontier = deque([("user", u_idx, [u_emb])])

            for _hop in range(walk_length):
                next_frontier: deque = deque()
                seen_states: set = set()

                while frontier:
                    cur_type, cur_idx, path_embs = frontier.popleft()

                    for (st, dt), nbr_map in adj.items():
                        if st != cur_type:
                            continue
                        for nb_idx in nbr_map.get(cur_idx, [])[:8]:
                            state_key = (dt, nb_idx)
                            if state_key in seen_states:
                                continue
                            nb_emb = get_emb(dt, nb_idx)
                            if nb_emb is None:
                                continue
                            new_path = path_embs + [nb_emb]

                            if dt == "poi":
                                if len(poi_paths[nb_idx]) < walks_per_pair:
                                    arr = np.stack(new_path).astype(np.float32)
                                    poi_paths[nb_idx].append(arr)
                                    seen_states.add(state_key)
                            else:
                                next_frontier.append((dt, nb_idx, new_path))
                                seen_states.add(state_key)

                frontier = next_frontier

            if poi_paths:
                self.user_poi_paths[u_idx] = dict(poi_paths)
                generated_pairs += len(poi_paths)

        print(f"[Engine] Generated {generated_pairs} user-POI path sets "
              f"for {len(self.user_poi_paths)} users")

    def _load_model(self):
        if not MODEL_PATH.exists(): return
        ckpt = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
        state, cfg = (ckpt["model_state"], ckpt.get("config",{})) if isinstance(ckpt,dict) and "model_state" in ckpt else (ckpt,{})
        m = AblationPathRecommendationModel(
            embedding_dim=cfg.get("embedding_dim",64), lstm_hidden_dim=cfg.get("lstm_hidden_dim",128),
            attention_dim=cfg.get("attention_dim",64), mlp_dims=cfg.get("mlp_dims",[128,64]),
            dropout=cfg.get("dropout",.4), use_lstm=cfg.get("use_lstm",True),
            use_attention_aggregation=cfg.get("use_attention_aggregation",True),
            use_mlp=cfg.get("use_mlp",True), lstm_layers=cfg.get("lstm_layers",2))
        m.load_state_dict(state, strict=False); m.to(DEVICE).eval(); self.model = m

    # ── helpers ──────────────────────────────────────────────────────────────
    def user_str_key(self,idx): return self.idx_to_user.get(idx, f"user_{idx}")
    def poi_str_key(self,idx):  return self.idx_to_poi.get(idx,  f"poi_{idx}")
    def poi_display_name(self,idx):
        key = self.poi_str_key(idx); return self.poi_names.get(key, key)
    def get_profile(self,user_idx):
        return self.profiles.get(self.user_str_key(user_idx)) or self.profiles.get(user_idx)
    def get_user_list(self):
        if self.idx_to_user: return [{"id":i,"name":k} for i,k in sorted(self.idx_to_user.items())]
        return [{"id":i,"name":f"user_{i}"} for i in range(self.num_users)]

    # ── scoring ───────────────────────────────────────────────────────────────
    def recommend(self, user_idx: int, top_k: int=10) -> Dict:
        scored, method = [], "none"
        if self.model and user_idx in self.user_poi_paths:
            method = "PathLSTM"
            with torch.no_grad():
                for poi_idx, pl in self.user_poi_paths[user_idx].items():
                    scored.append((poi_idx, float(self.model.score_candidate(pl))))
            scored.sort(key=lambda x: x[1], reverse=True)
        elif self.gat_embeddings and "user" in self.gat_embeddings and "poi" in self.gat_embeddings:
            method = "GAT-cosine"
            ue, pe = self.gat_embeddings["user"], self.gat_embeddings["poi"]
            if user_idx < ue.size(0):
                u = F.normalize(ue[user_idx].unsqueeze(0), dim=-1)
                p = F.normalize(pe, dim=-1)
                sims = (u @ p.T).squeeze(0)
                tv, ti = sims.topk(min(top_k, sims.size(0)))
                scored = [(int(ti[j]), float(tv[j])) for j in range(len(ti))]
        return {
            "results":[{"rank":r+1,"poi_id":poi,"poi_key":self.poi_str_key(poi),
                         "name":self.poi_display_name(poi),"score":round(s,4),"method":method}
                        for r,(poi,s) in enumerate(scored[:top_k])],
            "method": method
        }

    # ── graph evidence ────────────────────────────────────────────────────────
    def explain(self, user_idx: int, poi_idx: int) -> Dict:
        if self.graph is None: return {}
        ed = self.graph.edge_index_dict

        def nbrs(st,dt,nidx,side=0):
            out=set()
            for (s,_r,d),ei in ed.items():
                if s==st and d==dt:
                    for i in range(ei.shape[1]):
                        if ei[side,i].item()==nidx: out.add(ei[1-side,i].item())
            return out

        uc = {self.idx_to_category.get(i,f"c{i}") for i in nbrs("user","category",user_idx)
              if self.idx_to_category.get(i,"").lower()!="general"}
        us = {self.idx_to_sensory.get(i,f"s{i}") for i in nbrs("user","sensory_attr",user_idx)}
        pc = {self.idx_to_category.get(i,f"c{i}") for i in nbrs("poi","category",poi_idx)
              if self.idx_to_category.get(i,"").lower()!="general"}
        ps = {self.idx_to_sensory.get(i,f"s{i}") for i in nbrs("poi","sensory_attr",poi_idx)}
        mc = uc & pc; ms = us & ps

        shared: Dict[int,int] = defaultdict(int)
        for (st,_r,dt),ei in ed.items():
            if st=="user" and dt=="category":
                for i in range(ei.shape[1]):
                    u,c=ei[0,i].item(),ei[1,i].item()
                    if u!=user_idx and self.idx_to_category.get(c) in uc: shared[u]+=1
            if st=="user" and dt=="sensory_attr":
                for i in range(ei.shape[1]):
                    u,s=ei[0,i].item(),ei[1,i].item()
                    if u!=user_idx and self.idx_to_sensory.get(s) in us: shared[u]+=1
        similar = {u for u,cnt in shared.items() if cnt>=2}
        swr = similar & nbrs("user","poi",poi_idx,side=1)
        div_t = (["category"] if mc else []) + (["sensory"] if ms else []) + (["collaborative"] if swr else [])
        return {
            "preferences":{"matched_categories":list(mc),"matched_sensory":list(ms),
                           "cat_ratio":f"{len(mc)}/{len(uc)}","sensory_ratio":f"{len(ms)}/{len(us)}"},
            "similar_users":{"count":len(swr),"names":[self.idx_to_user.get(u,f"u{u}") for u in list(swr)[:10]]},
            "diversity":{"count":len(div_t),"types":div_t},
        }

    def find_shortest_paths(self, user_idx: int, poi_idx: int, k: int = 5) -> Optional[List]:
        """
        Find k shortest paths between user and POI through the knowledge graph,
        skipping direct user→poi edges and the 'General' category node.
        Each path is a list of (entity_type, entity_key, relation) tuples.
        """
        if self.graph is None:
            return None

        hetero_graph = self.graph
        if not hasattr(hetero_graph, 'edge_index_dict'):
            return None

        # ── build adjacency list (skip direct user→poi edges) ────────────────
        adj_list: Dict = {}
        for edge_type, edge_index in hetero_graph.edge_index_dict.items():
            if not isinstance(edge_type, tuple) or len(edge_type) != 3:
                continue
            src_type, relation, dst_type = edge_type
            try:
                num_edges = edge_index.shape[1]
            except Exception:
                continue
            for i in range(num_edges):
                try:
                    src_idx = int(edge_index[0, i].item())
                    dst_idx = int(edge_index[1, i].item())
                except Exception:
                    continue
                # Skip direct user→poi connections so paths go through intermediate nodes
                if src_type == 'user' and dst_type == 'poi':
                    continue
                src_node = (src_type, src_idx)
                dst_node = (dst_type, dst_idx)
                adj_list.setdefault(src_node, []).append((dst_node, relation))

        user_node  = ('user',  user_idx)
        target_node = ('poi', poi_idx)
        user_key   = self.user_str_key(user_idx)
        poi_key    = self.poi_str_key(poi_idx)

        if user_node not in adj_list:
            return None

        # ── BFS for k shortest paths ──────────────────────────────────────────
        queue       = deque([(user_node, [('user', user_key, None)], set())])
        visited     : Dict = {}
        found_paths : List = []
        iterations  = 0
        max_iter    = 50000

        while queue and iterations < max_iter and len(found_paths) < k:
            iterations += 1
            current, path, used_edges = queue.popleft()

            for neighbor, relation in adj_list.get(current, []):
                n_type, n_idx = neighbor

                # Resolve readable key
                if n_type == 'user':
                    n_key = self.idx_to_user.get(n_idx, f"user_{n_idx}")
                elif n_type == 'poi':
                    n_key = self.idx_to_poi.get(n_idx, f"poi_{n_idx}")
                elif n_type == 'category':
                    n_key = self.idx_to_category.get(n_idx, f"category_{n_idx}")
                    # Skip the generic "General" category node
                    if isinstance(n_key, str) and n_key.lower() == 'general':
                        continue
                    if isinstance(n_key, tuple):
                        n_key = " ".join(str(x) for x in n_key)
                elif n_type == 'sensory_attr':
                    n_key = self.idx_to_sensory.get(n_idx, f"sensory_{n_idx}")
                    if isinstance(n_key, tuple):
                        n_key = " ".join(str(x) for x in n_key)
                elif n_type == 'other':
                    n_key = self.idx_to_other.get(n_idx, f"other_{n_idx}")
                    if isinstance(n_key, tuple):
                        n_key = " ".join(str(x) for x in n_key)
                else:
                    n_key = f"{n_type}_{n_idx}"

                # Reached target POI
                if neighbor == target_node:
                    found_paths.append(path + [('poi', poi_key, relation)])
                    continue

                # Anti-backtrack
                if (neighbor, current) in used_edges:
                    continue

                # Allow re-visiting a node only if this path isn't much longer
                path_len = len(path)
                if neighbor in visited and path_len > visited[neighbor] + 2:
                    continue
                visited[neighbor] = min(visited.get(neighbor, float('inf')), path_len)

                new_used = used_edges | {(current, neighbor)}
                queue.append((neighbor, path + [(n_type, n_key, relation)], new_used))

        return found_paths if found_paths else None

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _format_path(path: List) -> str:
        """Turn a list of (entity_type, entity_key, relation) into a readable chain string."""
        parts = []
        for i, (etype, ekey, rel) in enumerate(path):
            label = str(ekey).replace("_", " ").title()
            if i == 0:
                parts.append(f"[{etype.upper()}] {label}")
            else:
                arrow = f" --[{rel}]--> " if rel else " --> "
                parts.append(f"{arrow}[{etype.upper()}] {label}")
        return "".join(parts)

    def explain_narrative(self, user_idx: int, poi_idx: int) -> Dict:
        """
        Find up to 5 graph paths from user → POI, format them, and ask OpenAI
        to explain in plain language why this POI was recommended.
        Falls back to a template string if OpenAI is unavailable.
        """
        poi_name = str(self.poi_display_name(poi_idx))
        user_key = str(self.user_str_key(user_idx))

        # ── find paths ───────────────────────────────────────────────────────
        paths = self.find_shortest_paths(user_idx, poi_idx, k=5)

        # ── user sensory profile (for fallback / extra context) ──────────────
        user_profile = self.get_profile(user_idx) or {}
        def _pick(obj, keys):
            for k in keys:
                v = obj.get(k)
                if v:
                    if isinstance(v, list): return [str(x) for x in v]
                    if isinstance(v, str):  return [x.strip() for x in v.split(",") if x.strip()]
            return []
        user_sensory = _pick(user_profile, [
            "sensoryPreferences","sensoryPreference","sensoryAttributes","sensoryAttrs",
            "sensory_preferences","sensory_attributes","sensory_attrs"
        ])
        user_cats = _pick(user_profile, [
            "preferredCategories","categories","categoryPreferences","category_preferences"
        ])

        # ── build paths section ──────────────────────────────────────────────
        if paths:
            path_lines = "\n".join(
                f"  Path {i+1}: {self._format_path(p)}"
                for i, p in enumerate(paths)
            )
            paths_section = f"Graph paths from user to this POI (top {len(paths)}):\n{path_lines}"
        else:
            paths_section = "Graph paths: none found (no connecting path through intermediate nodes)."

        # ── build full prompt (paths only — stats are shown separately in UI) ─
        context = f"""
User     : {user_key}
Recommended POI : {poi_name}

{paths_section}
""".strip()

        prompt = (
            "You are a recommendation explainer. You will receive internal graph evidence "
            "showing how a user connects to a POI through intermediate nodes like categories, "
            "sensory attributes (e.g. brightness, noise level, crowdedness, calmness), and "
            "other users with similar tastes.\n\n"
            "Your task: write a natural, friendly 2-3 sentence explanation of why this POI "
            "suits this user — like a knowledgeable friend recommending it.\n\n"
            "STRICT RULES:\n"
            "1. NEVER mention paths, graphs, nodes, edges, connections, or any technical terms.\n"
            "2. NEVER say phrases like 'based on the paths', 'the graph shows', 'path evidence', etc.\n"
            "3. Speak naturally about the user's inferred preferences and how the POI matches them.\n"
            "4. Pull out the specific attributes you see (e.g. quiet atmosphere, low brightness, "
            "casual dining) and weave them into the explanation naturally.\n"
            "5. If you see other users as intermediate nodes, phrase it as 'people with similar "
            "tastes to yours have enjoyed this place' — never say 'other users in the graph'.\n"
            "6. Write with confidence, as if you know the user's taste well.\n\n"
            "Internal evidence (extract insights from this, do NOT reproduce it in your output):\n"
            + context
        )

        # ── check key at call time (not import time) ─────────────────────────
        client, key = _get_openrouter_client()
        if not client:
            print(f"[OpenRouter] No client — falling back to template narrative")
            if paths:
                # Extract meaningful intermediate node labels (skip user and target poi nodes)
                sensory_nodes, category_nodes, user_nodes = [], [], []
                for path in paths:
                    for etype, ekey, _ in path[1:-1]:  # skip first (user) and last (target poi)
                        label = str(ekey).replace("_", " ").title()
                        if etype == "sensory_attr" and label not in sensory_nodes:
                            sensory_nodes.append(label)
                        elif etype == "category" and label not in category_nodes:
                            category_nodes.append(label)
                        elif etype == "user" and label not in user_nodes:
                            user_nodes.append(label)

                parts = []
                if sensory_nodes:
                    parts.append(f"it matches your sensory preferences like {', '.join(sensory_nodes[:3])}")
                if category_nodes:
                    parts.append(f"it fits categories you enjoy such as {', '.join(category_nodes[:3])}")
                if user_nodes:
                    parts.append(f"people with similar tastes to yours have enjoyed this place")

                if parts:
                    joined = "; ".join(parts)
                    narrative = f"You might enjoy {poi_name} because {joined}."
                else:
                    narrative = f"{poi_name} aligns well with your personal preferences and past interests."
            else:
                cats_str    = ", ".join(user_cats[:3])    or "your interests"
                sensory_str = ", ".join(user_sensory[:3]) or "your sensory preferences"
                narrative = (
                    f"{poi_name} was recommended because it aligns with {cats_str} "
                    f"and suits {sensory_str}."
                )
            return {"narrative": narrative, "source": "fallback", "paths_found": len(paths) if paths else 0}

        # ── OpenRouter call ───────────────────────────────────────────────────
        try:
            print(f"[OpenRouter] Calling API with model mistralai/mistral-7b-instruct:free")
            resp = client.chat.completions.create(
                model="google/gemma-3-27b-it:free",
                messages=[
                    {"role": "system", "content": "You explain POI recommendations clearly and concisely."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=250,
                temperature=0.7,
            )
            narrative = resp.choices[0].message.content.strip()
            print(f"[OpenRouter] Success, narrative length: {len(narrative)}")
            return {"narrative": narrative, "source": "openrouter", "paths_found": len(paths) if paths else 0}
        except Exception as exc:
            print(f"[OpenRouter] API call failed: {exc}")
            return {"narrative": f"Could not generate explanation: {exc}", "source": "error", "paths_found": 0}

    def stats(self):
        enriched_files = (list(ENRICHED_DIR.glob("enriched_poi_paths_chunk_*.pkl")) +
                          list(ENRICHED_DIR.glob("*_paths_chunk_*.pkl"))) if ENRICHED_DIR.exists() else []
        paths_source = ("none" if not self.user_poi_paths else
                        "pkl"       if enriched_files else
                        "generated")
        return {"num_users":self.num_users,"num_pois":self.num_pois,
                "poi_names_loaded":len(self.poi_names),
                "profiles_loaded":len([k for k in self.profiles if isinstance(k,str)]),
                "users_with_paths":len(self.user_poi_paths),
                "paths_source": paths_source,
                "model_loaded":self.model is not None,
                "gat_loaded":self.gat_embeddings is not None,
                "graph_loaded":self.graph is not None,"device":str(DEVICE)}


engine = Engine()
engine.load_all()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", stats=engine.stats(),
                           users=engine.get_user_list()[:500])

@app.route("/api/debug")
def api_debug():
    """Diagnostic endpoint — shows exactly what files were found and loaded."""
    enriched_files = []
    if ENRICHED_DIR.exists():
        for f in sorted(ENRICHED_DIR.iterdir()):
            enriched_files.append({"name": f.name, "size_mb": round(f.stat().st_size / 1e6, 2)})

    sample_users = sorted(engine.user_poi_paths.keys())[:5]
    sample_info  = {str(u): len(engine.user_poi_paths[u]) for u in sample_users}

    return jsonify({
        "paths": {
            "ENRICHED_DIR":          str(ENRICHED_DIR),
            "enriched_dir_exists":   ENRICHED_DIR.exists(),
            "files_found":           enriched_files,
            "GRAPH_PATH_exists":     GRAPH_PATH.exists(),
            "GAT_EMB_PATH_exists":   GAT_EMB_PATH.exists(),
            "MODEL_PATH_exists":     MODEL_PATH.exists(),
        },
        "engine": {
            **engine.stats(),
            "sample_users_with_paths": sample_info,
        }
    })

@app.route("/api/user-profile/<int:user_idx>")
def api_profile(user_idx):
    p = engine.get_profile(user_idx)
    if p is None: return jsonify({"success":False,"error":"No profile found"}), 404
    return jsonify({"success":True,"profile":p,"user_key":engine.user_str_key(user_idx)})

@app.route("/api/recommend")
def api_recommend():
    try:
        user_idx = int(request.args.get("user_id",0))
        top_k    = max(1, min(int(request.args.get("top_k",10)), 50))
    except ValueError:
        return jsonify({"error":"bad params"}), 400
    out = engine.recommend(user_idx, top_k)
    for rec in out["results"]: rec["explanation"] = engine.explain(user_idx, rec["poi_id"])
    return jsonify({"user_id":user_idx,"user_key":engine.user_str_key(user_idx),
                    "top_k":top_k,"method":out["method"],"results":out["results"]})

@app.route("/api/explain-narrative")
def api_explain_narrative():
    try:
        user_idx = int(request.args.get("user_id", 0))
        poi_idx  = int(request.args.get("poi_id",  0))
    except ValueError:
        return jsonify({"error": "bad params"}), 400
    try:
        result = engine.explain_narrative(user_idx, poi_idx)
    except Exception as exc:
        return jsonify({"error": str(exc), "narrative": f"Explanation failed: {exc}", "source": "error"}), 500
    return jsonify({"user_id": user_idx, "poi_id": poi_idx, **result})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860, threaded=True)