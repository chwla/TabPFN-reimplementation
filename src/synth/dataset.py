import numpy as np
import pandas as pd
import networkx as nx
import random
import string
import os
import json
from sklearn.preprocessing import minmax_scale
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple, List

# --- 1. Expanded Configuration with Priors for Transformations & Structure ---

DIFFICULTY_PROFILES = {
    'easy': {
        'n_samples_range': (100, 1000), 'n_features_range': (4, 15),
        # SCM Structure Priors
        'scm_layers_range': (2, 4), 
        'scm_nodes_per_layer_range': (8, 20),
        'scm_sparsity_range': (0.1, 0.3),
        # SCM Transformation Priors (Probabilities)
        'transform_dist': {'Linear': 0.5, 'Polynomial': 0.2, 'Sinusoidal': 0.1, 'ReLU': 0.1, 'Sigmoid': 0.1, 'Interaction': 0.0},
        'poly_degree_range': (2, 3),
        'noise_scale_range': (0.01, 0.1),
        # Post-Processing & Corruption Priors
        'correlation_blocks': {'prob': 0.1, 'n_blocks_range': (1, 2), 'block_size_range': (2, 3), 'strength_range': (0.4, 0.6)},
        'missing_fraction_range': (0.0, 0.05), 'outlier_fraction_range': (0.0, 0.02),
        'warp_prob': 0.1, 'categorical_frac_range': (0.0, 0.2),
        # Target Priors
        'target_type_prob': {'regression': 0.7, 'classification': 0.3}, 'n_classes_range': (2, 3),
    },
    'medium': {
        'n_samples_range': (500, 5000), 'n_features_range': (15, 50),
        'scm_layers_range': (3, 6), 'scm_nodes_per_layer_range': (20, 50), 'scm_sparsity_range': (0.2, 0.5),
        'transform_dist': {'Linear': 0.3, 'Polynomial': 0.2, 'Sinusoidal': 0.2, 'ReLU': 0.1, 'Sigmoid': 0.1, 'Interaction': 0.1},
        'poly_degree_range': (2, 4),
        'noise_scale_range': (0.05, 0.3),
        'correlation_blocks': {'prob': 0.3, 'n_blocks_range': (1, 4), 'block_size_range': (2, 5), 'strength_range': (0.5, 0.8)},
        'missing_fraction_range': (0.01, 0.15), 'outlier_fraction_range': (0.01, 0.05),
        'warp_prob': 0.3, 'categorical_frac_range': (0.1, 0.4),
        'target_type_prob': {'regression': 0.5, 'classification': 0.5}, 'n_classes_range': (2, 10),
    },
    'hard': {
        'n_samples_range': (1000, 10000), 'n_features_range': (40, 100),
        'scm_layers_range': (5, 8), 'scm_nodes_per_layer_range': (50, 100), 'scm_sparsity_range': (0.4, 0.8),
        'transform_dist': {'Linear': 0.2, 'Polynomial': 0.2, 'Sinusoidal': 0.2, 'ReLU': 0.1, 'Sigmoid': 0.1, 'Interaction': 0.2},
        'poly_degree_range': (2, 5),
        'noise_scale_range': (0.1, 0.5),
        'correlation_blocks': {'prob': 0.6, 'n_blocks_range': (2, 6), 'block_size_range': (3, 8), 'strength_range': (0.7, 0.95)},
        'missing_fraction_range': (0.05, 0.3), 'outlier_fraction_range': (0.03, 0.1),
        'warp_prob': 0.5, 'categorical_frac_range': (0.2, 0.6),
        'target_type_prob': {'regression': 0.4, 'classification': 0.6}, 'n_classes_range': (5, 20),
    }
}

class TabPFNConfig:
    """Samples and holds all hyperparameters for generating a single dataset."""
    def __init__(self, seed: int, difficulty: str = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        if not difficulty: difficulty = self.rng.choice(['easy', 'medium', 'hard'], p=[0.3, 0.5, 0.2])
        self.difficulty = difficulty
        profile = DIFFICULTY_PROFILES[self.difficulty]

        # First, copy all profile attributes directly
        for key, value in profile.items():
            setattr(self, key, value)

        # Second, unpack any top-level _range tuples into specific values
        for key, value in profile.items():
            if isinstance(value, dict):
                continue

            if isinstance(value, tuple) and len(value) == 2:
                if isinstance(value[0], int):
                    setattr(self, key.replace('_range', ''), self.rng.randint(*value))
                else:
                    setattr(self, key.replace('_range', ''), self.rng.uniform(*value))
        
        # Add the missing logic to determine if the task is classification
        self.is_classification = self.rng.rand() < self.target_type_prob['classification']

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in ['rng'] and not callable(v)}


# --- 2. Modular SCM Transformation Library ---
class Transformation:
    def __call__(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError
    def to_dict(self) -> Dict[str, Any]: raise NotImplementedError

class Linear(Transformation):
    def __init__(self, rng): self.w, self.b = rng.randn(2)
    def __call__(self, x): return x * self.w + self.b
    def to_dict(self): return {'type': 'Linear', 'w': self.w, 'b': self.b}

class Polynomial(Transformation):
    def __init__(self, rng, degree_range):
        self.degree = rng.randint(*degree_range)
        self.coeffs = rng.randn(self.degree + 1)
    def __call__(self, x): return np.polyval(self.coeffs, x)
    def to_dict(self): return {'type': 'Polynomial', 'degree': self.degree, 'coeffs': self.coeffs.tolist()}

class Sinusoidal(Transformation):
    def __init__(self, rng): self.amp, self.freq, self.phase = rng.randn(3)
    def __call__(self, x): return self.amp * np.sin(self.freq * x + self.phase)
    def to_dict(self): return {'type': 'Sinusoidal', 'amp': self.amp, 'freq': self.freq, 'phase': self.phase}

class ReLU(Transformation):
    def __init__(self, rng): self.w, self.b = rng.randn(2)
    def __call__(self, x): return np.maximum(0, x * self.w + self.b)
    def to_dict(self): return {'type': 'ReLU', 'w': self.w, 'b': self.b}

class Sigmoid(Transformation):
    def __init__(self, rng): self.w, self.b = rng.randn(2)
    def __call__(self, x): 
        with np.errstate(over='ignore'):
            return 1 / (1 + np.exp(-(x * self.w + self.b)))
    def to_dict(self): return {'type': 'Sigmoid', 'w': self.w, 'b': self.b}
    
class Interaction(Transformation):
    def __init__(self, rng): self.w1, self.w2, self.w3 = rng.randn(3)
    def __call__(self, x1, x2): return x1*self.w1 + x2*self.w2 + x1*x2*self.w3
    def to_dict(self): return {'type': 'Interaction', 'w1': self.w1, 'w2': self.w2, 'w3': self.w3}


TRANSFORMATION_MAP = {
    'Linear': Linear, 'Polynomial': Polynomial, 'Sinusoidal': Sinusoidal,
    'ReLU': ReLU, 'Sigmoid': Sigmoid, 'Interaction': Interaction
}

# --- 3. Enhanced Structural Causal Model (SCM) ---
class StructuralCausalModel:
    def __init__(self, config: TabPFNConfig):
        self.config = config
        self.rng = config.rng
        self.graph = self._create_graph()
        self._assign_transformations()

    def _create_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        node_counter = 0; layers: List[List[int]] = []
        for _ in range(self.config.scm_layers):
            layer = [node_counter + i for i in range(self.config.scm_nodes_per_layer)]
            node_counter += self.config.scm_nodes_per_layer
            layers.append(layer)
        for i in range(self.config.scm_layers - 1):
            for u in layers[i]:
                for v in layers[i+1]:
                    if self.rng.rand() > self.config.scm_sparsity: G.add_edge(u, v)
        return G

    def _assign_transformations(self):
        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            if not parents: continue
            
            use_interaction = 'Interaction' in self.config.transform_dist and len(parents) >= 2
            probs = self.config.transform_dist.copy()
            if not use_interaction: probs['Interaction'] = 0
            
            prob_sum = sum(probs.values())
            if prob_sum > 0: probs = {k: v / prob_sum for k, v in probs.items()}

            transform_name = self.rng.choice(list(probs.keys()), p=list(probs.values()))
            
            if transform_name == 'Interaction':
                chosen_parents = self.rng.choice(parents, 2, replace=False)
                transform_obj = TRANSFORMATION_MAP[transform_name](self.rng)
                self.graph.nodes[node]['transform'] = {'obj': transform_obj, 'parents': chosen_parents.tolist()}
            else:
                transforms = {}
                for p in parents:
                    if transform_name == 'Polynomial':
                         transforms[p] = TRANSFORMATION_MAP[transform_name](self.rng, self.config.poly_degree_range)
                    else:
                         transforms[p] = TRANSFORMATION_MAP[transform_name](self.rng)
                self.graph.nodes[node]['transform'] = {'obj': transforms, 'parents': parents}

    def propagate(self) -> np.ndarray:
        n_samples, n_nodes = self.config.n_samples, self.graph.number_of_nodes()
        node_outputs = np.zeros((n_samples, n_nodes))
        CLIP_VALUE = 1e6
        
        for node in nx.topological_sort(self.graph):
            base_noise = self.rng.randn(n_samples) * self.config.noise_scale
            if node not in self.graph.nodes or 'transform' not in self.graph.nodes[node]:
                node_outputs[:, node] = base_noise; continue

            transform_info = self.graph.nodes[node]['transform']
            parents = transform_info['parents']
            
            if isinstance(transform_info['obj'], dict):
                parent_signals = [transform_info['obj'][p](node_outputs[:, p]) for p in parents]
                combined_signal = np.sum(parent_signals, axis=0)
            else:
                p1_out, p2_out = node_outputs[:, parents[0]], node_outputs[:, parents[1]]
                combined_signal = transform_info['obj'](p1_out, p2_out)

            combined_signal = np.clip(combined_signal, -CLIP_VALUE, CLIP_VALUE)
            node_outputs[:, node] = combined_signal + base_noise
        
        return np.nan_to_num(node_outputs, nan=0.0, posinf=CLIP_VALUE, neginf=-CLIP_VALUE)
    
    def get_metadata(self) -> Dict[str, Any]:
        structure = {'nodes': list(self.graph.nodes()), 'edges': []}
        for node in self.graph.nodes():
            if 'transform' in self.graph.nodes[node]:
                t_info = self.graph.nodes[node]['transform']
                if isinstance(t_info['obj'], dict):
                    for parent, obj in t_info['obj'].items():
                        structure['edges'].append({'source': parent, 'target': node, 'transform': obj.to_dict()})
                else:
                    structure['edges'].append({'source': t_info['parents'], 'target': node, 'transform': t_info['obj'].to_dict()})
        return structure

# --- 4. Post-Processing with Structured Correlations ---
def inject_structured_correlations(X: pd.DataFrame, config: TabPFNConfig) -> pd.DataFrame:
    if config.rng.rand() > config.correlation_blocks['prob']: return X
    
    used_cols = set()
    for _ in range(config.rng.randint(*config.correlation_blocks['n_blocks_range'])):
        available_cols = [c for c in X.columns if c not in used_cols]
        if len(available_cols) < 2: break
        
        block_size = config.rng.randint(*config.correlation_blocks['block_size_range'])
        block_cols = config.rng.choice(available_cols, min(block_size, len(available_cols)), replace=False)
        
        base_col = block_cols[0]
        for other_col in block_cols[1:]:
            strength = config.rng.uniform(*config.correlation_blocks['strength_range'])
            if X[base_col].std() > 1e-6:
                noise = config.rng.normal(0, X[base_col].std() * (1 - strength) * 0.5)
                X[other_col] = X[base_col] * strength + noise
        
        used_cols.update(block_cols)
    return X

def apply_postprocessing(X_raw, y_raw, config):
    X = pd.DataFrame(X_raw, columns=[f'feat_{i}' for i in range(X_raw.shape[1])])
    X = inject_structured_correlations(X, config)

    if config.rng.rand() < config.warp_prob:
        for col in X.select_dtypes(include=np.number).columns:
            u = minmax_scale(X[col]); a, b = config.rng.uniform(0.5, 5.0, 2)
            with np.errstate(invalid='ignore'):
                 X[col] = (1-(1-u+1e-6)**(1/b))**(1/a)*(X[col].max()-X[col].min())+X[col].min()

    n_cat = int(config.n_features * config.categorical_frac)
    for col in config.rng.choice(X.columns, n_cat, replace=False):
        try:
            X[col] = pd.qcut(X[col], q=config.rng.randint(2,20), labels=False, duplicates='drop')
            cats = [''.join(config.rng.choice(list(string.ascii_lowercase),k=5)) for _ in range(int(X[col].max()+1))]
            X[col] = X[col].apply(lambda x: cats[int(x)] if pd.notna(x) else x)
        except: pass

    X.loc[config.rng.rand(*X.shape) < config.missing_fraction] = np.nan
    n_outliers = int(config.n_samples * config.outlier_fraction)
    out_idx = config.rng.choice(config.n_samples, n_outliers, replace=False)
    for col in X.select_dtypes(include=np.number).columns:
        if config.rng.rand() < 0.5:
            col_std = X[col].std()
            if pd.notna(col_std) and col_std > 1e-6:
                ext_val = X[col].mean() + config.rng.choice([-1,1])*config.rng.uniform(5,10)*col_std
                X.loc[out_idx, col] = ext_val

    # Finalize target
    if config.is_classification:
        # Corrected line: Wrap the output of qcut in a pd.Series to use .fillna()
        binned_y = pd.qcut(y_raw, q=config.n_classes, labels=False, duplicates='drop')
        y = pd.Series(binned_y).fillna(0).astype(int)
    else:
        y = y_raw
        
    # Final cleanup of any potential NaNs from processing steps
    if X.isnull().values.any():
        X = X.fillna(X.median(numeric_only=True))
        
    return X, pd.Series(y, name='target')


# --- 5. Main Orchestration and Parallel Execution ---
def generate_dataset(config: TabPFNConfig) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    scm = StructuralCausalModel(config)
    all_node_data = scm.propagate()
    
    n_nodes = all_node_data.shape[1]
    if config.n_features + 1 > n_nodes: raise ValueError("SCM too small.")
    
    indices = config.rng.choice(n_nodes, config.n_features + 1, replace=False)
    X_raw, y_raw = all_node_data[:, indices[:-1]], all_node_data[:, indices[-1]]
    
    X, y = apply_postprocessing(X_raw, y_raw, config)
    
    metadata = {
        'config': config.to_dict(),
        'scm_structure': scm.get_metadata(),
        'selected_feature_nodes': indices[:-1].tolist(),
        'selected_target_node': int(indices[-1])
    }
    return X, y, metadata

def worker_generate_and_save(config: TabPFNConfig, save_dir: str) -> str:
    try:
        X, y, metadata = generate_dataset(config)
        df = pd.concat([X, y], axis=1)
        
        file_name = f"dataset_{config.seed}_{config.difficulty}"
        df.to_csv(os.path.join(save_dir, f"{file_name}.csv"), index=False)
        with open(os.path.join(save_dir, f"{file_name}.json"), 'w') as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(metadata, f, indent=2, cls=NpEncoder)
            
        return f"✅ Successfully generated {file_name}.csv"
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return f"❌ FAILED to generate dataset for seed {config.seed}: {e}\n{tb_str}"

def generate_dataset_suite(num_datasets: int, save_dir: str, n_jobs: int = -1):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    n_procs = cpu_count() if n_jobs == -1 else n_jobs
    
    configs = [TabPFNConfig(seed=i) for i in range(num_datasets)]
    args = [(config, save_dir) for config in configs]

    print(f"🚀 Starting generation of {num_datasets} datasets using {n_procs} processes...")
    with Pool(processes=n_procs) as pool:
        results = pool.starmap(worker_generate_and_save, args)
        for res in results: print(res)
    print("✨ Generation complete.")


if __name__ == '__main__':
    NUM_DATASETS_TO_GENERATE = 10
    SAVE_DIRECTORY = "./tabpfn_replicated_datasets"
    
    generate_dataset_suite(num_datasets=NUM_DATASETS_TO_GENERATE, save_dir=SAVE_DIRECTORY)
    
    print("\n--- Inspecting metadata of first generated dataset ---")
    first_json_path = None
    if os.path.exists(SAVE_DIRECTORY):
        for f in sorted(os.listdir(SAVE_DIRECTORY)):
            if f.endswith(".json"):
                first_json_path = os.path.join(SAVE_DIRECTORY, f)
                break

    if first_json_path and os.path.exists(first_json_path):
        with open(first_json_path, 'r') as f:
            meta = json.load(f)
        print(f"Successfully loaded metadata from {first_json_path}.")
        print("Config used:", meta['config'])
        print("\nSCM Structure (sample):")
        if meta['scm_structure']['edges']:
            print(json.dumps(meta['scm_structure']['edges'][:2], indent=2))
        else:
            print("No edges in this SCM.")
    else:
        print("Could not find any metadata file to inspect. All generations may have failed.")
