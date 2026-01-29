import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm
import traceback

try:
    from src.utils.docking_inference_utils import read_ligands, docking
    from src.utils.docking_utils import read_mol 
    from src.utils.utils import get_abs_path, args_parse
    from src.data import Dictionary
    from src.modeling.modeling_hf_unimol import UnimolConfig
    
    from src.modeling.modeling_foldock2_all import FoldForDocking
    from src.modeling.NA_graph import DockingModelWrapper
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

try:
    import pydock
    USE_CUDA_LBFGS = True
except ImportError:
    print("[WARN] pydock not found. LBFGS running on CPU.")
    USE_CUDA_LBFGS = False


import copy
from torch_geometric.data import Batch, Data

class NAInjector(torch.nn.Module):
    def __init__(self, base_model, rna_batch_data, atom_to_res_map):
        super().__init__()
        self.base_model = base_model

        if isinstance(rna_batch_data, Batch):
            data_list = rna_batch_data.to_data_list()
            assert len(data_list) == 1, f"Expect 1 RNA graph, got {len(data_list)}"
            self.rna_single = data_list[0]
        else:
            self.rna_single = rna_batch_data

        self.register_buffer("atom_to_res_map", atom_to_res_map)

        self._cached_B = None
        self._cached_device = None
        self._cached_rna_batch = None

    def _get_rna_batch(self, B: int, device: torch.device):
        if self._cached_rna_batch is None or self._cached_B != B or self._cached_device != device:
            data_list = [copy.deepcopy(self.rna_single) for _ in range(B)]
            self._cached_rna_batch = Batch.from_data_list(data_list).to(device)
            self._cached_B = B
            self._cached_device = device
        return self._cached_rna_batch

    def forward(self, *args, **kwargs):
        if "mol_src_tokens" in kwargs:
            B = kwargs["mol_src_tokens"].shape[0]
            current_device = kwargs["mol_src_tokens"].device
        elif len(args) > 0 and torch.is_tensor(args[0]):
            B = args[0].shape[0]
            current_device = args[0].device
        else:
            B = 1
            current_device = self.atom_to_res_map.device

        map_expanded = self.atom_to_res_map.unsqueeze(0).expand(B, -1)

        rna_batch = self._get_rna_batch(B, current_device)

        return self.base_model(
            *args,
            rna_batch_data=rna_batch,
            atom_to_res_map=map_expanded,
            **kwargs
        )

def load_model_and_config(ckpt_path, yaml_path, device, mol_dict_path, pocket_dict_path):
    print(f"[INFO] Parsing config from {yaml_path}")
    cfg = args_parse(yaml_path)
    
    mol_dict = Dictionary.load(mol_dict_path)
    pocket_dict = Dictionary.load(pocket_dict_path)

    hidden_size = getattr(cfg.MODEL, 'HIDDEN_SIZE', 768)
    num_heads = getattr(cfg.MODEL, 'NUM_ATTENTION_HEADS', 16)
    num_layers = getattr(cfg.MODEL, 'NUM_LAYERS', 6)
    recycling = getattr(cfg.MODEL, 'RECYCLING', 3)

    print(f"[CONFIG] Layers={num_layers}, Hidden={hidden_size}, Heads={num_heads}")

    config = UnimolConfig(
        num_hidden_layers=num_layers,
        recycling=recycling,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        mol_config=UnimolConfig(vocab_size=len(mol_dict)+1, hidden_size=hidden_size, num_attention_heads=num_heads, num_hidden_layers=num_layers),
        pocket_config=UnimolConfig(vocab_size=len(pocket_dict)+1, hidden_size=hidden_size, num_attention_heads=num_heads, num_hidden_layers=num_layers)
    )

    config.rna_input_dim = getattr(cfg.MODEL, "RNA_INPUT_DIM", 771)
    config.rna_embed_dim = getattr(cfg.MODEL, "RNA_EMBED_DIM", 128)
    config.rna_gcn_depth = getattr(cfg.MODEL, "RNA_GCN_DEPTH", 3)
    
    print(f"[CONFIG] RNA Module: In={config.rna_input_dim}, Hidden={config.rna_embed_dim}")


    print(f"[INFO] Loading Checkpoint: {ckpt_path}")

    pl_model = DockingModelWrapper.load_from_checkpoint(
        get_abs_path(ckpt_path),
        training_config=cfg,
        model_config=config,
        map_location=device,
        strict=True 
    )
    
    base_model = pl_model.model
    base_model.to(device).eval()
    
    return base_model, mol_dict, pocket_dict


def flush_results(results, output_dir):
    if not output_dir or not results: return
    out_dir_abs = get_abs_path(output_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    
    df = pd.DataFrame(results)
    if "score" in df.columns:
        df = df.sort_values(by="score", ascending=True)
    
    final_path = os.path.join(out_dir_abs, "score.dat")
    tmp_path = final_path + ".tmp"
    df.to_csv(tmp_path, index=False, sep="\t")
    os.replace(tmp_path, final_path)


def main(args):
    local_cuda_index = 0
    device = torch.device(f"cuda:{local_cuda_index}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    lbfgsbsrv = None
    if args.cuda_convert and USE_CUDA_LBFGS:
        lbfgsbsrv = pydock.LBFGSBServer(args.num_threads, local_cuda_index)
        print("[INFO] CUDA LBFGSB enabled.")


    base_model, mol_dict, pocket_dict = load_model_and_config(
        args.ckpt_path, args.config, device,
        get_abs_path('example_data/molecule/dict.txt'),
        get_abs_path('example_data/pocket/dict.txt')
    )

 
    print(f"[INFO] Loading Input Data from: {args.input_dir}")
    

    pocket_sdf_path = os.path.join(args.input_dir, "pocket.sdf")
    if not os.path.exists(pocket_sdf_path):
        raise FileNotFoundError(f"Missing pocket.sdf in {args.input_dir}")
    

    pocket = read_mol(pocket_sdf_path)[0]
    
    rna_pkl_path = os.path.join(args.input_dir, "rna_data.pkl")
    with open(rna_pkl_path, 'rb') as f:
        rna_data = pickle.load(f)
        
    rna_batch = rna_data['rna_batch_data'].to(device)
    rna_map = rna_data['atom_to_res_map'].long().to(device)
    
    print(f"[INFO] Receptor Loaded. Atoms in Map: {len(rna_map)}")

    # Injector
    model = NAInjector(base_model, rna_batch, rna_map).to(device)
    model.eval()

    # prepare ligand
    if args.ligands.endswith('.txt'):
        with open(get_abs_path(args.ligands), 'r') as f:
            smiles_list = [line.strip().split()[0] for line in f if line.strip()]
    elif args.ligands.endswith('.sdf'):
        suppl = Chem.SDMolSupplier(get_abs_path(args.ligands))
        smiles_list = [Chem.MolToSmiles(m) for m in suppl if m]
    else:
        raise ValueError("Ligands file must be .txt or .sdf")

    # inference
    results = []
    out_dir = get_abs_path(args.output_dir) if args.output_dir else None
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    
    print(f"[INFO] Start docking {len(smiles_list)} ligands...")

    for i, smi in enumerate(tqdm(smiles_list, desc="Docking")):
        try:
            
            init_mol_list = read_ligands(smiles=[smi])[0]
            if not init_mol_list: continue
            
            torch.cuda.empty_cache()

          
            out_path = None
            sdf_name = None
            if out_dir:
                sdf_index = len(results) + 1
                sdf_name = f"{sdf_index}.sdf"
                out_path = os.path.join(out_dir, sdf_name)

            
            outputs = docking(
                model, 
                pocket, 
                init_mol_list, 
                mol_dict, 
                pocket_dict, 
                device=device, 
                output_path=out_path, #
                num_threads=args.num_threads, 
                lbfgsbsrv=lbfgsbsrv
            )

            
            scores = np.array(outputs['conf_ranking_score'], dtype=float)
            best_idx = int(scores.argmin())
            
            results.append({
                "sdf_file": sdf_name,
                "smiles": smi,
                "score": float(scores[best_idx]),
                "convert_loss": float(outputs['conf_convert_loss'][best_idx])
            })

            
            if len(results) % 500 == 0:
                flush_results(results, args.output_dir)

        except Exception as e:
            # print(f"[WARN] Failed {smi}: {e}")
            pass

    flush_results(results, args.output_dir)
    print(f"[SUCCESS] Done. Results in {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="RNA_graph_presentation.yml", help='Training YAML config')
    parser.add_argument('--ckpt_path', default=True, help='Model checkpoint (.ckpt)')
    parser.add_argument('--input_dir', default="inputs", help='Directory containing pocket.sdf and rna_data.pkl')
    parser.add_argument('--ligands', default='"inputs/candidates.txt"', help='SMILES txt')
    parser.add_argument('--output_dir', default="checkpoints/graph_99.ckpt")
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--cuda_convert', action='store_true')
    
    args = parser.parse_args()
    main(args)