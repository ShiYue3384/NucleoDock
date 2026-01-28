import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from Bio.PDB import PDBParser, NeighborSearch
from Bio import BiopythonWarning
from rdkit import Chem
from torch_geometric.data import Data, Batch

# 引入 HuggingFace / RNA-FM
from transformers import AutoTokenizer, AutoModelForMaskedLM
from multimolecule import RnaTokenizer, RnaFmModel

# 引入 Uni-Mol 字典
try:
    from dictionary import Dictionary
except ImportError:
    print("[Error] Cannot import Dictionary from src.data.unicore")
    sys.exit(1)

warnings.simplefilter('ignore', BiopythonWarning)
warnings.filterwarnings("ignore")

# =============================================================================
# 配置与辅助函数
# =============================================================================

RESIDUE_MAP = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}
allow_pocket_atoms = ['C', 'H', 'N', 'O', 'S', 'P']

def get_1letter_code(resname):
    return RESIDUE_MAP.get(resname.strip().upper(), 'X')

def filter_pocketatoms(atom_name):
    if not atom_name: return None
    element = "".join([c for c in atom_name if c.isalpha()])
    if len(element) > 0 and element[:1] in allow_pocket_atoms:
        return element[:1]
    return None

# =============================================================================
# Embedding generator
# =============================================================================
class EmbeddingGenerator:
    def __init__(self, nt_model_dir, rnafm_model_dir, device='cuda'):
        self.device = device
        print(f"[INFO] Loading Sequence Models on {device}...")
        
        self.nt_tok = AutoTokenizer.from_pretrained(nt_model_dir, trust_remote_code=True, local_files_only=True)
        self.nt_mdl = AutoModelForMaskedLM.from_pretrained(nt_model_dir, trust_remote_code=True, local_files_only=True).eval().to(device)
        self.nt_h = self.nt_mdl.config.hidden_size
        
        self.rna_tok = RnaTokenizer.from_pretrained(rnafm_model_dir, local_files_only=True)
        self.rna_mdl = RnaFmModel.from_pretrained(rnafm_model_dir, local_files_only=True).eval().to(device)
        self.rna_h = self.rna_mdl.config.hidden_size
        self.base_dim = max(self.nt_h, self.rna_h)

    def _expand_kmer(self, emb, target_len, k=6):
        """Fix the NT-v2 6-mer issue"""
        L_curr, H = emb.shape
        if L_curr > 2: core = emb[1:-1] 
        else: core = emb
        
        expanded = core.unsqueeze(1).repeat(1, k, 1).reshape(-1, H)
        
        if expanded.shape[0] >= target_len: return expanded[:target_len]
        else: return F.pad(expanded, (0, 0, 0, target_len - expanded.shape[0]))

    def _pad_and_tag(self, vec, target_dim, type_idx):
        if vec.shape[1] < target_dim:
            vec = F.pad(vec, (0, target_dim - vec.shape[1]))
        tail = torch.zeros(vec.shape[0], 3, device=vec.device)
        tail[:, type_idx] = 1.0
        return torch.cat([vec, tail], dim=1)

    def get_features(self, sequence):
        """(Embedding[L, 771], EdgeIndex[2, E])"""
        is_dna = 'T' in sequence.upper()
        target_len = len(sequence)
        
        # --- 1. Embedding ---
        with torch.no_grad():
            if is_dna:
                seq_fmt = sequence.upper().replace("U", "T")
                inputs = self.nt_tok([seq_fmt], return_tensors="pt", padding=True, truncation=True)
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                out = self.nt_mdl(**inputs, output_hidden_states=True)
                hs = out.hidden_states[-1][0] 
                
                # Check 6-mer
                if hs.shape[0] < target_len * 0.8:
                    hs = self._expand_kmer(hs, target_len)
                else:
                    # standard cls/eos removal
                    if hs.shape[0] == target_len + 2: hs = hs[1:-1]
                
                # Double check length
                if hs.shape[0] != target_len:
                     if hs.shape[0] > target_len: hs = hs[:target_len]
                     else: hs = F.pad(hs, (0,0,0, target_len - hs.shape[0]))

                emb = self._pad_and_tag(hs, self.base_dim, 0)
            else:
                seq_fmt = sequence.upper().replace("T", "U")
                inputs = self.rna_tok([seq_fmt], return_tensors="pt", truncation=True)
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                out = self.rna_mdl(**inputs)
                hs = out.last_hidden_state[0]
                
                if hs.shape[0] == target_len + 2: hs = hs[1:-1]
                elif hs.shape[0] > target_len: hs = hs[:target_len]
                elif hs.shape[0] < target_len: hs = F.pad(hs, (0,0,0, target_len - hs.shape[0]))
                
                emb = self._pad_and_tag(hs, self.base_dim, 1)

        # --- 2. 2D Structure ---
        seq_fold = sequence.upper().replace("T", "U")
        try:
            p = subprocess.Popen(['RNAfold', '--noPS'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            o, _ = p.communicate(input=seq_fold)
            structure = o.strip().split('\n')[-2].split()[0] if p.returncode==0 else "."*target_len
        except:
            structure = "."*target_len
            
        # Parse Edge
        edges = []
        stack = []
        for i, char in enumerate(structure):
            if char == '(': stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    edges.append((i, j)); edges.append((j, i))
        for i in range(len(structure)-1):
            edges.append((i, i+1)); edges.append((i+1, i))
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros(2,0).long()
        
        return emb.cpu(), edge_index
        
  

# =============================================================================
# Generate Pocket Mol and Map
# =============================================================================
def process_and_save(pdb_file, reflig_path, emb_gen, output_dir):
    print(f"[INFO] Processing {os.path.basename(pdb_file)}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract the Pocket atomic object
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('tmp', pdb_file)
    
    mol = Chem.MolFromMolFile(reflig_path)
    if not mol: mol = Chem.MolFromMol2File(reflig_path)
    lcoords = mol.GetConformer().GetPositions()
    
    all_atoms = [a for a in structure.get_atoms() if a.element != 'H' and a.get_parent().id[0] == ' ']
    ns = NeighborSearch(all_atoms)
    
    pocket_atoms_obj = []
    seen = set()
    for c in lcoords:
        for a in ns.search(c, 8.0, level='A'):
            if not filter_pocketatoms(a.element): continue
            if id(a) not in seen:
                seen.add(id(a))
                pocket_atoms_obj.append(a)
    
    if not pocket_atoms_obj: raise ValueError("No pocket atoms")

    # Generate the RDKit Mol object for Pocket and save it as SDF
    
    rw_mol = Chem.RWMol()
    conf = Chem.Conformer()
    
    valid_indices = [] 
    
    for idx, atom_obj in enumerate(pocket_atoms_obj):
        sym = filter_pocketatoms(atom_obj.element)
        
        new_atom = Chem.Atom(sym)
        idx_in_mol = rw_mol.AddAtom(new_atom)
        
        
        coord = atom_obj.get_coord()
        conf.SetAtomPosition(idx_in_mol, (float(coord[0]), float(coord[1]), float(coord[2])))
        valid_indices.append(idx)

    mol_out = rw_mol.GetMol()
    mol_out.AddConformer(conf)
    
   
    pocket_sdf_path = os.path.join(output_dir, "pocket.sdf")
    w = Chem.SDWriter(pocket_sdf_path)
    w.write(mol_out)
    w.close()
    print(f"[INFO] Saved Pocket SDF to {pocket_sdf_path}")

    # Generate RNA Graph and Map (based on pocket_atoms_obj)
    # Determine the main chain
    counts = {}
    for a in pocket_atoms_obj:
        c = a.get_parent().get_parent().id
        counts[c] = counts.get(c, 0) + 1
    main_chain_id = max(counts, key=counts.get)
    
    # Extract sequence
    chain = structure[0][main_chain_id]
    valid_res = [r for r in chain.get_residues() if r.id[0] == ' ']
    seq = "".join([get_1letter_code(r.get_resname()) for r in valid_res])
    
    # generate Graph
    emb, edge_index = emb_gen.get_features(seq)
    rna_graph = Data(x=emb, edge_index=edge_index)
    
    # generate Map
    res_lookup = {r.id: i for i, r in enumerate(valid_res)}
    map_list = [0] # BOS 
    
    for atom in pocket_atoms_obj:
        idx = 0
        if atom.get_parent().get_parent().id == main_chain_id:
            idx = res_lookup.get(atom.get_parent().id, 0)
        map_list.append(idx)
        
    map_list.append(0) # EOS 
    
    atom_to_res_map = torch.tensor(map_list, dtype=torch.long)
    
    # save NA data
    rna_data = {
        'rna_batch_data': Batch.from_data_list([rna_graph]),
        'atom_to_res_map': atom_to_res_map
    }
    rna_pkl_path = os.path.join(output_dir, "rna_data.pkl")
    with open(rna_pkl_path, "wb") as f:
        pickle.dump(rna_data, f)
    print(f"[INFO] Saved RNA Data to {rna_pkl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", required=True)
    parser.add_argument("--reflig", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nt_model", required=True)
    parser.add_argument("--rnafm_model", required=True)
    args = parser.parse_args()
    
    emb_gen = EmbeddingGenerator(args.nt_model, args.rnafm_model, 'cpu')
    process_and_save(args.pdb_file, args.reflig, emb_gen, args.output_dir)