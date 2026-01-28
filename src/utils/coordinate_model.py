import sys
import os

    
_project_root_guess = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root_guess not in sys.path:
# print(f"DEBUG (coordinate_model.py): Adding '{_project_root_guess}' to sys.path")
    sys.path.insert(0, _project_root_guess)
from src.utils.docking_utils import multithreading_process_single_docking
import torch
import argparse

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Docking with gradient")
    parser.add_argument("--input", type=str,
                        default='/mnt/d/workspace/data/4cr9.pkl',
                        
                        help="input file.")
    parser.add_argument("--output", type=str, default='4cr9.out.pkl', help="output path.")
    parser.add_argument(
        "--output-ligand", type=str, default='4cr9.out.sdf', help="output ligand sdf path."
    )
    args = parser.parse_args()

    multithreading_process_single_docking(args.input, args.output, args.output_ligand)
