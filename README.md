# NucleoDock
**NucleoDock: A pretrained deep learning framework for sequence- and structure-aware nucleic acid–ligand docking and screening**
![Overview](image/framework.png)
## Abstract
Nucleic acids, including DNA and RNA, have emerged as critical therapeutic targets, offering substantial potential in drug discovery alongside traditional protein-based methods. Modulating nucleic acid function through small-molecule interactions enables the regulation of gene expression and disease pathways. However, experimental screening of nucleic acid targets remains highly resource-intensive, demanding significant time, cost, and labor. This has led to a gap in the development of efficient computational tools. Here, we present NucleoDock, the first deep learning framework designed for nucleic acid–small molecule docking. NucleoDock integrates both structure- and sequence-informed representations, operating at atomic and nucleotide levels. It combines an MDN-based geometric scoring mechanism with data augmentation from synthetic docking complexes. In benchmark evaluations, NucleoDock outperforms traditional methods, such as rDock, by 20% in top 1 conformation prediction. However, its performance in virtual screening is still limited. NucleoDock represents a significant step forward in computational drug discovery, offering a robust, data-driven tool for virtual screening and conformation prediction of nucleic acid targets.

## Environment

We provide two separate Conda environments: one for **data preparation** and one for **inference**.

### 1) Data Preparation Environment

During the data preparation stage, create the environment using `data/envir/environment_prep.yml` and install additional pip dependencies from `data/envir/data_prep_requirements.txt`:

```bash
conda env create -f data/envir/environment_prep.yml
conda activate data_prep
pip install -r data/envir/data_prep_requirements.txt

```
### 2) Inference

For inference, create the environment using environment.yml and install pip dependencies from requirements.txt:
```bash
conda env create -f environment.yml
conda activate inference2
pip install -r requirements.txt
```
#### Python Version Recommendation (Important)

We recommend using Python 3.8.

After the inference environment is installed, manually copy the following two items into the site-packages directory of the inference2 environment:
```
data/envir/pydock
data/envir/pydock-0.4-py3.8.egg-info
```

Target path (example):
```bash
/anaconda3/envs/inference2/lib/python3.8/site-packages
```
Note: The exact site-packages path may vary depending on your Conda installation. You can locate it via:
```bash
python -c "import site; print(site.getsitepackages())"
```
## Docking/screening

**docking/screening**

```shell
python  inference_graph.py \
  --config "RNA_graph_presentation.yml" \
  --input "inputs" \
  --ligands "inputs/candidates.txt" \
  --ckpt_path "checkpoints/graph_99.ckpt" \
  --output_dir "outputs/screening_result" \
  --num_threads 1 \
  --cuda_convert
```


The docking conformation will be stored in the ```outputs/screening_result``` folder with .sdf as the file name.
The score table will be stored in the ```outputs/screeing_outputs ``` folder with ```score.dat``` as the file name. 
 

## License
The code of this repository is licensed under [Aapache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0). The use of the NucleoDock model weights is subject to the [Model License](./MODEL_LICENSE.txt). NucleoDock weights are completely open for academic research.

## Checkpoints

If you agree to the above license, please download checkpoints from the following link and put them in the ``checkpoints`` folder.

The ckpt of Nt-v2 and rnafm can be downloaded from https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species and https://github.com/ml4bio/RNA-FM 

The ckpt of NucleoDock can be downloaded from 



