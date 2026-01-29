# NucleoDock
**NucleoDock: A pretrained deep learning framework for sequence- and structure-aware nucleic acid–ligand docking and screening**

## Abstract
Cs

## Environment

Build 

## Evaluate on PDBBind Core Set.
```shell
docker run -v ./:/Docking --gpus all carsidock:v1 \
  python /Docking/run_core_set_eval.py \
  --cuda_convert
```

## Docking

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



The docking conformation will be stored in the outputs/screening_result folder with .sdf as the file name.
The score table will be stored in the outputs/screeing_outputs folder with score.dat as the file name. 

## Screening
The score table will be stored in the outputs/ace folder with score.dat as the file name. 




## License
The code of this repository is licensed under [Aapache Licence 2.0](https://www.apache.org/licenses/LICENSE-2.0). The use of the CarsiDock model weights is subject to the [Model License](./MODEL_LICENSE.txt). CarsiDock weights are completely open for academic research, please contact <bd@carbonsilicon.ai> for commercial use. 

## Checkpoints

If you agree to the above license, please download checkpoints from the following link and put them in the ``checkpoints`` folder.

Carsidock: [GoogleDrive](https://drive.google.com/file/d/1OweBn07R4bpoC0gETezKrOoK7xYreO4O/view?usp=drive_link) / [飞书](https://szuy1h04n8.feishu.cn/file/C3uqbkc6UoNI6kxsw2Ycg8cOnnf?from=from_copylink) 

RTMScore: [GitHub](https://github.com/sc8668/RTMScore/raw/main/trained_models/rtmscore_model1.pth)

## Copyright


## Citation
