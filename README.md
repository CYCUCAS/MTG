<p align="center">
  <img src="https://github.com/CYCUCAS/MTG/blob/main/%E2%80%8EMTG.png" width="50%">
</p>

<h2 align="center"><span style="color:#8000FF">✻MTG</span>: Message Tuning for Graph Foundation Models</h2>

<div align="center">

<img src="https://img.shields.io/badge/Latest_version-v1-8000FF" style="height: 26px;">
<img src="https://img.shields.io/badge/PyTorch-v2.2.1-8000FF" style="height: 26px;">
<img src="https://img.shields.io/badge/python->=3.12-8000FF" style="height: 26px;">
<img src="https://img.shields.io/badge/license-MIT-8000FF" style="height: 26px;">

</div>

<span style="color:#8000FF">**✻MTG**</span> is the PyTorch implementation of "**With Great Power Comes Great Adaptation: Message Tuning Outshines Prompt Tuning for Graph Foundation Models**", built upon the graph prompting library [<span style="color:black">**🌟ProG🌟**</span>](https://github.com/sheldonresearch/ProG).

## Datasets
| Dataset    | Task  | # Graphs | # Nodes  | # Edges   | # Features | # Classes | Graph Type      |
|------------|-------|----------|----------|-----------|------------|-----------|-----------------|
| Cora       | Node  | 1        | 2,708    | 5,429     | 1,433      | 7         | Homophilic      |
| CiteSeer   | Node  | 1        | 3,327    | 9,104     | 3,703      | 6         | Homophilic      |
| Pubmed     | Node  | 1        | 19,717   | 88,648    | 500        | 3         | Homophilic      |
| Texas      | Node  | 1        | 183      | 325       | 1,703      | 5         | Heterophilic    |
| Actor      | Node  | 1        | 7,600    | 30,019    | 932        | 5         | Heterophilic    |
| Wisconsin  | Node  | 1        | 251      | 515       | 1,703      | 5         | Heterophilic    |
| ogbn-arxiv | Node  | 1        | 169,343  | 1,166,243 | 128        | 40        | Large-scale     |
| D&D        | Graph | 1,178    | 284.1    | 715.7     | 89         | 2         | Proteins        |
| ENZYMES    | Graph | 600      | 32.6     | 62.1      | 3          | 6         | Proteins        |
| PROTEINS   | Graph | 1,113    | 39.1     | 72.8      | 3          | 2         | Proteins        |
| BZR        | Graph | 405      | 35.8     | 38.4      | 3          | 2         | Small Molecule  |
| COX2       | Graph | 467      | 41.2     | 43.5      | 3          | 2         | Small Molecule  |
| MUTAG      | Graph | 188      | 17.9     | 19.8      | 7          | 2         | Small Molecule  |
| COLLAB     | Graph | 5,000    | 74.5     | 2,457.8   | 0          | 3         | Social Network  |
| IMDB-B     | Graph | 1,000    | 19.8     | 96.53     | 0          | 2         | Social Network  |

## Environment Setup
To proceed, ensure that Anaconda or Miniconda is installed on your system. This guide requires a CUDA-capable GPU for optimal execution.

```shell
# Create and activate a new Conda environment named 'MTG'
conda create -n MTG
conda activate MTG

# Install Pytorch and PyG with CUDA Version 12.5
# If your use a different CUDA version, please refer to the PyTorch and DGL websites for the appropriate versions.
conda install numpy
conda install pytorch==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg==2.5.2 

# Install additional dependencies
pip install pandas torchmetrics Deprecated 
pip install pyg-lib torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```
The torch and cuda version can refer to https://data.pyg.org/whl/.


## Quick Start
Before running the script, add the project root directory to the Python path.

**Note**: Replace **user_name** with the actual username path.

```bash
export PYTHONPATH=/home/user_name/MTG-main:$PYTHONPATH
```

### Pre-train GFMs
Taking the **Cora** dataset and **GCN** model as an example.
```shell
python pre_train.py --pretrain_task 'DGI' --dataset_name Cora --gnn_type GCN --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
python pre_train.py --pretrain_task 'GraphMAE' --dataset_name Cora --gnn_type GCN --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
python pre_train.py --pretrain_task 'Edgepred_GPPT' --dataset_name Cora --gnn_type GCN --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
python pre_train.py --pretrain_task 'Edgepred_Gprompt' --dataset_name Cora --gnn_type GCN --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
python pre_train.py --pretrain_task 'GraphCL' --dataset_name Cora --gnn_type GCN --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
python pre_train.py --pretrain_task 'SimGRACE' --dataset_name Cora --gnn_type GCN --hid_dim 128 --num_layer 2 --epochs 1000 --seed 42 --device 0
```
### Prompt Tuning for Downstream Tasks
Taking the **Cora** dataset, **GCN** model, and **GraphMAE** pretrain task as an example.

#### With Customized Hyperparameters
```shell
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'GPPT' --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'Gprompt' --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'GPF' --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'GPF-plus' --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'All-in-one' --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
```

#### With Optimal Hyperparameters through Random Search
```shell
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --pretrain_task GraphMAE --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'GPPT' --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --pretrain_task GraphMAE --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'Gprompt' --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --pretrain_task GraphMAE --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'GPF' --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --pretrain_task GraphMAE --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'GPF-plus' --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --pretrain_task GraphMAE --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type 'All-in-one' --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
```

### Message Tuning for Downstream Tasks
Taking the **Cora** dataset and **GCN** model as an example.

#### With Customized Hyperparameters
```shell
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/DGI.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/Edgepred_GPPT.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/Edgepred_Gprompt.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
python downstream_task.py --pre_train_model_path './Experiment/pre_trained_model/Cora/SimGRACE.GCN.128hidden_dim.pth' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.001 --decay 1e-5 --seed 42 --device 0
```

#### With Optimal Hyperparameters through Random Search
```shell
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/DGI.GCN.128hidden_dim.pth' --pretrain_task 'DGI' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphMAE.GCN.128hidden_dim.pth' --pretrain_task 'GraphMAE' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/Edgepred_GPPT.GCN.128hidden_dim.pth' --pretrain_task 'Edgepred_GPPT' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/Edgepred_Gprompt.GCN.128hidden_dim.pth' --pretrain_task 'Edgepred_Gprompt' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/GraphCL.GCN.128hidden_dim.pth' --pretrain_task 'GraphCL' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
python bench.py --pre_train_model_path './Experiment/pre_trained_model/Cora/SimGRACE.GCN.128hidden_dim.pth' --pretrain_task 'SimGRACE' --downstream_task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type MTG --shot_num 1 --hid_dim 128 --num_layer 2 --seed 42 --device 0
```
