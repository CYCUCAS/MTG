import subprocess

node_task_datasets = ['PubMed', 'CiteSeer', 'Cora', 'ogbn-arxiv', 'Actor', 'Texas', 'Wisconsin']
graph_task_datasets = ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'COX2', 'BZR', 'DD']
pretrain_methods = ['None', 'DGI', 'GraphMAE', 'Edgepred_GPPT', 'Edgepred_Gprompt', 'GraphCL', 'SimGRACE']
prompt_types = ['None', 'GPPT', 'All-in-one', 'Gprompt', 'GPF', 'GPF-plus', 'MTG']
shot_nums = [1, 3, 5]

for dataset in node_task_datasets + graph_task_datasets:
    downstream_task = 'NodeTask' if dataset in node_task_datasets else 'GraphTask'
    for pretrain in pretrain_methods:
        if pretrain == 'None':
            for shot in shot_nums:
                cmd = (
                    f"python bench.py --pre_train_model_path 'None' "
                    f"--pretrain_task {pretrain} --downstream_task {downstream_task} --dataset_name {dataset} "
                    f"--gnn_type GCN --prompt_type 'None' --shot_num {shot} --hid_dim 128 "
                    f"--num_layer 2 --seed 42 --device 1"
                )
                print(f"Running command: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
        else:
            for prompt in prompt_types:
                for shot in shot_nums:
                    cmd = (
                        f"python bench.py --pre_train_model_path './Experiment/pre_trained_model/{dataset}/{pretrain}.GCN.128hidden_dim.pth' "
                        f"--pretrain_task {pretrain} --downstream_task {downstream_task} --dataset_name {dataset} "
                        f"--gnn_type GCN --prompt_type '{prompt}' --shot_num {shot} --hid_dim 128 "
                        f"--num_layer 2 --seed 42 --device 1"
                    )
                    print(f"Running command: {cmd}")
                    subprocess.run(cmd, shell=True, check=True)

print("All experiments are completed.")