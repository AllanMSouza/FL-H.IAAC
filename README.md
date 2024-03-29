
# FL-H.IAAC  
  
FL-H.IAAC is a project based on Flower that implements the traditional architecture `client|server` of Federated Learning (FL).  
It has support for TensorFlow and PyTorch.   
  
## Current implemented strategies  
  
| Strategy | Pytorch | TensorFlow | Approach type |  
| :---         |     :---:      |     :---:     |--------------:|  
| [FedAvg](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) | :heavy_check_mark:  | :heavy_check_mark:  |   Traditional |  
| [FedPer](https://arxiv.org/pdf/1912.00818.pdf) | :heavy_check_mark: | :heavy_check_mark: |  Personalized |  
| [FedProto](https://ojs.aaai.org/index.php/AAAI/article/view/20819) | :heavy_check_mark: | :heavy_check_mark: |  Personalized |  
| [FedClassAvg](https://dl.acm.org/doi/pdf/10.1145/3545008.3545073) | :heavy_check_mark: | -|   Traditional |  
| [FedAvgM](https://arxiv.org/pdf/1909.06335.pdf) | :heavy_check_mark: | - |  Personalized |  
| [FedYogi](https://arxiv.org/pdf/2003.00295.pdf) | :heavy_check_mark: | - |   Traditional |  
| [QFedAvg](https://arxiv.org/pdf/1905.10497.pdf) | :heavy_check_mark: | - |   Traditional |  
| FedPredict | :heavy_check_mark: | - |  Personalization plugin |  
  
## Available datasets  
  
The following datasets are available both in IID and non-IID settings:  
  
- CIFAR-10  
- MNIST  
- Motion sense  
- UCI-HAR  
  
## Project structure - main components   
  
Each strategy is evaluated through `simulation.py` which creates the set of clients and the server of the selected strategy under the given setting.  
Additionally, FedLTA can generate various visualizations to help researchers to analyze the results (see folder `analysis`).  
  
 ├── analysis          # Tables and plots creation - Folder ├── client        # Client-side code - Folder │   ├── client_tf      # Client-side code for TensorFlow - Folder │   ├── client_torch       # Client-side code for Pytorch - Folder ├── data          # Datasets - Folder ├── dataset_utils_torch.py  # Dataset utilities for Pytorch ├── exc_all.sh        # Executes all the experiments ├── exc_joint_plot.sh      # Generates joint plots    ├── exec_simulation.py     # Executes the selected experiment  
 ├── execution_log      # Contains the terminal logs of each executed experiment - Folder ├── log_exc_all.txt ├── logs          # Contains the results of each executed experiment - Folder ├── model_definition_tf.py  # Deep learning models in TensorFlow ├── model_definition_torch.py   # Deep learning models in Pytorch ├── push_logs.sh       # Pushes the generated logs to GitHub    ├── server        # Server-side code  
 │   ├── common_base_server  # Common base server code (i.e., it is independent of Tensorflow and Pytorch) - Folder │   ├── server_tf      # TensorFlow-specific implementations of servers - Folder │   └── server_torch       # Pytorch-specific implementations of servers - Folder ├── simulation.py      # Main file. Executes the selected strategy in the Federated |Learning under the specified setting  
  
  
  
## Arguments used in `simulation.py`:  
- `--clients` `-c`: Total number of clients  
- `--strategy` `-s`: Strategy to simulate (e.g., FedAvg, FedProto, FedPer, FedAvgM, and FedPredict)   
- `--aggregation_method` `-a`: Method to aggregate parameters (e.g., `None` and `POC`)  
- `--model` `-m`: The machine learning model (e.g., DNN, CNN, or Logistic Regression)  
- `--dataset` `-d`: The dataset used (e.g., MNIST and CIFAR10)  
- `--epochs` `-e`:  Number of epochs in each round  
- `--round` `-r`: Number of communication rounds  
- `--poc` `-`: Percentage of clients selected per round  
- `--decay` `-`: Decay factor for FedLTA  
- `--non-iid` `-`: Whether to apply or not non-IID distribution (i.e., True or False)  
- `--classes` `-`: Number of classes  
- `--new_clients` `-`: whether to add or not new clients after a specific number of rounds  
- `--new_clients_train` `-`: Whether to train or not the new clients (i.e., True or False)  
  
## How to run  
  
The execution is divided into two options: (1) simulation, in which a strategy is executed under a given setting; (2) experiment, in which many simulations are executed under a given setting that represents a specific scenario.  
  
### Simulation  
  
The following code is an example of a simulation of the strategy "FedPredict", using the following configuration: dataset "MNIST" under the non-IID context,  
model "DNN", 1 local epoch, 10 rounds, 50 clients, tf (TensorFlow) framework.  
  
```python  
python simulation.py --strategy='FedPredict' --dataset='MNIST' --non-iid=True --model='DNN'   
 --epochs=1 --round=10 --client=10 --type='torch'  
```  
  
  
  
### Experiment  
  
The configuration of an experiment is pre-defined in the file "exec_experiments.py". The experiments to be executed are indicated in the file "exc_experiments.sh", which allows for running multiple experiments in sequence.  
To execute the experiments run:  
  
```  
bash exc_experiments.sh 
```

### Citing

If FedPredict has been useful to you, please cite our [paper](https://ieeexplore.ieee.org/abstract/document/10257293). The BibTeX is presented as follows:

```
@inproceedings{capanema2023fedpredict,
  title={FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning},
  author={Capanema, Cl{\'a}udio GS and de Souza, Allan M and Silva, Fabr{\'\i}cio A and Villas, Leandro A and Loureiro, Antonio AF},
  booktitle={2023 19th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)},
  pages={17--24},
  year={2023},
  doi={https://doi.org/10.1109/DCOSS-IoT58021.2023.00012},
  organization={IEEE}
}
```


