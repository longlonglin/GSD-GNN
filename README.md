# GSD-GNN
Our code is divided into a C++ part and a python part
# C++ dependence
The C++ part dependence versions that the code is tested:

| Dependence 	| Version     	|
|------------	|-------------	|
| g++        	| 11.4.0       	|
| cmake      	| 3.22.1      	|
| gflags     	| 2.1.2-3     	|
| openmp     	| 3.7.0-3     	|
| SWIG      	| 4.0.2       	|

## Preprocessing for C++ code 
Create an interface shared by C++ code and Python code
```bash
cd sampler
./compile.sh
```
### Start with conda for python part
```bash
conda create -n your_env_name python=3.8.16
```
Install dependencies by
```bash
pip install -r requirements.txt
```

### Run 
You can run the code using the following command
```bash
python main.py --dataname cora --gpu 0  --sample 4 --input_droprate 0.5 --hidden_droprate 0.5 --dropnode_rate 0.5 --hid_dim 32 --early_stopping 100 --lr 1e-2  --epochs 2000
python main.py --dataname citeseer --gpu 0  --sample 2 --input_droprate 0.0 --hidden_droprate 0.2 --dropnode_rate 0.5 --hid_dim 128 --early_stopping 100 --lr 1e-2  --epochs 2000
python main.py --dataname pubmed --gpu 0  --sample 4 --input_droprate 0.6 --hidden_droprate 0.8 --dropnode_rate 0.5 --hid_dim 16 --early_stopping 200 --lr 0.2 --epochs 2000 --use_bn
```
The dataset is automatically downloaded from the DGL library on execution.


## Modify hyperparameters

You can modify the hyperparameters in line 117 of the main.py file (we will update the code version later to make it easier to modify the parameters).