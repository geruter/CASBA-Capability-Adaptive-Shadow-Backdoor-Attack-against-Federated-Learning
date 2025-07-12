# CASBA


## Installation
Install Pytorch

## Usage
### Prepare the dataset:

#### Tiny-imagenet dataset:

- download the dataset [tiny-imagenet-200.zip](https://tiny-imagenet.herokuapp.com/) into dir `./utils` 
- reformat the dataset.
```
cd ./utils
./process_tiny_data.sh
```

#### Others:
MNIST and CIFAR will be automatically download

### Reproduce experiments: 

- prepare the pretrained model:
Our pretrained clean models for attack can be downloaded from [Google Drive](https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing). You can also train from the round 0 to obtain the pretrained clean model.

- we can use Visdom to monitor the training progress.
```
python -m visdom.server -p 8098
```

- run experiments for the four datasets:
```
python main.py --params utils/X.yaml
```
`X` = `mnist_params`, `cifar_params`, Parameters can be changed in those yaml files to reproduce our experiments.
```
```
## Acknowledgement 
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
