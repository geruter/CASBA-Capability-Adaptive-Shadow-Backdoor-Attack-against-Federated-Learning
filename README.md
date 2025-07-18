# CASBA


## Installation
Install Pytorch

## Usage
You can also train from the round 0 to obtain the pretrained clean model.

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
- [AI-secure/DBA](https://github.com/AI-secure/DBA)
