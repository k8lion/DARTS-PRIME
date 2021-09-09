This is the python version of https://github.com/SuReLI/DifferentiableNAS. 

Requirements:
```
Python >= 3.6.8, PyTorch >= 1.6.0, torchvision >= 0.7.0
```
We recommend training on GPU with CUDA >= 11.0.

CIFAR-10 and CIFAR-100 may be downloaded here: https://www.cs.toronto.edu/~kriz/cifar.html 

Penn TreeBank (preprocessed) may be downloaded here: https://github.com/salesforce/awd-lstm-lm

To run DARTS-PRIME search on CIFAR-10/100:
```
python src/train_scheduled_hard_statebn.py \\
  --task CIFAR100 \\ #remove this line for CIFAR-10
  --reg prox \\
  --schedfreq 10 \\
  --dyno_split \\
  --dyno_schedule
```


To run DARTS-PRIME search on Penn TreeBank:
```
python src/train_search_rnn.py \\
  --reg prox \\
  --schedfreq 10 \\
  --dyno_split \\
  --dyno_schedule
```
