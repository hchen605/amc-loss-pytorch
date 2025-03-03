# amc-loss-pytorch

This is Pytorch implementation of AMC-loss. We extend the AMC-loss to additional modalities, such as audio and time-series.

Original AMC-loss paper: https://arxiv.org/abs/2004.09805

## Training

### MNIST 

Please run the command:
```
python run_train_mnist.py --model_type simple_model --dataset mnist --angular 0.5 --lambda_ 0.1 --save_path ./weights/mnist_re/ --note sim1

```

### Cifar-10
```
python run_train.py --dataset cifar10 --angular 0.25 --lambda_ 0.1 --save_path ./weights/cifar_re/ 

```

### Speech Command
```
python run_train_sc.py --batch_size 256 --angular 0.5 --lambda_ 0.1 --save_path ./weights/sc_re --note re6

```

### ECG5000
```
python run_train_ecg.py --angular 0.5 --lambda_ 0.1
```
## Evaluation

Please run the command:
```
python run_inference.py --dataset $DATASET --model_type $MODEL
```



