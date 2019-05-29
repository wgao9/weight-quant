# Weight Quantization

This is the source code for the paper [*Information-Theoretic Understanding of Population Risk Improvement with Model Compression*]
(https://arxiv.org/abs/1901.09421)

To use this code, it is required to install **Python3**, with latest *NumPy*, *PyTorch*, *TorchVision*.

Two steps for running experiments:

**Step 1**: Retraining the model by subsampling
```
python3 retrain.py --type=mnist --subsample_rate=0.1
```
Then the retrained model will be  stored in *sub_models/mnist/ssr=100/model*

Note that if you set --compute_gradient=True or --compute_hessian=True, the gradient and hessian of the weights are also computed and stored in *sub_models/mnist/ssr=100/importances*

**Step 2**: load pretrained model and run
```
python3 deep_compress.py --type=mnist --mode=normal --fix_bit=2 --subsample_rate=0.1
```
and the results are
```
Before quantization, type=mnist, training acc1=0.9956+-0.0023, acc5=1.0000+-0.0001, loss=0.015394+-0.007845
Before quantization, type=mnist, validation acc1=0.9468+-0.0046, acc5=0.9959+-0.0007, loss=0.320736+-0.029296
Compression ratio = 0.0585+-0.0004
After quantization, type=mnist, training acc1=0.9839+-0.0068, acc5=0.9999+-0.0001, loss=0.060263+-0.015268
After quantization, type=mnist, validation acc1=0.9286+-0.0075, acc5=0.9941+-0.0009, loss=0.259629+-0.018100
```

**Important arguments**

1. *--type* denotes the dataset and model architecture. Supported: mnist, cifar10, cifar100

2. *--mode* denotes the weight quantization algorithms. Supported:
    - Typical K-means: Set --mode=normal
    - Hessian-weighted K-means: Set --compute_hessian=True for retraining and --mode=hessian for quantization
    - Diameter-regularized Hessian-weighted (*DRHW*) K-means: Set --compute_hessian=True for retraining and --mode=hessian and --diameter_reg>0 for quantization
    
3. *--subsample_rate* denotes the sub sample rate for retraining data. We use 0.1 for MNIST and 0.2 for CIFAT 

4. *--fix_bits* denotes the bits for model compression (same for all layer). If you want different ratio for different layer, use --bits=[3,4,5] for example

5. Please see the helps in the code for other arguments.

**Contact** wgao9@illinois.edu

