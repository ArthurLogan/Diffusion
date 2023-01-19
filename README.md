# Diffusion

复现《Denoising Diffusion Probabilistic Models》，基于官方tensorflow版本https://github.com/hojonathanho/diffusion和非官方pytorch实现https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-。



## 数据集

目前支持CIFAR10数据。

训练：

```shell
python3 train.py --dataset cifar10
```

生成：

```shell
python3 train.py --state eval
```



## 相关库

通过requirements.txt维护，通过如下命令安装相关库。

```shell
python3 -m pip install -r requirements.txt
```

依赖库如下：

* python>=3.7.5
* torch>=1.13.1
* torchvision>=0.14.1





