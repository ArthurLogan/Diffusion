# Diffusion

复现论文《Denoising Diffusion Probabilistic Models》。基于

* 官方tensorflow版本<https://github.com/hojonathanho/diffusion>。
* 非官方pytorch实现<https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm->。



## 数据集

目前支持CIFAR10数据。

训练：

```shell
python train.py --dataset cifar10
```

生成：

```shell
python train.py --state eval --checkpoint_name ckpt_199_.pt
```



## 相关库

通过requirements.txt维护，通过如下命令安装相关库。

```shell
python -m pip install -r requirements.txt
```

