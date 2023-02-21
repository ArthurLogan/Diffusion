# Diffusion

复现论文列表
* [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
* [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

参考代码

* ddpm官方tensorflow实现<https://github.com/hojonathanho/diffusion>
* ddpm非官方pytorch实现<https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm->
* ddim官方pytorch实现<https://github.com/ermongroup/ddim>



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

