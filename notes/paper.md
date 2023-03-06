# Paper Reading

略读Diffusion论文，记录亮点和思考，大概需要思考工作是否满足如下几点

Blog：[What Makes a (Graphics) Systems Paper Beautiful](https://graphics.stanford.edu/~kayvonf/notes/systemspaper/)

* 这项工作是基于一组有意义的目标和限制

* 这项工作的核心见解和组织原则

* 这项工作是在设定情境和限制下较为简洁可实现的

* 这项工作提出的观点有助于实现目标（比如通过控制变量，结果有所差异）

* 这项工作对领域和社区提供了新的观点



## List

|                            Paper                             | Application |
| :----------------------------------------------------------: | :---------: |
| Deep Unsupervised Learning using Nonequilibrium Thermodynamics |     ML      |
|                                                              |             |
|                                                              |             |



## ML

Paper: [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

1. Base：概率模型通常受限于可操作性(tractability)和复杂性(flexibility)的权衡，复杂模型难以调整和采样，可操作模型难以拟合复杂的数据分布。

2. Insights：通过蒙特卡洛过程定义了一组概率模型，将可操作性强和复杂性高的概率模型之间建立可操作地过渡，从而实现采样过程（拟合分布后采样）、条件后验（将数据分布和其他信息结合）。
3. Interesting Ideas：
   1. 始终存在Diffusion过程能将单峰的高斯分布过渡到任意平滑的数据分布上
   2. 如何设定分部Diffusion速度 $\beta_t$ ，作者通过在ELBO上梯度上升去调整。

