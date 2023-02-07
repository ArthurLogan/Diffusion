# Diffusion Model

学习笔记



## Basic Idea

1. 图像分布：图像可以理解成高维空间下的分布。如64\*64的图像等价于4096维空间下的一个点。所有有意义的图像仅对应高维空间的一个子集，大部分空间点转换回图像仍是无意义噪声。因此数据集可以看成高维空间下的数据分布，有语义的点的概率高，无语义的点概率低。很多图像分类都基于一个假设，即语义信息相近的图像在高维空间下(通常是transform之后的空间)的距离相近，因此通过训练高维空间下的分类器可以一定程度解决图像分类问题。
2. 优化目标：在图像生成中，模型的输出对应高维空间的点，对应图像分布的概率值，我们希望调整输出，使得对应真实图像的概率值更大，等价于最小化概率值的负对数，即 $-\log p(x)$ 。
3. 变分下界：由于我们无法收集所有数据集，因此准确计算图像分布下的概率密度。因此大多数情况通过优化变分下界ELBO来逼近对概率优化，有 $-\log p(x)\le \mathcal{L}_{elb}$ 。



## ELBO

VAE文章指出，可以将概率密度的负对数，转换为KL散度和期望之和。假设真实分布由参数 $\theta$ 控制，模型参数由 $\phi$ 控制，则概率负对数表示如下。

$$
\begin{align*}
\log p_{\theta}(x)&=\mathbb{E}_{q_{\phi}(z|x)}\left[\log p_{\theta}(x,z)-\log\frac{p_{\theta}(x,z)}{p_{\theta}(x)}\right]\\
&=\mathbb{E}_{q_{\phi}(z|x)}\left[\log\frac{q_{\phi}(z|x)}{p_{\theta}(z|x)}\right]+\mathbb{E}_{q_{\phi}(z|x)}\left[-\log q_{\phi}(z|x)+\log p_{\theta}(x,z)\right]
\end{align*}
$$

前者为模型分布和真实分布的KL散度，后者为ELBO，并且ELBO可以进一步转换成编码解码过程和正则项。

$$
\begin{align*}
\text{ELBO}=-\mathbb{E}_{q_{\phi}(z|x)}\left[\log\frac{q_{\phi}(z|x)}{p_{\theta}(z)}\right]+\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]
\end{align*}
$$

前者为正则项，最大化ELBO相当于最小化后验分布和先验分布的KL散度，使得后验分布具有一定随机性，不完全受控于图像点。后者为重建损失的负值，即从从图像推测隐变量，再从隐变量重建图像。

最终优化目标如下。

$$
\max\mathbb{E}_D\mathbb{E}_{q_{\phi}(z|x)}\left[-\log q_{\phi}(z|x)+\log p_{\theta}(x,z)\right]
$$

通过假设部分过程为高斯噪声，能将优化简化成对高斯均值和方差的估计。



## DDPM

VAE假设隐变量服从高斯分布，训练过程编码器输出高斯均值和方差，再通过重参数技巧采样，经过解码器还原成输入图像。这个过程涉及到编码器和解码器两个网络，由于参数量很大，训练过程(即参数搜索过程)很容易产生类似于GAN的不稳定情况。

Diffusion Model简化这一步骤，将编码器转变为确定参数过程，即在图像上不断增加噪声，最终使图像趋于高斯噪声，只有解码过程涉及可训练参数，使解码器输出尽量趋近输入图像。将不断增加噪声的过程看作马尔可夫链，增加噪声的过程为前向过程(记作 $q$ )，减少噪声的过程为逆向过程(记作 $p_{\theta}$ )。我们希望优化逆向过程生成图像的真实度，即极小化负对数概率，同样存在ELBO。

$$
\mathbb{E}[-\log p_{\theta}(x_0)]\le\mathbb{E}_q\left[-\log\frac{p_{\theta}(x_{0:T})}{q(x_{1:T}|x_0)}\right]=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t\ge 1}\log\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]=L
$$

$q(x_0)$ 为数据分布，根据构造 $q(x_t|x_{t-1})$ 条件分布得到 $x_{0:T}$ 的联合分布。在不等式左侧仅对 $x_0$ 求积分，由ELBO放缩过程包括条件期望，综合得到以图像分布为概率密度的联合分布期望。

**前向过程**：以当前像素值为基础构造前向高斯过程，具体形式如下式。

$$
q(x_{t}|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\text{I})
$$

前向过程不断增加噪声，因此方差系数 $\beta_t$ 随时间逐渐增大，最初接近0，最终趋于1。由高斯分布的边缘分布和条件分布仍然是高斯分布，因此将 $x_1$ 积分，通过重参数技巧表示，令 $\alpha_t=1-\beta_t$， $\bar{\alpha}_t=\prod_{s=1}^t\alpha_s$ 。

$$
x_2=\sqrt{\alpha_2}x_1+\sqrt{\beta_2}\epsilon_2=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2\beta_1}\epsilon_1+\sqrt{\beta_2}\epsilon_2
$$

$$
\mathbb{E}[x_2]=\sqrt{\bar{\alpha}_2}x_0,\text{Var}[x_2]=\alpha_2\beta_1+\beta_2=1-\bar{\alpha}_2
$$

递推得到第t步的条件概率为：

$$
q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\text{I})
$$

由于方差系数均在0-1之间，当时间趋于无穷时，条件概率会趋近于标准正太分布，且和输入无关，符合我们所希望随着噪声添加，图像趋于高斯噪声。

**逆向过程**：希望用高斯分布近似每一步的前向过程，但因为数据分布始终不可知，我们只知道以 $x_0$ 为条件的分布，此处便是需要用神经网络拟合的地方。

$$
q(x_{t-1}|x_t,x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0),\tilde{\beta}_t\text{I})
$$

$$
\tilde{\mu}_t(x_t,x_0)=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0+\frac{\sqrt{\alpha}_t(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t,\ \ \tilde{\beta}=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

说明对于不同输入图像，逆向过程的高斯均值不相同，但在生成采样的过程中，并不知道输入图像。DDPM假设去除 $x_0$ 条件后仍然为高斯分布，因此在逆向过程中同样使用高斯分布估计近似。

$$
p_{\theta}(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

**优化目标**：得到ELBO表达式后可以进一步调整转换为：

$$
\begin{align*}
L&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t\ge 1}\log\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\\
&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t>1}\log\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\cdot\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}-\log\frac{p_{\theta}(x_0|x_1)}{q(x_1|x_0)}\right]\\
&=\mathbb{E}_q\left[-\log\frac{p(x_T)}{q(x_T|x_0)}-\sum_{t>1}\log\frac{p_{\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}-\log p_{\theta}(x_0|x_1)\right]\\
&=\mathbb{E}_q\left[D_{KL}(q(x_T|x_0)||p(x_T))+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))-\log p_{\theta}(x_0|x_1)\right]
\end{align*}
$$

由于前向过程中我们固定扰动方差，前向过程不具有可学习参数，而逆向过程的初始采样分布即标准高斯分布，因此第一项在训练中固定。第二项高维高斯分布的KL散度可以具体表示参考 [Normal-KL](https://www.cnblogs.com/qizhou/p/13804283.html) ，DDPM论文假设逆向过程的方差固定和前向过程的条件方差相同，进一步简化只需要训练期望，则散度简化成下式。

$$
L_{t-1}=\mathbb{E}_q\left[\frac{1}{2\sigma_t^2}||\tilde{\mu}_t(x_t,x_0)-\mu_\theta(x_t,t)||^2\right]+C
$$

通过从数据库采样得到 $x_0$ ，利用重参数技巧生成 $x_t$ 并代入公式得到 $\tilde{\mu}_t$ ，通过神经网络以 $x_t,t$ 为输入预测当前时刻的均值，从而计算得到损失。通过重参数技巧转换积分变量得到下式。

$$
L_{t-1}-C=\mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma_t^2}\left|\left|\frac{1}{\sqrt{\alpha_t}}\left(x_t(x_0,\epsilon)-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)-\mu_\theta(x_t(x_0,\epsilon))\right|\right|^2\right]
$$

以类似形式构建预测均值，将$x_t$部分消除，则从均值损失转换成随机误差的损失。

$$
L_{t-1}-C=\mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}||\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,t)||^2\right]
$$

最后一项将$(-\infty,\infty)$范围归一化到概率，通过划定正态分布的不同区域，公式如下。

$$
p_\theta(x_0|x_1)=\prod_{i=1}^D\int_{\delta_-}^{\delta_+}\mathcal{N}(x;\mu_\theta(x_1,1),\sigma_1^2)dx
$$

$$
\delta_+=\left\{\begin{matrix}
\infty & \text{if}\ x=1\\ 
x+\frac{1}{255} & \text{if}\ x<1
\end{matrix}\right.,\ \
\delta_-=\left\{\begin{matrix}
-\infty & \text{if}\ x=-1\\ 
x-\frac{1}{255} & \text{if}\ x>-1
\end{matrix}\right.
$$

在实际训练中，通常忽略前置系数，将优化目标转化为下式。

$$
L_{\text{simple}}(\theta)=\mathbb{E}_{t,x_0,\epsilon}\left[||\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon, t)||^2\right]
$$

## DDIM

DDPM基于马尔可夫过程构建前向过程，不断向图像增加噪声，直至图像变成高斯噪声。但最后构建的目标函数及训练过程和某一步前向过程无关，仅依赖从 $x_0\rightarrow x_t$ 的边缘过程。能否保证边缘概率相同的基础上，修改前向过程，从而提高逆向过程采样效率。

**前向过程重组**：在DDPM中，作者假设前向过程为马尔可夫过程，此时可以将联合分布拆解成逆向过程乘积。

$$
q_{\sigma}(x_{1:T}|x_0)=q_\sigma(x_T|x_0)\prod_{t=2}^Tq_\sigma(x_{t-1}|x_t,x_0)
$$

其中保证从 $x_0\rightarrow x_T$ 的边缘概率为：

$$
q_\sigma(x_T|x_0)=\mathcal{N}(\sqrt{\alpha_T}x_0,(1-\alpha_T)I)
$$

通过构造逆向过程形式如下，可以保证中间时刻的边缘概率形式和DDPM相同。

$$
q_\sigma(x_{t-1}|x_t,x_0)=\mathcal{N}\left(\sqrt{\alpha_{t-1}}x_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\alpha_t}x_0}{\sqrt{1-\alpha_t}},\sigma_t^2I\right)
$$

此时单步前向过程形式如下，此时前向过程仍为高斯过程，但不再是马尔可夫过程。

$$
q_\sigma(x_t|x_{t-1},x_0)=\frac{q_\sigma(x_{t-1}|x_t,x_0)q_\sigma(x_t|x_0)}{q_\sigma(x_{t-1}|x_0)}
$$

**逆向过程**：在DDPM中，作者固定方差，只通过学习均值使得网络逼近逆向过程概率。在DDIM中，由于不要求前向过程为马尔可夫过程，因此逆向过程的形式丰富了很多，并引入可调节方差参数 $\sigma$ 。在逆向采样时，首先通过 $x_t$ 估计 $x_0$ 。
$$
f_\theta^{(t)}(x_t)=\frac{x_t-\sqrt{1-\alpha_t}\cdot \epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}
$$

再代入理论的逆向过程公式 $q_\sigma(x_{t-1}|x_t,x_0)$ 得到单步逆向过程如下。

$$
p_\theta^{(t)}(x_{t-1}|x_t)=\left\{\begin{matrix}
\mathcal{N}(f_\theta^{(1)}(x_1),\sigma_1^2I) & \text{if}\ t=1\\ 
q_\sigma(x_{t-1}|x_t,f_\theta^{(t)}(x_t))& \text{otherwise}
\end{matrix}\right.
$$

**优化目标**：根据ELBO推导进一步转化成非马尔可夫链的前向过程形式。

$$
\begin{align*}
J&=\mathbb{E}_q[\log q_\sigma(x_{1:T}|x_0)-\log p_\theta(x_{0:T})]\\
&=\mathbb{E}_q\left[\log q_\sigma(x_T|x_0)+\sum_{t=2}^T\log q_\sigma(x_{t-1}|x_t,x_0)-\sum_{t=1}^T\log p_\theta^{(t)}(x_{t-1}|x_t)-\log p_\theta(x_T)\right]
\end{align*}
$$

可以证明该优化目标和DDPM的优化目标相差常数，一组 $\sigma$ 参数对应一组 $\gamma$ 参数。如果误差函数 $\epsilon_{\theta}$ 在时间维度不共享，则DDPM中的优化目标等价于 $L_{0:T}$ 每项的最优和，在该条件下如下式子成立。

$$
J_\sigma\equiv L_\gamma \equiv L_1
$$

以 $L_1$ 为优化目标的模型同样学习了以非马尔可夫前向过程属性，因此可以通过修改 $\sigma$ 去优化前向过程。 

**采样过程**：和DDPM相同，作者通过当前 $x_t$ 预测 $x_0$ 并代入 $q_\sigma$ 公式，通过重参数计算 $x_{t-1}$ 。

$$
x_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}\right)+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot \epsilon_\theta^{(t)}(x_t)+\sigma_t\epsilon_t
$$

通过调整 $\sigma$ 可以得到不同的前向过程和逆向过程，当 $\sigma=\sqrt{(1-\alpha_{t-1})/(1-\alpha_t)}\sqrt{1-\alpha_t/\alpha_{t-1}}$ 时，前向过程退化为马尔可夫链即DDPM，对应采样过程如下。

$$
x_{t-1}=\sqrt{\frac{\alpha_{t-1}}{\alpha_t}}x_t+
$$

另一种特殊情况，当 $\sigma=0$ 时，采样过程退化成确定过程，此时采样过程如下，这种逆向过程相当于隐式生成模型，因而称为DDIM。

$$
x_{t-1}=\sqrt{\frac{\alpha_{t-1}}{\alpha_t}}x_t+\sqrt{\alpha_{t-1}}\left(\sqrt{\frac{1-\alpha_{t-1}}{\alpha_{t-1}}}-\sqrt{\frac{1-\alpha_t}{\alpha_t}}\right)\epsilon_\theta^{(t)}(x_t)
$$
