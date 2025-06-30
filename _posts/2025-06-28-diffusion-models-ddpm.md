---
layout: post
title: "Diffusion Models: Denoising Perspective"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a5_ddpm_butterflies.png
---

Diffusion models are a powerful class of deep generative models, breaking the long-standing dominance of generative adversarial networks (GANs). Key developments in the literature include works such as [Sohl-Dickstein et al. (2015)](https://arxiv.org/abs/1503.03585), [Song and Ermon (2019)](https://arxiv.org/abs/1907.05600), [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), [Song and Sohl-Dickstein et al. (2020)](https://arxiv.org/abs/2011.13456), and [Karras et al. (2022)](https://arxiv.org/abs/2206.00364).

Previously, I explored Diffusion Models through the lens of [Stochastic Flow Matching](https://bachvietdo01.github.io/intro-diffusion-models), controlled by stochastic differential equations (SDEs). THis posts provides a different and the mainstream perspective of denoising diffusion models (DDMs). Over the past five years, DDMs have grown immensely popular, with many open-source implementations available. Rather than building one from scratch, we'll use Diffusers, a powerful Python library from HuggingFace, to build a generative model which can generate butterfly images. For those interested in a from-scratch implementation, see the [post](https://bachvietdo01.github.io/intro-diffusion-models).


# Denoising with Markov Chain

The core idea behind diffusion models is construct a transport path from a noise distribution to a target data distribution. [Ho et al. (2020)](https://arxiv.org/abs/2006.11239) proposed a framework to achieve this with a forward and a backward process. Starting from a data point $x_0 \sim p_{\text{data}}(x)$,

* **Forward process** constructs a Markov chain $x_0, x_1, \ldots, x_T$ using Gaussian transitions $`q(x_t \mid x_{t-1}) = \mathcal{N}(\cdot \mid \sqrt{1 - \beta_t} x_{t-1} \; , \; \beta_t I)`$ for $0 < \beta_t < 1$.

* **Backward process** learns to reverse the forward chain through another Gaussian-transition Markov chain $p(x_{t-1} \mid x_t) = \mathcal{N}(\cdot \mid \mu_\theta(x_t, t) , \Sigma_\theta(x_t, t))$ where the learnable mean and variance are parameterized by $\theta$.

The backward process is the path we want to construct taking a noise sample to a target data sample. To accurately reverse the forward process, we learn $\theta^*$ such that,

$$
\begin{align}
\theta^* = \text{arg } min_{\theta} \text{ KL}(q(x_0, x_1, \ldots, x_T), p_{\theta}(x_0, x_1, \ldots, x_T)) \quad (2)
\end{align}
$$

Note that the model $p_{\theta}$ is in the second argument of the KL, known as the forward KL, which differs from the typical reverse KL used in Variational Inference. Nevertheless, we can use Jensen Inequality to show that $(2)$ is lower bounded by log negavitve log likelihood under the model $p_{\theta}$, i.e.,

$$
\begin{align}
\text{ KL}(q(x_0, x_1, \ldots, x_T), p_{\theta}(x_0, x_1, \ldots, x_T)) \ge E(-\log p_{\theta}(x_0)) + \text{const-free-of-}\theta
\end{align}
$$

Thus minizing the KL will push down the negative log likelihood or equivalently increase the log likelihood. Regarding the **Forward Process**, it is straightforward to derive the following using the fact that a sum of two Gaussians is also Gaussian,


$$
\begin{align}
x_t \mid x_0 = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
\end{align}
$$

where $\epsilon \sim \text{N}(\cdot \mid 0, I)$ and $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_i \alpha_i$. 

Because $1 - \beta_t \in (0,1)$, $\bar{\alpha}}_t \to 0$ as $t \to \infty$. Therefore $x_t \to N(\cdot | 0, I)$ as $t \to \infty$. In other words, the forward process progressively diffuses data into Standard Gaussian noise. 

Now, we can expand and rewrite $(2)$ as follows,

$$
\begin{align}
(2) = L_T + \sum_{t=1}^{T-1} L_i + L_0
\end{align}
$$

where $L_T = \text{KL}(q(x_T \mid x_0), p_{\theta}(x_T))$, $L_t = \text{KL}(q(x_t \mid x_{t+1}, x_0), p_{\theta}(x_t \mid x_{t+1}))$ and $L_0 = -\log p_{\theta}(x_0 \mid x_1)$. Moreover, by using conjacy of Gaussian, we can derive that,

$$
\begin{align}
q(x_t \mid x_{t+1}, x_0) = \text{N}(\cdot | \tilde{\mu}_{t,0} , \tilde{\beta}_t I )
\end{align}
$$


## Reference

[1] Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." International conference on machine learning. pmlr, 2015.

[2] Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution." Advances in neural information processing systems 32 (2019).

[3] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.






