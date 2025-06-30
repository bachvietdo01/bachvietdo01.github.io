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

* **Forward process** constructs a Markov chain $x_0, x_1, \ldots, x_T$ using Gaussian transitions: $q(x_t \mid x_{t-1}) = \mathcal{N}(\cdot \mid \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)$.

* **Backward process** learns to reverse this chain through another Gaussian Markov process $p(x_{t-1} \mid x_t) = \mathcal{N}(\cdot \mid \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$, where the mean and variance are learnable with parameter $\theta$.


## Reference

[1] Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." International conference on machine learning. pmlr, 2015.

[2] Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution." Advances in neural information processing systems 32 (2019).

[3] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.






