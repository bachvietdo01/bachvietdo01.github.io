---
layout: post
title: "Diffusion Models: Denoising Perspective"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a5_ddpm_butterflies.png
---

Diffusion models have recently emerged as a powerful class of deep generative models, achieving state-of-the-art results in image generation, audio synthesis, protein design, and many scientific domains. Notable works in the literature include [Sohl-Dickstein et al. (2015)](https://arxiv.org/abs/1503.03585), [Song, Ermon et al. (2019)](https://arxiv.org/abs/1907.05600), [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), [Song, Sohl-Dickstein et al. (2020)](https://arxiv.org/abs/2011.13456), and [Karras et al. (2022)](https://arxiv.org/abs/2206.00364). 

This posts gives a gentle introduction to diffusion models through the mainstream viewpoint of denoising diffusion models (DDMs). Over the past five years, DDMs have gained significant popularity, with many available implementations. Instead of building one from scratch, we’ll use Diffusers, a powerful Python library from HuggingFace, to creat a DDPM for generating butterfly images. Readers who want to see an implementation from scratch can refer to this [post](https://bachvietdo01.github.io/intro-diffusion-models).



## Reference

[1] Sohl-Dickstein, Jascha, et al. "Deep unsupervised learning using nonequilibrium thermodynamics." International conference on machine learning. pmlr, 2015.

[2] Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution." Advances in neural information processing systems 32 (2019).

[3] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851.






