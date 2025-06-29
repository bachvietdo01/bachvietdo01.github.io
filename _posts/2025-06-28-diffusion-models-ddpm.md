---
layout: post
title: "Diffusion Models: Denoising Perspective"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a1_flowmatching_logo.gif
---

Diffusion models have recently emerged as a powerful class of deep generative models, achieving state-of-the-art results in image generation, audio synthesis, protein design, and many scientific domains. Notable works in the literature include [Sohl-Dickstein et al. (2015)](https://arxiv.org/abs/1503.03585), [Song, Ermon et al. (2019)](https://arxiv.org/abs/1907.05600), [Ho et al. (2020)](https://arxiv.org/abs/2006.11239), [Song, Sohl-Dickstein et al. (2020)](https://arxiv.org/abs/2011.13456), and [Karras et al. (2022)](https://arxiv.org/abs/2206.00364). 

This posts gives a gentle introduction to diffusion models through the mainstream viewpoint of denoising diffusion models (DDMs). Over the past five years, DDMs have gained significant popularity, with many available implementations. Instead of building one from scratch, we’ll use Diffusers, a powerful Python library from HuggingFace, to creat a DDPM for generating butterfly images. Readers who want to see an implementation from scratch can refer to this [post](https://bachvietdo01.github.io/intro-diffusion-models).



## Reference

[1] Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

[2] Holderrieth, Peter, and Ezra Erives. "An Introduction to Flow Matching and Diffusion Models." arXiv preprint arXiv:2506.02070 (2025).




