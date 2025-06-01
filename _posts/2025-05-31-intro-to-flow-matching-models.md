---
layout: post
title: "Flow Matching: A Gentle Intro to Flow Matching Models"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: mountains.jpg
---

**Flow Matching** (Lipman et al., 2023) is a powerful class of generative models behind systems like [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) and [Metagen](https://ai.meta.com/research/publications/flow-matching-guide-and-code/), capable of generating highly realistic images and videos. This gentle introduction covers the core ideas behind Flow Matching and walks through a complete example of using Flow Matching models to sample from an 11-component multivariate Gaussian mixture, starting from an initial standard Gaussian distribution. For those interested in practical applications, stay tuned for upcoming posts where we’ll use these same techniques to build a Flow Matching generative model for sampling MNIST images.

## It's all about the vector field

The key idea of [Flow Matching](https://arxiv.org/abs/2210.02747) is to construct a probability flow — a path $X_0, \ldots, X_t, \ldots, X_1$ over time index $t \in [0, 1]$ — where  $X_0$ is sampled from an initial distribution $p_{\text{init}}$ and $X_1$ comes from the target distribution $p_{\text{data}}(\cdot)$. This path is a solution to the following ODE (Ordinary Differential Equation), 

$$\begin{align} 
\cfrac{d}{dt} X_t(x) = u_t(x) \\quad (1)&
\end{align}$$  

where $u_t(x)$ is the vector field and $X_t(x)$ a trajectory for location $x$ at time $t$. Formally, the solution to ODE $(1)$ is called flows, with each flow corresponding to a different initial point. We can also say vector field $u_t(x)$ generates flow $X_t(x)$ as demonstrated in the below figure.

![VectorFieldFLow](https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/vf_flow.gif?raw=true)
<br>
*Vector field in black generates flows in red. Image Credit: David Jeffery at UNLV*

Intutively, the vector field $u_t(x)$ generates a flow because for a sufficent small $h$, $X_{t+h} = X_t + h \cdot u_t$. If the target data comes from $p_{\data}(\cdot)$, there exists a vector field $u_t(x)$ that will take "particles" from an intial distribution toward $p_{\data}$. A common choice for initial distribution is standard Gaussian.

