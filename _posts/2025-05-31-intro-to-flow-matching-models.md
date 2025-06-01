---
layout: post
title: "Flow Matching: A Gentle Intro to Flow Matching Models"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: mountains.jpg
---

**Flow Matching** (Lipman et al., 2023) is a powerful class of generative models behind systems like [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) and [Metagen](https://ai.meta.com/research/publications/flow-matching-guide-and-code/), capable of generating highly realistic images and videos. This gentle introduction covers the core ideas behind Flow Matching and walks through a complete example of using Flow Matching models to sample from an 11-component multivariate Gaussian mixture, starting from an initial standard Gaussian distribution. For those interested in practical applications, stay tuned for upcoming posts where we’ll use these same techniques to build a Flow Matching generative model for sampling MNIST images. You can find the code for this post [here](https://github.com/bachvietdo01/generative_models/tree/main/flow_matching).



## It's all about the vector field

The key idea of [Flow Matching](https://arxiv.org/abs/2210.02747) is to construct a probability flow — a path $X_0, \ldots, X_t, \ldots, X_1$ over time index $t \in [0, 1]$ — where  $X_0$ is sampled from an initial distribution $p_{\text{init}}$ and $X_1$ comes from the target distribution $p_{\text{data}}(\cdot)$. This path is a solution to the following ODE (Ordinary Differential Equation), 

$$\begin{align} 
\cfrac{d}{dt} X_t = u_t(X_t) \quad (1)&
\end{align}$$  

where $u_t(x): \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is the vector field and $X_t: [0,1] \to \mathbb{R}^d$ a trajectory for location $x$ at time $t$. Formally, the solution to ODE $(1)$ is called flows, with each flow corresponding to a different initial point. We can also say vector field $u_t(x)$ generates flow $X_t$ as demonstrated in the below figure.


<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/vf_flow.gif?raw=true" alt="vectorfieldflow" width="380"/>
<br>
<em>Vector field in black generates flows in red. Image Credit: David Jeffery at UNLV</em>
</p>

Intuitively, the vector field $u_t(x)$ generates flows, as for a small $h$, the ODE $(1)$ imples that $X_{t+h} = X_t + h \cdot u_t$. If the target data is drawn from $p_{\text{data}}(\cdot)$, there exists a vector field that transports samples from an initial distribution, typically a standard Gaussian $N(0, I)$, to $p_{\text{data}}$. The main goal of Flow Matching is to learn or estimate the target vector field $u_t(x)$ using a neural network.

## Gaussian Probability Path

To reiterate, given a data sample $z_1, \ldots, z_n \sim p_{\text{data}}(\cdot)$, the Flow Matching model aims to construct a probability path $p_t(\cdot)$ such that $X_0 \sim p_{\text{init}}(\cdot), \ldots, X_t \sim p_t(\cdot), \ldots, X_1 \sim p_{\text{data}}(\cdot)$. However, directly building this path from the marginal $p_{\text{data}}$ is highly challenging. To address this, Lipman proposed first constructing a Gaussian conditional path $p_t(x \mid z)$,

where the nosie scheduler $\alpha_t = t$ and $\beta_t^2 = 1 -t$ and data point $z \sim p_{\text{data}}(\cdot)$.

By definition, we see that $p_0(\cdot) = p_{\text{init}}(\cdot) = N(\cdot \mid 0, I)$ and $p_1(\cdot) = \delta_z$. In other words, the conditional probability path $p_t(\cdot | z)$ starts from $p_{\text{init}}$ and converges to the data point $z$ as $t \to 1$.


In addition, we can write $X_t = \alpha_t z + \beta_t X_0$ and $X_0 \sim N(\cdot | 0, I)$. For $X_t$ to be the solution of $\frac{d}{dt} X_t = u_t(X_t | z)$, it is true


