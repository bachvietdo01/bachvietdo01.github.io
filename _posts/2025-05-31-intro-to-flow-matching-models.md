---
layout: post
title: "Flow Matching: A Gentle Intro to Flow Matching Models"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: mountains.jpg
---

**Flow Matching** is a powerful class of generative models behind systems like Stable [Diffusion 3](https://stability.ai/news/stable-diffusion-3) and [Metagen](https://ai.meta.com/research/publications/flow-matching-guide-and-code/), capable of generating highly realistic images and videos. This gentle introduction covers the core ideas behind Flow Matching and walks through a complete example of using Flow Matching models to sample from an 11-component multivariate Gaussian mixture, starting from an initial standard Gaussian distribution. For those interested in practical applications, stay tuned for upcoming posts where we’ll use these same techniques to build a Flow Matching generative model for sampling MNIST images.

## It's all about the vector field

The key idea is to construct a probability flow — a path $X_0, \ldots, X_t, \ldots, X_1$ over time index $t \in [0, 1]$ — where  $X_0$ is sampled from an initial distribution $p_{\text{init}}$ and $X_1$ comes from the target distribution $p_{\text{data}}(\cdot)$.  This path is a solution to the following ODE (Ordinary Differential Equation), 

$$\begin{align} 
\cfrac{d}{dt} X_t(x) = u_t(x)  \label{eqn:ode}\tag{1}  &
\end{align}$$  

where $u_t(x)$ is the vector field for location $x$ at time $t$. Formally, the solution to ODE $\ref{eq:ode}$ is called flows, with each flow corresponding to a different initial point.

![VectorFieldFLow](https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/vf_flow.gif?raw=true)
<br>
*Vector field in black generates flows in red. Image Credit: David Jeffery at UNLV*

