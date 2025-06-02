---
layout: post
title: "Flow Matching: A Gentle Intro to Flow Matching Models"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a1_flowmatching_logo.gif
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

To reiterate, given a data sample $z_1, \ldots, z_n \sim p_{\text{data}}(\cdot)$, the Flow Matching model aims to construct a probability path $p_t(\cdot)$ such that
$X_0 \sim p_{\text{init}}(\cdot), \ldots, X_t \sim p_t(\cdot), \ldots, X_1 \sim p_{\text{data}}(\cdot)$.
However, directly building this path from the marginal $p_{\text{data}}$ is highly challenging. To address this, Lipman proposed first constructing a Gaussian conditional path $p_t(x \mid z)$,

where the noise scheduler is defined as $\alpha_t = t$ and $\beta_t^2 = 1 - t$, and the data point $z \sim p_{\text{data}}(\cdot)$.

By definition, we see that $p_0(\cdot) = p_{\text{init}}(\cdot) = N(\cdot \mid 0, I)$ and $p_1(\cdot) = \delta_z$. In other words, the conditional probability path $p_t(\cdot \mid z)$ starts from $p_{\text{init}}$ and converges to the data point $z$ as $t \to 1$.

<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a1_gcp.png?raw=true" alt="a1_gcp" width="380"/>
<br>
<em>Gaussian Conditionn path converges to data point z as time t goes to 1</em>
</p>

In addition, we can write
$X_t = \alpha_t z + \beta_t X_0$, where $X_0 \sim N(\cdot \mid 0, I)$.
For $X_t$ to be the solution of $\frac{d}{dt} X_t = u_t(X_t \mid z)$, it is true that:

$$
\begin{align}
u_t( |z) = \left ( \dot \alpha_t -  \cfrac{\dot \beta_t}{\beta_t} \alpha_t \right )z + \cfrac{\dot \beta_t}{\beta_t} x &
\end{align}
$$

where $\dot \alpha_t = \frac{d}{dt} \alpha_t$ and $\dot \beta_t = \frac{d}{dt} \beta_t$. With ths result, if we define

$$
\begin{align}
u^{\text{target}}_t(x) := \int u_t(x | z) \cfrac{p(x | z) p(x)}{p(x)} dz.
\end{align}
$$

The flow solution to $\frac{d}{dt} X_t = u_t^{\text{target}} (X_t)$ can be shown to describe the probabilistic path $X_0, \ldots, X_t, \ldots, X_1$, where $X_1 \sim p_{\text{data}}$. This key result is also known as the continuity equation, which is a special case of the Fokker–Planck equation. The proof can be found in Theorem 1 of Lipman et al., 2023.

## Flow Matching and Conditional Flow Matching Objective Loss

The vector field $u_t^{\text{target}}(x)$ captures everything needed to define the path toward the target distribution. To learn it, we train a neural network denoted by $u_t^{\theta}(x)$. A natural choice for the objective loss function is:

$$
L_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}(0,1),\, x \sim p_t} \left\lVert u_t^{\text{target}}(x) - u_t^{\theta}(x) \right\rVert_2^2
$$

However, this loss is intractable since the form of $p_{\text{data}}(z)$ is unknown. A key result from Flow Matching (Lipman et al., 2023) shows it is proportional to a tractable objective, i.e., $L_{\text{FM}}(\theta) = L_{\text{CFM}}(\theta) + C$, where $C$ is a constant independent of $\theta$, and $L_{\text{CFM}}(\theta)$ is given by:

$$
L_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}(0,1),\, x \sim p_t} \left\lVert u_t^{\text{target}}(x \mid z) - u_t^{\theta}(x) \right\rVert_2^2
$$

where the form of $u_t(x \mid z)$ is shown in the last section.


## Putting it all together into practice

Alright, now that we've covered the foundation, it's time to dive into the implementation.

#### Step 0: specify the target and an inital distribution

```
from gaussian import Gaussian, GaussianMixture
from ultility import plot_comparison_heatmap

# Constants for the duration of our use of Gaussian conditional probability paths, to avoid polluting the namespace...
PARAMS = {
    "scale": 15.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}

p_init = Gaussian.standard(dim=2, std = 1.0).to(device)
p_data = GaussianMixture.symmetric_2D(nmodes=11, std=PARAMS["target_std"], scale=PARAMS["target_scale"]).to(device)
plot_comparison_heatmap(p_init, p_data, PARAMS['scale'])
```
<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a1_target_and_initial_dist.png?raw=true" alt="vectorfieldflow" width="380"/>
<br>
<em>Vector field in black generates flows in red. Image Credit: David Jeffery at UNLV</em>
</p>




