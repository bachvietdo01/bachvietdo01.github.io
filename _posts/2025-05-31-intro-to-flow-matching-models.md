---
layout: post
title: "Flow Matching: A Gentle Intro to Flow Matching Models"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a1_flowmatching_logo.gif
---

**Flow Matching** (Lipman et al., 2023) is a powerful class of generative models behind systems like [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3) and [Metagen](https://ai.meta.com/research/publications/flow-matching-guide-and-code/), capable of generating highly realistic images and videos. This gentle introduction covers the core ideas behind Flow Matching and walks through a complete example of using Flow Matching models to sample from an 11-component multivariate Gaussian mixture, starting from an initial standard Gaussian distribution. 

You can find the code for this post [here](https://github.com/bachvietdo01/generative_models/tree/main/flow_matching). For those interested in practical applications, stay tuned for upcoming posts where we’ll use these same techniques to build a Flow Matching generative model for sampling MNIST images. 


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

$$
\begin{align}
p(x \mid z) = N(\cdot \mid \alpha_t z , \beta_t^2 \cdot I)
\end{align}
$$

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

### Step 0: specify the target and an inital distribution

The initial distribution is a standard Gaussian distribution.

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
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a1_target_and_initial_dist.png?raw=true" alt="vectorfieldflow" width="1000"/>
</p>


### Step 1: build the Gaussin Conditional Probability Path

$\alpha_t = t$ and $\beta_t = \sqrt{1-t}$, and so  $\dot \alpha_t = 1$ and $\dot \beta_t = -\cfrac{1}{2\sqrt{1-t}}$.

```
from gaussian import Sampleable

class StandardNormal(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """
    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples) -> torch.Tensor:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device)
```

```
class LinearAlpha:
    """
    Implements alpha_t = t
    """
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        return torch.ones_like(t)
        

class SquareRootBeta:
    """
    Implements beta_t = rt(1-t)
    """
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """
        return torch.sqrt(1 - t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)
```

```
class GaussianConditionalProbabilityPath(nn.Module):
    def __init__(self, p_data: Sampleable, alpha: LinearAlpha, beta: SquareRootBeta):
        super().__init__()
        p_init = StandardNormal(shape = [p_data.dim], std = 1.0)
        self.p_init = p_init
        self.p_data = p_data
        
        self.alpha = alpha
        self.beta = beta

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, c, h, w)
        return x

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        return self.p_data.sample(num_samples)

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)

    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2
```

```
# Construct conditional probability path
path = GaussianConditionalProbabilityPath(
    p_data = p_data,
    alpha = LinearAlpha(),
    beta = SquareRootBeta()
).to(device)
```

### Step 2: learn the vector field with an MLP neural net

In the following, we choose $u_t^{\theta}(x)$ to be an MLP and caclulate the Conditional Flow Matching loss $L_{\text{CFM}}(\theta)$.

```
class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def get_mlp(self, dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = self.get_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)
```

```
from trainer import Trainer

class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
      z = self.path.p_data.sample(batch_size)
      t = torch.rand(batch_size, 1)
      x = self.path.sample_conditional_path(z, t)
      u_theta = self.model(x, t)
      u_ref = self.path.conditional_vector_field(x, z, t)

      return torch.mean((u_theta - u_ref)**2)
```

```
# Construct learnable vector field
flow_model = MLPVectorField(dim=2, hiddens=[1024,16])

# Construct trainer
trainer = ConditionalFlowMatchingTrainer(path, flow_model)
losses = trainer.train(num_epochs=5000, device=device, lr=1e-3, batch_size=1000)
```


### Step 3: Generate samples from the learned model

Given the estimated $u^{\hat{\theta}}(x)$, our goal is to sample (or generate) data from the learned vector field. As shown below, the red dots represent the desired samples drawn from a Gaussian mixture with 11 components.


```
from ode import LearnedVectorFieldODE, EulerSimulator

num_samples = 1000
num_timesteps = 300
num_marginals = 3

ode = LearnedVectorFieldODE(flow_model)
simulator = EulerSimulator(ode)
x0 = path.p_init.sample(num_samples) # (num_samples, 2)
ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
xts = simulator.simulate_with_trajectory(x0, ts) # (bs, 
```

```
from importlib import reload
from ultility import plot_generated_sample

plot_generated_sample(xts, ts, p_init, p_data, scale = PARAMS['scale'], num_samples=num_samples, num_timesteps=num_timesteps, num_marginals=num_marginals)
```
<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a1_sampled_gm.png?raw=true" alt="vectorfieldflow" width="1000"/>
</p>

## Reference



