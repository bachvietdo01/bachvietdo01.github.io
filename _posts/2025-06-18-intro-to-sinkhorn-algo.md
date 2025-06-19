---
layout: post
title: "A Gentle Intro to Entropic Optimal Transport: Sinkhorn Algorithm"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a3_sinkhorn_algo.png
---

**Optimal Transport (OT)** has deep roots in mathematics, started with the work of [Monge](https://tinyurl.com/4aa33a2f) then [Kantorovich](https://tinyurl.com/bdeys323). Interest in OT was revived in the 1990s by Yann Brenier then followed in the 2000s by Cédric Villani, who authored two monographs: Topics in Optimal Transportation and Optimal Transport: Old and New, helping spread knowledge about the applications of OT. More recently, OT has gained traction in machine learning in areas like deep generative modeling, transfer learning, and reinforcement learning. Given its growing importance, this post introduces OT by discussing how to solve the **entropic optimal transport** problem with the **Sinkhorn algorithm**.

## Kantorovich Relaxation Problem and Entropic Regularization

Let vector $a = (a_i)$ for $i = 1, \ldots, m$ and vector $b = (b_j)$ for $j = 1, 2, \ldots, m$ are (finite) discrete distributions, i.e., $\sum_{i=1}^m a_i = 1 = \sum_{j=1}^m b_j$. We can think of $a$ and $b$ as histograms, e.g., the value at bin $i$ is the probability of item $i$. 

Let matrix $C(i,j) \in R^{m \times n}$ be the cost of moving an "atom" from bin $i$ of $a$ to bin $j$ of $b$, and $P(i, j) \in R^{m \times n}$ the **coupling** matrix, describing the amount of mass transporting from bin $i$ of a to bin $j$ of b. The Kantorvich problem is the following optimization problem,

$$
\begin{aligned}
&\min_{P} \sum_{i,j} P_{i,j} \cdot C_{i,j} \quad (1) \\
&\text{st. } \sum_i P_{i,j} = a, \sum_j P_{i,j} = b
\end{aligned}
$$

It can be shown $(1)$ admits a unique solution. Moreover, this is a linear programming problem, and the typical solvers such has network simplex or interior point have cubic time complexity $O(n^3)$. These approaches become computationally expensive when the discrete distributions have many support bins. Cuturi (2013) proposed a computionally efficient algorithm by introducing an entropic regularization term to the optimization,

$$
\begin{aligned}
&\min_{P} \sum_{i,j} P_{i,j} \cdot C_{i,j} - \epsilon H(P) \quad (2) \\
&\text{st. } \sum_i P_{i,j} = a, \sum_j P_{i,j} = b
\end{aligned}
$$

where the entropy as $H(P) = -\sum_{i,j} P_{i,j} \log P_{i,j} \propto -\text{KL}(P, a \otimes b)$, KL denotes the Kullback–Leibler divergence and $a \otimes b$ is the product (independent) distribution with marginals $a$ and $b$. Intuitively, $P_{i,j}$ represents possible joint distributions matching marginals $a$ and $b$, and the entropic regularization in (2) penalizes large deviations from the independent joint. 

Under suitable conditions, (2) admits an unique solution $P_{\epsilon}^\ast$. Let $P^\ast$ denote the unique solution to the original Kantorovich problem (1). Remarkably, $P_\epsilon^* \to P^*$ as $\epsilon \to 0$ ([see Nutz, 2022](https://www.math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf)).

## Sinkhorn Algorithm

For discrete finite measures, the derivation is surprisingly straightforward, involving only basic calculus. Let $1_m \in R^m, 1_n \in R^n$ are vectors of only one elements.

Moreover, $1_{m \times n} \in \mathbb{R}^{m \times n}$ is a matrix of ones.  The optimality conditions for the dual form of (2) can be compactly expressed in matrix form using the dot product $\langle \cdot , \cdot \rangle$ and Lagrange multiplier vectors $\lambda_1 \in \mathbb{R}^m$ and $\lambda_2 \in \mathbb{R}^n$.

$$
\begin{aligned}
&\frac{d L(P, \lambda_1, \lambda_2)}{dP} = 0 \\
\iff\ &\frac{dL}{dP} \left( \langle P, C \rangle - \langle \lambda_1, P 1_m - a \rangle - \langle \lambda_2, P^T 1_n - b \rangle + \epsilon \langle P, \log P \rangle \right) = 0 \\
\iff\ & C - \lambda_1 1_m^T - 1_n \lambda_2^T + \epsilon \log P + \epsilon 1_{m \times n} = 0 \\
\iff\ & P = \exp \left( \frac{-C + \lambda_1 \cdot 1^T + 1 \cdot \lambda_2^T}{\epsilon} - 1 \right) \\
\iff\ & P = \exp \left( \frac{\lambda_1 \cdot 1^T}{\epsilon} - 1 \right) \cdot \exp \left( \frac{-C}{\epsilon} \right) \cdot \exp \left( \frac{1 \cdot \lambda_2^T}{\epsilon} \right)
\end{aligned}
$$

Refer to [Petersen, 2012](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for common rules of matrix derivatives. Here, $\exp(\cdot)$ and $\log(\cdot)$ are applied element-wise to matrices. The term $\lambda_1 \cdot 1^T$ produces a matrix with repeated rows of $\lambda_1$, $1 \cdot \lambda_2^T$ repeated columns of $\lambda_2$. 

As such, this gives  $\lambda_1 1_n^T = \operatorname{diag}(\lambda_1) \cdot \mathbf{1}_{n \times n} \quad \text{and} \quad 1_m \lambda_2^T = \mathbf{1}_{m \times m} \cdot \operatorname{diag}(\lambda_2)$

where $\operatorname{diag}(\cdot)$ creates a diagonal matrix from a vector, and $\mathbf{1}_{p \times q}$ denotes a $p \times q$ matrix of ones.


Define vectors $u := \exp\left(\frac{\lambda_1}{\epsilon}\right)$ and $v := \exp\left(\frac{\lambda_2}{\epsilon}\right)$, and let $K := \exp\left(-\frac{C}{\epsilon}\right)$. The optimality conditions then imply:

$$
\begin{aligned}
&& P^\ast = \text{diag}(u) \cdot K \cdot \text{diag}(v) \\
&& \text{diag}(u) \cdot K \cdot \text{diag}(v) \cdot 1_m = a \\
&& \text{diag}(v) \cdot K^T \cdot \text{diag}(u) \cdot 1_n = b \\
\end{aligned}
$$

Equivalently,

$$
\begin{aligned}
&& u \odot Kv = a \\
&& v \odot (K^T u) = b
\end{aligned}
$$

As $K$, the function of cost, is a constant, we can construct an iterative algorithm as follows,

$$
\begin{aligned}
u^{t+1} &= a / Kv^{t} \\
v^{t+1} &= b / (K^T u^{t+1})  
\end{aligned}
$$

where the operator $/$ denotes element-wise division. This procedure is formally known as the Sinkhorn algorithm (Cuturi, 2013).

# Python Implementation

Using the iterative equations, the Python implementation is straightforward,

```
import numpy as np

def sinkhorn_algorithm(a, b, C, eps, max_iter=1000, threshold=1e-9):
    # Normalize cost matrix for numerical stability
    C = C / C.max()
    # Compute the Gibbs kernel
    K = np.exp(-C / eps)

    # reshape a, b into column vec format
    a = a.reshape((C.shape[0], 1))
    b = b.reshape((C.shape[1], 1))

    # Initialize scaling factors
    u = np.ones_like(a)
    v = np.ones_like(b)

    # Sinkhorn iterations
    for _ in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()

        # Update v
        v = b / (K.T @ u)
        # Update u
        u = a / (K @ v)

        # Check for convergence
        if np.linalg.norm(u - u_prev) < threshold and np.linalg.norm(v - v_prev) < threshold:
            break

    # Compute the optimal transport plan
    P_star = np.diag(u.flatten()) @ K @ np.diag(v.flatten())

    return P_star
```

## From Stochastic to Deterministic Matching

<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps100.png?raw=true" alt="eps100" width="190"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps020.png?raw=true" alt="eps020" width="200"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps005.png?raw=true" alt="eps005" width="185"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps000.png?raw=true" alt="eps000" width="200"/>
<br>
<em>Optimal solutions of moving mass from blue to red distributions for different epsilons. Image Credit: G. Peyre’s twitter account</em>
</p>


## Reference

[1] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." Advances in neural information processing systems 26 (2013)






