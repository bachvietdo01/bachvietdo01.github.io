---
layout: post
title: "A Gentle Intro to Entropic Optimal Transport: Sinkhorn Algorithm"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a3_sinkhorn_algo.png
---

**Optimal Transport (OT)** has deep roots in mathematics, started with the work of [Monge](https://tinyurl.com/4aa33a2f) then [Kantorovich](https://tinyurl.com/bdeys323). Interest in OT was revived in the 1990s by Yann Brenier then followed in the 2000s by Cédric Villani, who authored two monographs: Topics in Optimal Transportation and Optimal Transport: Old and New, helping to spread knowledge about the applications of OT. More recently, OT has gained traction in machine learning in areas like deep generative modeling, transfer learning, and reinforcement learning. Given its growing importance, this post introduces OT by discussing how to solve the **entropic optimal transport** problem with the **Sinkhorn algorithm**.

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

In the following, we denote $1_m \in R^m$ and $1_n \in R^n$ and $1_{m \times n} \in R^{m \times n}$ are vectors and matrix of only one elements. The optimal condition for for the dual form $(2)$ can be compactly written in matrix format with dot product operator $<\cdot , \cdot>$ and Larange multiplier vectors $\lambda_1 \in R^{m}$ and $\lambda_2 \in R^{n}$,

$$
\begin{aligned}
&\cfrac{d L(P, \lambda_1, \lambda_2)}{dP} = 0 \\
\iff &\cfrac{d L}{dP} \left(  < P, C> - <\lambda_1, P 1_m - a> - <\lambda_2, P^T 1_n - b> + \epsilon < P, \log P > \right)   = 0 \\
\iff & C - \lambda_1 1_n^T - 1_m \lambda_2^T + \epsilon \log P + \epsilon 1_{m \times n} = 0 \\
\iff & P = \exp \left( \cfrac{-C + \lambda_1 \cdot 1^T + 1 \cdot \lambda_2^T  \}{\epsilon} - 1 \right) \\
\iff & P = \exp \left( \cfrac{\lambda_1 \cdot 1^T}{\epsilon} - 1 \right) \cdot \exp \left( \cfrac{-C }{\epsilon}  \right) \cdot \exp \left( \cfrac{1 \cdot \lambda_2^T }{\epsilon}  \right)
\end{aligned}
$$

Here $exp(\cdot)$ and $\log(\cdot)$ means element-wise application to the matrix. Moreover, $\lambda_1 \cdot 1^T$ is a matrix of repeated rows and $1 \cdot \lambda_2^T$ matrix of repated collumns. Thus, we have $\lambda_1 1_n^T = \text{diag}(\lambda_1) \cdot 1_{n \times n}$ and $1_m \lambda_2^T =  1_{m \times m} \cdot \text{diag}(\lambda_2)$ where diag$(\cdot)$ operator creates a diagional matrix with the vector argument. 

As such, define vectors $u:= \exp \left ( \cfrac{\lambda_{1, i}}{\epsilon} \right )_i$, $v := \exp \left ( \cfrac{\lambda_{2, i}}{\epsilon} \right )_j$ and matrix $K =  \exp \left ( \cfrac{-C }{\epsilon}  \right )$. The optimality condition implies,

$$
\begin{aligned}
P^\ast = \text{diag}(u) \cdot K \cdot \text{diag}(v)
\end{aligned}
$$

## From Stochastic to Deterministic Matching

<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps100.png?raw=true" alt="eps100" width="190"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps020.png?raw=true" alt="eps020" width="200"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps005.png?raw=true" alt="eps005" width="185"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps000.png?raw=true" alt="eps000" width="200"/>
<br>
<em>Optimal solutions of moving from blue to red distributions for different epsilons. Image Credit: G. Peyre’s twitter account</em>
</p>


## Reference

[1] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." Advances in neural information processing systems 26 (2013)






