---
layout: post
title: "A Gentle Intro to Entropic Optimal Transport: Sinkhorn Algorithm"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a3_sinkhorn_algo.png
---

**Optimal Transport (OT)** has deep roots in mathematics, started with the work of [Monge](https://tinyurl.com/4aa33a2f) then [Kantorovich](https://tinyurl.com/bdeys323). Interest in OT was revived in the 1990s by Yann Brenier and the followed in the 2000s by Cédric Villani, who authored two monographs: Topics in Optimal Transportation and Optimal Transport: Old and New, helping to spread knowledge about the applications of OT. More recently, OT has gained traction in machine learning in areas like deep generative modeling, transfer learning, and reinforcement learning. Given its growing importance, this post introduces OT by discussing how to solve the **entropic optimal transport** problem with the **Sinkhorn algorithm**.

## Kantorovich Relaxation Problem and Entropic Regularization

Let $a = (a_i)$ for $i = 1, \ldots, m$ and $b = (b_j)$ for $j = 1, 2, \ldots, m$ are (finite) discrete distributions, i.e., $\sum_{i=1}^m a_i = 1 = \sum_{j=1}^m$. We can think of $a$ and $b$ as histograms, e.g., the value at bin $i$ is the probability of item $i$. 

Let matrix $C(i,j) \in R^{m \times n}$ be the cost of moving an "atom" from bin $i$ of $a$ to bin $j$ of $b$, and $T(i, j) in $R^{m \times n}$ the *coupling* matrix, describing the amount of mass transporting from bin $i$ to bin $j$. The Kantorvich probem is to solve for the optimal $P^*$,

$$
\begin{aligned}
&\min_{P} \sum_{i,j} P_{i,j} * C_{i,j} \\
&\text{st. } \sum_i P_{i,j} = a, \sum_j P_{i,j} = b
\end{aligned}
$$



## From Stochastic to Deterministic Matching

<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps100.png?raw=true" alt="eps100" width="160"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps020.png?raw=true" alt="eps020" width="170"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps005.png?raw=true" alt="eps005" width="155"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps000.png?raw=true" alt="eps000" width="170"/>
<br>
<em>Optimal Matching solutions with different epsilons. Image Credit: G. Peyre’s twitter account</em>
</p>


## Reference

[1] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." Advances in neural information processing systems 26 (2013)






