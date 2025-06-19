---
layout: post
title: "A Gentle Intro to Entropic Optimal Transport: Sinkhorn Algorithm"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: a3_sinkhorn_algo.png
---

**Optimal Transport (OT)** has deep roots in mathematics, started with the work of [Monge](https://tinyurl.com/4aa33a2f) then [Kantorovich](https://tinyurl.com/bdeys323). Interest in OT was revived in the 1990s by Yann Brenier and the followed in the 2000s by Cédric Villani, who authored two monographs: Topics in Optimal Transportation and Optimal Transport: Old and New, helping to spread knowledge about the applications of OT. More recently, OT has gained traction in machine learning in areas like deep generative modeling, transfer learning, and reinforcement learning. Given its growing importance, this post introduces OT by discussing how to solve the **entropic optimal transport** problem with the **Sinkhorn algorithm**.

## From Stochastic to Deterministic Matching

<p align="center">
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps100.png?raw=true" alt="eps100" width="190"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps020.png?raw=true" alt="eps020" width="200"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps005.png?raw=true" alt="eps005" width="185"/>
<img src="https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/a3_sinkhorn_eps000.png?raw=true" alt="eps000" width="200"/>
<br>
<em>Optimal Matching solutions with different epsilons. Image Credit: G. Peyre’s twitter account</em>
</p>


## Reference

[1] Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." Advances in neural information processing systems 26 (2013)






