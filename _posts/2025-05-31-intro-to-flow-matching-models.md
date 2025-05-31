---
layout: post
title: "Flow Matching: A Gentle Intro to Flow Matching Models"
author: "Bach Do"
categories: journal
tags: [documentation,sample]
image: mountains.jpg
---

## It's all about the vector field

Flow Matching is a powerful class of generative models behind systems like Stable Diffusion 3 and Metagen, capable of generating highly realistic images and videos. The key idea is to construct a probability flow — a path $X_0, \ldots, X_t, \ldots, X_1 \) over time \( t \in [0, 1]$ — where  $X_0$ is sampled from an initial distribution $p_{\text{init}} $, and $X_1$ comes from the target distribution $p_{\text{data}}(\cdot)$.

<p align=center>
![VectorFieldFLow](https://github.com/bachvietdo01/bachvietdo01.github.io/blob/main/assets/img/vf_flow.gif?raw=true)
<br>
*Vector field in black generates flows in red*


