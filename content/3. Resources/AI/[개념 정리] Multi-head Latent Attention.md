---
title: "[개념 정리] Multi-head Latent Attention"
description: Multi-head Latent Attention 기법에 대해 다룹니다.
date: 2026-04-28
tags:
  - AI
  - Concept
---
## Low-Rank Approximation of Matrices
우선 MLA를 제대로 이해하기 위해 **LoRA(Low-Rank Approximation of Matrices)** 기법부터 살펴보도록 하자.
핵심적인 아이디어는, 고차원 행렬을 두 개의 저차원 행렬로 변환하기 위해 저차원 근사법을 사용하는 것이다.
만약 행렬 $M$이 $n \times m$ 형태의 행렬이라면, $U$는 $n \times r$ 형태의 행렬이 되고, $V$는 $r \times m$ 형태의 행렬이 된다. 여기서 $r$은 $n$과 $m$보다 모두 작다.
행렬 $UV$는 $M$과 정확히 같지는 않겠지만, 실제 용도로는 충분히 유사할 수 있다. 행렬 $M$을 $U$와 $V$로 분해하는 방법 중 하나는 **특이값 분해(SVD)** 를 사용하는 것이다.
$$
M = U \Sigma V^T
$$
여기서, $U$와 $V$는 정규직교행렬이다. $\Sigma$는 $M$의 특이값들을 포함하는 대각행렬이다.
만약 $\Sigma$의 대각선상에 있는 하위 특이값들을 0으로 만든다면, 사실상 $U$와 $V$의 하위 행들도 함께 제거된 것이 된다. 이러한 곱셈의 결과는 $M$의 근사값이 된다.
만약 $\Sigma$에서 0으로 처리된 요소들의 값이 수치적으로 0에 가깝다면, 이 근사값은 상당히 정확할 것이다.

이 개념은 새로운 것이 아니다. LoRA 방식은 대규모 트랜스포머 모델을 세밀하게 조정하는 데 자주 사용되는 기법이다.
또한, 프로젝션 매트릭스를 근사화함으로써 모델의 기능을 향상시키는 데에도 이러한 방법이 활용된다.
## Multi-head Latent Attention(MLA)
[[[[논문 리뷰] Grouped Query Attention|GQA|]]와 마찬가지로, Multi-head Latent Attention(MLA)도 key와 value의 정보만을 처리한다.
하지만 GQA와 달리, MLA는 여러 query에 걸쳐 key와 value의 정보를 공유하지 않는다.
대신, MLA는 multi-head attention과 동일한 방식으로 작동한다.

입력 시퀀스 $X$에 대해, MLA를 사용한 self-attention 메커니즘에 의해 다음과 같은 결과가 도출된다.
$$
\begin{aligned}

Q &= XW_Q^DW_Q^U = (XW_Q^D)W_Q^U = C_QW_Q^U \\

K &= XW_{KV}^DW_K^U = (XW_{KV}^D)W_K^U = C_{KV}W_K^U \\

V &= XW_{KV}^DW_V^U = (XW_{KV}^D)W_V^U = C_{KV}W_V^U

\end{aligned}
$$
여기서,
- $W_Q^D,W_{KV}^D \in \mathbb{R}^{d\times r}$은 낮은 순위의 압축 행렬들로, $r$의 값이 매우 작다. 이러한 구조를 통해 데이터의 차원을 줄일 수 있다.
- $W_Q^U,W_K^U,W_V^U \in \mathbb{R}^{r\times(n_h d_h)}$는 데이터의 차원을 복원하기 위한 압축 해제용 행렬들이다.
- $r$은 잠재적인 차원을 나타낸다. 보통은 $r \ll n_h\cdot d_h$와 같다.
예를 들어, $K$는 $X$에서 유도된 값이지만, 이때 두 번의 행렬 곱셈이 필요하다.
이는 계산상 낭비처럼 보일 수 있지만, 아래의 설명을 통해 왜 이 방식이 효율적인지 알 수 있다.

이제, 표준적인 attention 연산에 대해 알아보자.
$$
\begin{aligned}

O_h &= \text{softmax}\big(\frac{QK^\top}{\sqrt{d_k}}\big)V \\

&= \text{softmax}\big(\frac{(XW_Q^D W_{Q,h}^U)(XW_{KV}^D W_{K,h}^U)^\top}{\sqrt{d_k}}\big)XW_{KV}^D W_V^U \\

&= \text{softmax}\big(\frac{XW_Q^D W_{Q,h}^U {W_{K,h}^U}^\top {W_{KV}^D}^\top X^\top}{\sqrt{d_k}}\big)XW_{KV}^D W_{V,h}^U \\

&= \text{softmax}\big(\frac{C_Q W_{Q,h}^U {W_{K,h}^U}^\top C_{KV}^\top}{\sqrt{d_k}}\big)C_{KV} W_{V,h}^U

\end{aligned}
$$
이것이 바로 MLA가 계산상의 이점을 얻는 방법이다. key와 value를 나타내는 행렬인 $W^K$과 $W^V$를 각각 처리하는 대신, 압축 행렬은 공유된다.
cross-attention에서도 마찬가지로, key와 value의 입력 시퀀스는 동일하기 때문에, $K$와 $V$의 경우에도 공통된 인자인 $C_{KV}$가 사용된다.

또 다른 중요한 기법은, 오직 압축 해제 행렬 $W^U_Q, W^U_K, W^U_V$에서만 head를 분리한다는 것이다.
따라서 단일 head의 경우, 위의 방정식들에서는 $W^U_{Q,h}, W^U_{K,h}, W^U_{V,h}$이라는 표기법이 사용된다.
이와 같은 방식으로, $C_Q$와 $C_{KV}$도 한번만 계산된 후, 모든 head들에서 공유된다.

또한, 위의 소프트맥스 계산에서 마지막 줄에 나타나는 행렬 곱셈 $W^U_{Q,h}{W^U_{K,h}}^\top$에 주목하자.
이는 입력값 $X$와는 무관한, 두 개의 행렬을 곱하는 연산이다. 따라서 이 행렬 곱셈은 미리 계산해두고 $W_{QK,h}=W^U_{Q,h}{W^U_{K,h}}^\top$로 저장해둘 수 있다.
## References
- https://arxiv.org/pdf/2405.04434
- https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla/