---
title: "[개념 정리] KL Divergence"
description: KL Divergence에 대해 다룹니다.
created: 2026-02-12
tags:
  - Math
  - Concept
---
ML/DL을 공부하다 보니 KL Divergence를 접할 때가 많은데, 생각난 김에 한번 정리해보고자 한다.

---

## KL Divergence란?
![[3. Resources/Math/attachments/image.png|image.png|500]]
**KL Divergence(Kullback-Liebler Divergence)** 란 두 확률분포 간의 정보량의 차이를 측정하는 데 사용되는 개념이다.
위처럼 설명하면 말이 어려운데, 그냥 두 확률분포 간의 차이를 측정하는 지표라고 보면 된다.

KL Divergence의 공식은 다음과 같다.
$$D_{KL}(P \| Q) = \sum P(x)\log(\frac{P(x)}{Q(x)})$$
그렇다면 KL Divergence의 의미를 좀 더 직관적으로 이해하기 위해 식을 다음과 같이 변형해보도록 하자.
$$\begin{align*}D_{KL}(P \parallel Q) &= \sum P(x) \log \left( \frac{P(x)}{Q(x)} \right) \\&= \sum P(x) (\log P(x) - \log Q(x)) \\&= - \sum P(x) (\log Q(x) - \log P(x)) \\&= - \sum P(x) \log Q(x) + \sum P(x) \log P(x) \\&= \underbrace{- \sum P(x) \log Q(x)}_{\text{Cross Entropy}}    - \left( \underbrace{- \sum P(x) \log P(x)}_{\text{Entropy}} \right)\end{align*}$$
이렇게 보면 KL Divergence는 결국 두 분포의 Cross-Entropy에서 Entropy를 뺀 값이 된다.
즉, 두 확률분포의 차이 뿐만 아니라 기준분포의 엔트로피(불확실성) 또한 고려한 개념이다.
추가적으로 KL Divergence의 중요한 특징은 비대칭이라는 점이다.
![[3. Resources/Math/attachments/image 1.png|image 1.png|500]]
이는 KL Divergence가 두 분포의 불일치를 계산할 때, 특정 분포를 기준으로 하여 상대적인 차이를 계산하기 때문에 이런 특징을 보이게 된다.
그래서 이를 보완한 척도로서 Jensen-Shannon Divergence가 있다.
$$JSD = \frac{D_{KL}(P \| Q) + D_{KL}(Q \| P)}{2}$$
JSD는 두 분포가 만드는 KL Divergence 값들을 서로 합하여 2로 나눈 값으로 훨씬 두 확률분포 사이의 상대적인 차이를 부드럽게 측정할 수 있다.
## 참고
- [https://www.youtube.com/watch?v=0_E1_GQ_OcQ](https://www.youtube.com/watch?v=0_E1_GQ_OcQ)