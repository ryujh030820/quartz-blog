---
title: "[CS236 7강] Normalizing Flows - 1"
description: CS236 7강 내용에 대해 다룹니다.
date: 2026-02-11
tags:
  - AI
  - Lecture
---
## 1. 변분 오토인코더(VAE)의 재해석과 한계
### 1.1 오토인코더로서의 VAE 구조
VAE는 잠재 변수 모델과 변분 추론 기술을 결합한 형태이다.
- **인코더 (Encoder, $Q$)**: 데이터 $X$를 입력받아 잠재 변수 $Z$의 사후 분포($Q(Z|X)$)를 근사하는 매개변수를 출력한다.
- **디코더 (Decoder, $P$)**: 잠재 변수 $Z$로부터 원본 데이터 $X$를 재구성하는 생성 과정을 정의한다.
### 1.2 목적 함수: ELBO (Evidence Lower Bound)
![[Pasted image 20260321222248.png]]
VAE의 학습은 다음 두 가지 항으로 구성된 하한선(ELBO)을 최대화하는 과정이다.
- **재구성 손실 (Reconstruction Loss):** $Z$를 통해 $X$를 얼마나 잘 복원하는지 측정한다.
- **KL 발산 규제 (KL Divergence Regularization):** 인코더가 생성한 잠재 변수의 분포가 단순한 사전 분포(예: 가우시안)와 유사하도록 강제한다. 이는 새로운 데이터를 생성할 때 사전 분포에서 $Z$를 샘플링하여 디코더에 통과시킬 수 있게 하는 핵심 장치다.
### 1.3 주요 한계
- **계산 난해성:** 주변 확률 가능도 $p_\theta(x) = \int p_\theta(x, z)dz$를 계산하기 위해 가능한 모든 $z$를 열거하거나 적분해야 하므로 계산 비용이 매우 높다.
- **추론의 부정확성:** 근사 사후 분포 $Q$가 실제 분포와 일치하지 않을 경우 모델 성능에 한계가 발생한다.
## 2. 정규화 유동(Normalizing Flows)의 핵심 개념
정규화 유동은 VAE와 유사한 잠재 변수 모델이지만, 매핑 과정을 가역적이고 결정론적으로 설계하여 차별화된다.
### 2.1 모델의 목표 및 정의
- **단순 분포에서 복잡 분포로:** 가우시안이나 균등 분포와 같은 단순한 사전 분포를 가역 변환을 통해 현실 데이터와 같은 복잡한 형태로 변형한다.
- **가역성(Invertibility):** $X = f_\theta(Z)$일 때, $Z = f_\theta^{-1}(X)$가 존재하며 이를 효율적으로 계산할 수 있어야 한다.
### 2.2 VAE와의 주요 차이점 비교

| 비교 항목        | 변분 오토인코더 (VAE)       | 정규화 유동 (Normalizing Flows)              |
| ------------ | -------------------- | --------------------------------------- |
| **매핑 방식**    | 확률적 (Stochastic)     | 결정론적 & 가역적 (Deterministic & Invertible) |
| **잠재 변수 차원** | 일반적으로 $X$보다 저차원 (압축) | $X$와 $Z$의 차원이 동일해야 함                    |
| **가능도 평가**   | 하한선(ELBO)으로 근사       | 정확한 주변 가능도 계산 가능                        |
| **추론 네트워크**  | 별도의 인코더 $Q$ 필요       | $f_\theta^{-1}$를 통해 직접 추론 가능            |

## 3. 수학적 원리: 변수 변환(Change of Variables)
정규화 유동의 핵심 수학적 기초는 변수 변환 공식이다.
### 3.1 1차원 사례
변수 $Z$가 가역 함수 $f$를 통해 $X = f(Z)$로 변환될 때, $X$의 밀도 함수 $p_X(x)$는 다음과 같다.
$$p_X(x) = p_Z(h(x)) |h'(x)| \quad \text{(단, } h = f^{-1}\text{)}$$
여기서 $|h'(x)|$는 변환에 따른 부피(Volume)의 변화를 보정하는 역할을 한다.
- 증명
	![[Pasted image 20260211202323.png]]
### 3.2 다변량 사례 (General Case)
다차원 벡터 $Z, X \in \mathbb{R}^n$에 대해, 야코비안(Jacobian) 행렬의 행렬식(Determinant)을 사용하여 부피 변화를 계산한다.
$$p_X(x) = p_Z(f^{-1}(x)) \left| \det \left( \frac{\partial f^{-1}(x)}{\partial x} \right) \right|$$
또는 순방향 변환 $f$의 관점에서 다음과 같이 표현할 수 있다.
$$p_X(x) = p_Z(z) \left| \det \left( \frac{\partial f(z)}{\partial z} \right) \right|^{-1}$$
- 증명
	![[Pasted image 20260211202615.png]]
	
	![[Pasted image 20260211203043.png|400]]
## 4. 모델의 명칭과 구조적 특징
### 4.1 "Normalizing"과 "Flow"의 의미
- **정규화 (Normalizing):** 변수 변환 공식을 통해 변환 후에도 확률 밀도의 적분값이 1이 되도록 자동으로 유지(정규화)됨을 의미한다.
- **유동 (Flow):** 여러 개의 가역 변환을 순차적으로 연결(Composition)하여 복잡한 변환을 형성할 수 있음을 의미한다.
    - $Z_0 \xrightarrow{f_1} Z_1 \xrightarrow{f_2} \dots \xrightarrow{f_M} Z_M = X$
    - 전체 행렬식은 각 단계 야코비안 행렬식들의 곱으로 계산된다.
### 4.2 차원 유지의 특성
함수가 가역적이기 위해서는 입력과 출력의 차원이 동일해야 한다. 이로 인해 VAE와 같은 데이터 압축 효과는 없으나, 정보 손실 없이 데이터를 다른 좌표계(Coordinate System)에서 바라보는 효과를 얻는다.
## 5. 학습 및 추론 (Learning and Inference)
### 5.1 학습 (Learning)
데이터셋 D에 대해 로그 가능도를 직접 최대화한다.
$$\max_\theta \sum_{x \in D} \left( \log p_Z(f_\theta^{-1}(x)) + \log \left| \det \left( \frac{\partial f_\theta^{-1}(x)}{\partial x} \right) \right| \right)$$
### 5.2 샘플링 및 추론 (Sampling & Inference)
- **생성:** 사전 분포에서 $Z \sim p_Z(z)$를 샘플링한 후 순방향 변환 $X = f_\theta(Z)$를 적용한다.
- **추론:** 별도의 인코더 없이 역방향 변환 $Z = f_\theta^{-1}(X)$를 통해 데이터의 잠재 표현을 즉시 얻을 수 있다.
## 6. 구현상의 과제: 야코비안 행렬식 계산 효율화
일반적인 $n \times n$ 행렬의 행렬식 계산 비용은 $O(n^3)$으로, 고차원 데이터(예: 이미지) 학습 시 매우 비효율적이다. 이를 해결하기 위해 특수한 구조를 가진 변환을 사용한다.
### 6.1 삼각 야코비안 (Triangular Jacobian)
변환 함수를 설계할 때 $x_i$가 $z_{\le i}$에만 의존하도록 만들면 야코비안 행렬이 하삼각(Lower Triangular) 구조를 갖게 된다. 이 경우 행렬식은 대각 성분들의 곱으로 단순화되어 $O(n)$ 시간에 계산 가능하다.
### 6.2 평면 유동 (Planar Flows)의 예시
Rezende & Mohamed(2016)가 제안한 방식으로, 다음과 같은 형태의 변환을 사용한다. $$f_\theta(z) = z + u h(w^T z + b)$$
이 구조는 행렬식 보조 정리(Matrix Determinant Lemma)를 통해 행렬식을 효율적으로 계산할 수 있으며, 여러 층을 쌓아 매우 복잡한 분포를 생성할 수 있다.
## 7. 결론
정규화 유동 모델은 VAE의 확률적 추론 문제와 가능도 계산의 난해함을 해결하는 우아한 수학적 대안이다. 가역적 신경망을 통해 데이터의 생성과 가능도 평가를 동시에 효율적으로 수행할 수 있으며, 이는 현대 생성 모델(확산 모델 포함)의 중요한 기반 기술로 자리 잡고 있다. 특히 고차원 데이터의 정확한 확률 밀도 추정이 필요한 분야에서 강력한 성능을 발휘한다.
## References
- https://youtu.be/m6dKKRsZwBQ?si=tD2VmgUlhUfLCBGB
- https://deepgenerativemodels.github.io