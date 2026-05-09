---
title: "[개념 정리] SwiGLU"
description: 활성화 함수 중 SwiGLU에 대해 다룹니다.
date: 2026-05-09
tags:
  - AI
  - Concept
---
## Gated Linear Unit(GLU)
**GLU**는 2016년 Dauphin 등이 [Language Modeling with Gated Convolutional Networks](https://arxiv.org/pdf/1612.08083) 논문에서 처음 소개한 활성화 함수이다.
GLU는 신경망의 학습 과정에서 입력 데이터의 중요도를 결정하는 **게이팅(Gating) 메커니즘**을 도입하여, 불필요한 정보는 차단하고 중요한 정보만을 다음 레이어로 전달하는 방식으로 작동한다.
이는 모델의 성능 향상에 크게 기여할 수 있는 방법이다. 수학적으로 GLU는 다음과 같이 정의된다.
$$
\text{GLU}(x,W,V,b,c) = (xW+b) \otimes \sigma(xV+c)
$$
여기서
- $x$는 입력 벡터
- $W, V$는 가중치 행렬
- $b, c$는 편향 벡터
- $\sigma$는 시그모이드 함수로, 비선형 활성화 함수이다.
	- 활성화 함수(일반적으로 시그모이드 $\sigma$)를 적용하여 출력할 정보의 양을 조절한다.
- $\otimes$는 요소별 곱셈을 의미한다.

Gating 메커니즘은 전통적인 비선형 활성화 함수 대신에 사용할 수 있다.
예를 들어, GLU는 입력과 출력의 관계를 제어하는 게이트를 사용하여 정보 흐름을 더 잘 조절한다.
이 방식을 사용하면 입력이 게이트와 결합되어 비선형성을 유지하면서도 정보가 보다 원활하게 전달될 수 있다.
또한 입력 데이터를 선형 변환한 후, 시그모이드 함수를 통해 게이트를 조절하여 결과를 선택적으로 통과시키기 때문에 선형성과 비선형성을 결합하여 더 강력한 표현력을 제공하는 효과가 있다.
## Swish(SiLU)
**Swish(SiLU)** 란 2017년에 발표된 구글의 논문 [Searching for Activation Function](https://arxiv.org/pdf/1710.05941)에서 소개된 활성화 함수이며, 아래와 같이 정의된다.
$$
f(x) = x \cdot \sigma(\beta x)
$$
해당 활성화 함수는 신경 아키텍처 검색을 통해 발견되었으며, 다양한 태스크에서 ReLU보다 우수한 성능을 보인다.
## SwiGLU
**SwiGLU**는 Gated Linear Unit(GLU) 계열에서 파생된 활성화 함수이다.
SwiGLU는 게이팅 부분에 SiLU를 사용한다는 점에서 다른 활성화 함수들과 구별된다.
실험 결과, SwiGLU를 사용할 경우, ReLU와 기타 활성화 함수들에 비해 성능이 크게 향상된다는 것이 입증되었다.

SwiGLU가 피드포워드 블록 내에서 작동하는 방식은 아래 공식과 같다.
$$
\text{SwiGLU}(x,W,V,b,c) = \text{SiLU}(xW+b) \otimes (xV+c)
$$
## References
- https://www.stephendiehl.com/posts/post_transformers/?utm_source=tldrai#swiglu
- https://velog.io/@euisuk-chung/%EA%B0%9C%EB%85%90-GLU%EC%99%80-%EA%B7%B8-%EB%B3%80%ED%98%95%EB%93%A4-%EC%97%AD%EC%82%AC%EC%99%80-%EC%A3%BC%EC%9A%94-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC