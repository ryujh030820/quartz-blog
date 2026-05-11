---
title: "[논문 리뷰] Grouped Query Attention"
description: "논문 GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints에 대해 다룹니다."
date: 2026-04-25
tags:
  - AI
  - Paper
---
## Introduction
Autoregressive 디코더 추론은 디코더 가중치와 모든 attention key 및 value를 각 디코딩 단계마다 로드해야 하기 때문에 Transformer 모델의 심각한 병목 현상이다.
Key와 value를 로드하는 데서 발생하는 메모리 대역폭은 여러 개의 query 헤드를 사용하지만 단일 key 및 value 헤드를 사용하는 **multi-query attention**을 통해 크게 줄일 수 있다.
그러나 multi-query attention(MQA)은 품질 저하 및 훈련 불안정성을 초래할 수 있다.

본 연구는 대규모 언어 모델의 빠른 추론을 위한 두 가지 기여를 포함한다.
1. multi-head attention(MHA)을 가진 언어 모델 체크포인트를 적은 비용으로 MQA를 사용하도록 업트레인할 수 있음을 보여준다.
2. multi-head attention과 multi-query attention 사이의 보간법으로, 쿼리 헤드 그룹당 단일 key 및 value 헤드를 사용하는 grouped-query attention(GQA)을 제안한다.
업트레인된 GQA는 multi-head attention에 가까운 품질을 달성하면서도 multi-query attention만큼 빠르다는 것을 보여준다.
## Method
### Uptraining
Multi-query 모델을 multi-head 모델에서 생성하는 과정은 두 단계로 이루어진다.
첫째, 체크포인트를 변환하고, 둘째, 모델이 새로운 구조에 적응할 수 있도록 추가적인 사전 훈련을 수행한다.
아래 그림은 multi-head 체크포인트를 multi-query 체크포인트로 변환하는 과정을 보여준다.
Key 및 value 헤드의 프로젝션 행렬은 단일 프로젝션 행렬로 평균 풀링(mean pooled)되며,
이는 단일 key 및 value 헤드를 선택하거나 새로운 key 및 value 헤드를 처음부터 무작위로 초기화하는 것보다 더 나은 성능을 보인다고 판단한다.
![[Pasted image 20260425175310.png]]
변환된 체크포인트는 동일한 사전 훈련 레시피를 사용하여 원본 훈련 단계의 작은 비율인 $\alpha$ 동안 추가 사전 훈련된다.
### Grouped-query attention
그룹화된 쿼리 어텐션은 query 헤드를 G개의 그룹으로 나누며, 각 그룹은 단일 key 헤드와 value 헤드를 공유한다.
GQA-G는 G개의 그룹을 가진 그룹화된 쿼리를 의미하며, 따라서 GQA-1은 MQA와 동일하고, 헤드 수와 같은 그룹을 갖는 GQA-H는 MHA와 동일하다.
아래 그림은 그룹화된 쿼리 어텐션과 멀티 헤드/멀티 쿼리 어텐션의 비교를 보여준다.
![[Pasted image 20260425175545.png]]
멀티 헤드 체크포인트를 GQA 체크포인트로 변환할 때, 각 그룹의 key 및 value 헤드는 해당 그룹 내의 모든 원래 헤드를 평균 풀링하여 구성한다.

중간 정도의 그룹 수는 MQA보다 높은 품질이지만 MHA보다 빠른 보간된 모델을 생성하며, 유리한 절충안을 나타낸다.
MHA에서 MQA로 전환하면 H개의 key 및 value 헤드가 단일 key 및 value 헤드로 줄어들어 key-value 캐시의 크기와 따라서 로드해야 하는 데이터 양을 H배로 줄인다.
GQA를 사용하면 모델 크기가 증가함에 따라 대역폭 및 용량이 동일하게 비례적으로 감소한다.

또한, KV 캐시는 모델 차원에 따라 확장되는 반면 모델 FLOPs와 파라미터는 모델 차원의 제곱에 따라 확장되므로, 대규모 모델은 어텐션으로 인한 메모리 대역폭 오버헤드로부터 상대적으로 덜 영향을 받는다.
마지막으로, 대규모 모델에 대한 표준 샤딩(모델을 여러 장치에 분산시키는 것)은 모델 파티션 수만큼 단일 key 및 value 헤드를 복제하지만, GQA는 이러한 파티셔닝으로 인한 낭비를 제거한다.

GQA는 인코더 셀프 어텐션 레이어에는 적용되지 않는다. 인코더 표현은 병렬로 계산되며, 따라서 메모리 대역폭은 일반적으로 주요 병목 현상이 아니다.
## References
- https://arxiv.org/pdf/2305.13245
## Backlinks
- [[[논문 리뷰] Attention Is All You Need]]