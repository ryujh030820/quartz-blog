---
title: "[논문 리뷰] Attention Is All You Need"
description: 논문 Attention Is All You Need에 대해 다룹니다.
date: 2026-02-12
tags:
  - AI
  - Paper
---
# 1. Sequence Modeling

![[3. Resources/AI/attachments/image.png|500]]
**Sequence modeling**은 어떠한 Sequence를 가지는 데이터로부터 또 다른 Sequence를 가지는 데이터를 생성하는 task이다. 대표적인 예로는 machine translation과 chatbot 등이 있다.
위 이미지는 터키어로 쓰인 문장을 sequence model에 입력하여 영어로 번역된 문장이 나오도록 하는 예제이다.
![[3. Resources/AI/attachments/image 1.png]]
이러한 Sequence modeling에는 대부분 RNN 기반 모델들이 주축으로 사용되었는데, 몇 가지 단점들이 있다.
위와 같은 Recurrent model들은 모든 데이터를 한꺼번에 처리하는 것이 아니라 sequence position $t$에 따라 순차적으로 입력에 넣어주어야 한다.
이러한 한계는 긴 sequence 길이를 가지는 데이터를 처리해야 할 때, memory와 computation에 많은 부담이 생기게 된다.

## Attention mechanism

![[image 2.png|300]]
Attention이라는 메커니즘은 이 논문에서 처음 나온 것이 아니다. Attention은 input 또는 output 데이터에서 sequence distance에 무관하게 서로 간의 dependencies를 모델링한다.
예를 들어, 위 이미지에서는 프랑스어를 영어로 번역하는 sequence modeling에서 attention을 사용할 때 그 correlation matrix를 나타낸다.

논문에서 제안한 Transformer는 기존의 RNN 또는 CNN 대신 오직 Attention 메커니즘만을 전적으로 사용하여, 기존 모델보다 훨씬 더 높은 병렬화 효율성과 빠른 훈련 속도 및 번역 태스크에서 좋은 성능을 보여준다.

## Sequence-to-Sequence

![[image 3.png]]
논문에는 나와 있지 않지만 Transformer의 장점을 소개할 때 Sequence-to-sequence method가 종종 등장한다. 그 이유를 간단히 살펴보자.
먼저 Sequence-to-sequence는 위와 같은 아키텍처를 가지고 있다.

이전에 설명했듯 Recurrent model은 sequence 순으로 데이터가 입력되는데, 이전 데이터의 hidden state $h_t$가 다음 데이터의 hidden state $h_{t+1}$를 구할 때 사용된다.
즉, 어떠한 시점 $t$에서 구한 hidden state $h_t$는 그 전 sequence들($1, 2, \dots, t-1$)의 정보를 함축하고 있다고 볼 수 있다.
따라서 위 이미지를 예로 들면, tomorrow를 입력으로 받아 출력되는 encoder의 마지막 hidden state는 그 이전 단어들(are, you, free)에 대한 정보까지 함축하고 있는 것이다.

Sequence-to-sequence 모델은 이 encoder의 최종 output을 일종의 embedded vector로써 사용하여 Decoder에 넣어주게 되는데, memory와 computation 때문에 embedded vector의 maximum length를 제한해야 한다.
긴 sequence 데이터를 처리해야 할 때, 제한된 크기의 vector로 모든 정보를 담아내야 하기 때문에 정보의 손실이 커지고 이에 따라 성능의 병목 현상이 일어난다.

이러한 문제를 완화하기 위해 encoder의 모든 state를 decoder에 참조시키거나, attention을 적용하는 등의 여러 시도가 있었는데, 그중 가장 효율적이고 대세가 된 것이 transformer라고 생각하면 된다.

# 2. Model Architecture

![[image 4.png|300]]
대부분 경쟁력 있는 Neural sequence transduction model들은 대부분 **encoder-decoder** 구조를 가지고 있다.
Transformer도 이 구조를 따르고, 그 내부는 self-attention과 fully connected layer로만 구성되어 있다.

## 2.1 Attention

![[image 5.png|400]]
본격적으로 Attention에 대해 알아보도록 하자.

### Scaled Dot-Product Attention

먼저 input으로 **Query(Q)**, **Key(K)**, **Value(V)** 총 3개의 입력 요소가 들어온다.
여기서 **Query**는 각각의 단어들에 다른 어떤 단어들이 집중할지 질문하는 주체, **Key**는 Query의 질문에 응답하는 주체, 마지막 **Value**는 Query와 Key가 매칭이 된 경우 그에 응답하는 데이터의 값들을 의미한다.
이렇게만 들으면 이해가 어렵지만, 이후 Query와 Key가 만나서 생성하는 score matrix를 본다면 어느정도 감이 올 것이다.

먼저 계산식을 살펴본 후에 예를 들어 이해해보자.
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
여기서 Query $Q$와 Key $K$는 문장의 모든 단어들에 대한 vector들을 쌓아놓은 행렬에 각각 다른 선형 변환을 거친 행렬이다.
또한 $d_k$는 $Q$와 $K$의 dimension을 의미한다.(value $v$의 dimension은 달라도 되지만 논문에서는 $d_k = d_v$로 두고 사용한다.)
![[image 6.png]]
예를 들어 AI is awesome이라는 문장에 attention을 적용한다고 해보자. 위 이미지에서 $Q$는 awesome, 그리고 $K$는 모든 단어들의 stacked matrix다.
여기서 $QK^T$는 한 단어(awesome)와 모든 단어(AI, is, awesome)들의 dot product를 해줌으로써 어떠한 relation matrix를 만들어낸다.
본 논문에서는 기존의 dot-product attention 방식과는 다르게 $\frac{1}{\sqrt{d_k}}$로 scaling을 해주는데, 이는 $d_k$ 값이 클 경우,
dot-product의 크기가 커져서 softmax 함수의 기울기가 매우 작은 영역으로 밀려날 수 있기 때문이다.
![[image 7.png]]
마지막으로 softmax를 통해 Query의 단어가 Key의 각 단어들에 어느정도의 상관관계가 있는지를 확률 분포 형태로 만들고,
이를 $V$와 dot product를 해줌으로써 기존 vector에 $Q$와 $K$의 correlation 정보를 더한 vector를 만든다.
![[image 8.png]]
옵션으로 Mask layer가 있는데, 이는

1. 패딩 토큰을 무시하거나,
2. GPT와 같이 Transformer의 Decoder 부분만을 사용하는 모델의 경우, 미래 단어를 보는 것을 방지한다.

### Multi-Head Attention

하나의 attention function을 사용하는 것보다, 서로 다른 학습된 linear projection을 통해 여러 개의 attention function들을 만드는 것이 더 효율적이다. (그러나 Results를 보면 head가 너무 많아도 성능이 저하된다고 한다.)
나중에 function의 출력들은 concatenate 되고 다시 linear function을 통해 매핑된다.
이러한 기법은 CNN이 여러 개의 필터를 통해서 다시 convolution output을 구하는 것과 비슷한 효과를 보인다. (개별 attention head가 서로 다른 작업을 수행한다.)
$$\begin{align*}\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O \\\qquad \text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\end{align*}$$
또한, head가 8개라고 했을 때 Query, Key, Value의 dimension은 $d_k = d_v = d_{model}/h = 64$를 사용한다.
각 head의 차원이 감소되었기 때문에 총 계산 비용은 전체 차원을 가진 단일 head attention과 유사하다.

### Different Use for multi-head attention

![[image 9.png|300]]
위 아키텍처 그림에서 나와 있듯이, 논문에서는 multi-head attention을 3종류로 나누어서 사용한다.

1. **self-attention in encoder(빨간색)**: encoder에서 사용되는 self-attention으로 queries, keys, values 모두 encoder로부터 가져온다. encoder의 각 position은 그 전 layer의 모든 position들을 참조하고, 이는 해당 position과 모든 position 간의 correlation information을 더해주게 된다.
2. **self-attention in decoder(연두색)**: 전체적인 과정과 목표는 encoder의 self-attention과 같다. 하지만 decoder의 경우, sequence model의 auto-regressive 속성을 보존해야 하기 때문에 masking vector을 사용하여 해당 position 이전의 벡터만을 참조한다.
3. **encoder-decoder attention(파란색)**: self-attention in decoder 다음으로 사용되는 layer이다. queries는 이전 decoder layer에서 가져오고, keys와 values는 encoder의 output에서 가져온다. 이는 decoder의 모든 position의 vector들로 encoder의 모든 position 값들을 참조함으로써 decoder의 sequence vector들이 encoder의 sequence vector들과 어떠한 correlation을 가지는지 학습한다.

## 2.2 Position-wise Feed-Forward Networks

Attention layer와 함께 fully connected feed-forward network가 사용된다. 이는 각 위치에 개별적으로 그리고 동일하게 적용된다.
이는 ReLU 활성화가 중간에 있는 두 개의 선형 변환으로 구성된다.
입력 및 출력의 차원은 $d_{model} = 512$이고, 내부 레이어의 차원은 $d_{ff} = 2048$이다.
$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

## 2.3 Embeddings and Softmax

다른 sequence transduction 모델과 마찬가지로, input과 output token을 embedding layer를 거쳐서 사용한다.
이렇게 생성된 embedded vector는 semantic한 특성을 잘 나타내게 된다.
또한, 논문에서는 input embedding과 output embedding에서 weight matrix는 서로 공유한다.

## 2.4 Positional Encoding

Transformer의 경우 순환(recurrence) 구조나 합성곱(convolution)을 포함하지 않기 때문에,
모델이 시퀀스의 순서를 활용하기 위해서는 시퀀스 내 토큰의 상대적 또는 절대적 위치에 대한 정보를 주입해야 한다. 이 역할을 하는 것이 바로 **positional encoding**이다.

이러한 positional encoding으로 선택할 수 있는 방식은 다양한데, 본 논문에서는 sin 및 cosine 함수를 사용한다.
$$\begin{align*}PE_{(pos,2i)} &= \sin\left(pos / 10000^{2i/d_{\text{model}}}\right) \\PE_{(pos,2i+1)} &= \cos\left(pos / 10000^{2i/d_{\text{model}}}\right)\end{align*}$$
여기서 $pos$는 position, $i$는 dimension이다.
논문에서 이 함수를 선택한 이유는 임의의 고정된 오프셋 $k$에 대해 $PE_{pos + k}$를 $PE_{pos}$의 선형 함수로 표현할 수 있기 때문에 모델이 상대적 위치에 따라 쉽게 attention할 수 있다고 가정했기 때문이다.
자세한 내용은 아래 링크를 참고하기를 바란다…
[https://skyjwoo.tistory.com/entry/positional-encoding%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80](https://skyjwoo.tistory.com/entry/positional-encoding%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)
![[transformer_decoding_1.gif|500]]

![[transformer_decoding_2.gif|500]]
전체 과정을 시각화한 이미지는 위와 같다.

# 3. Why Self-Attention

이 섹션에서는 왜 self-attention이 RNN이나 CNN 기반 모델들보다 좋은지에 대해 3가지 사항을 고려한다.
![[image 10.png]]

## 1) the total computational complexity per layer

계산 복잡성 측면에서 self-attention 레이어는 시퀀스 길이 $n$이 표현 차원 $d$보다 작을 때 Recurrent 레이어보다 빠르다.
이는 word-piece 및 byte-pair 표현과 같은 기계 번역의 최첨단 모델에서 사용되는 문장 표현에서 가장 흔한 경우다.

## 2) the amount of computation that can be parallelized

이전에 설명했듯 RNN은 input을 순차적으로 입력 받아 총 $n$번 RNN cell을 거치게 되고, self-attention layer는 input의 모든 position 값들을 연결하여 한번에 처리할 수 있다.
이는 병렬화 측면에서 큰 이점을 가진다.

## 3) the path length between long-range dependencies in the network

장거리 의존성을 학습하는 것은 많은 시퀀스 변환 작업에서 중요한 과제이다.
이러한 의존성을 학습하는 능력에 영향을 미치는 한 가지 중요한 요소는 순방향 및 역방향 신호가 네트워크에서 통과해야 하는 경로의 길이다.
![[image 11.png|400]]
self-attention은 각 token들을 모든 token들과 참조하여 그 correlation information을 구해서 더해주기 때문에, 장거리 의존성을 더 쉽게 학습할 수 있다는 장점을 가진다.

# 🔍 References

- [https://aistudy9314.tistory.com/63](https://aistudy9314.tistory.com/63)
- [https://www.youtube.com/watch?v=DdpOpLNKRJs](https://www.youtube.com/watch?v=DdpOpLNKRJs)
