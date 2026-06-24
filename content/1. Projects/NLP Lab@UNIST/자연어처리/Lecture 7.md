---
title: "\bLecture 7"
description: UNIST 나승훈 교수님의 수업 '자연어처리' 7강 PPT 내용에 대해 다룹니다.
date: 2026-06-24
tags:
  - AI
  - Lecture
---
## 1. Vanilla RNN의 문제점 복습
### 1.1 Vanishing / Exploding Gradient
Vanilla RNN에서 BPTT를 수행하면:

$$\frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{h}^{(k)}} = \prod_{j=k+1}^{t} \frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}} = \prod_{j=k+1}^{t} W_h^\top \cdot \text{diag}\left(\tanh\!\left(\mathbf{h}^{(j-1)}\right)\right)$$
- 각 단계에서 $W_h^\top$와 $\tanh$ (≤ 1)이 반복 곱해짐
- 시퀀스가 길어질수록 gradient가 **지수적으로 소멸(vanish)** 하거나 **폭발(explode)**
- **Vanishing**: 먼 과거의 gradient가 0에 수렴 → long-range dependency 학습 불가
- **Exploding**: gradient가 발산 → 학습 불안정 (대책: **Gradient Clipping**)
### 1.2 Gradient Clipping (Exploding 대책)
$$\text{if } \|\mathbf{g}\| > \text{threshold}: \quad \mathbf{g} \leftarrow \frac{\text{threshold}}{\|\mathbf{g}\|} \mathbf{g}$$
- 방향은 유지, 크기만 제한
- Exploding에는 효과적, vanishing에는 무력
## 2. LSTM (Long Short-Term Memory)
### 2.1 핵심 아이디어
> **Cell state** $\mathbf{c}^{(t)}$라는 별도의 "컨베이어 벨트"를 두어, 게이트를 통해 정보 흐름을 능동적으로 제어
### 2.2 LSTM 수식

| 게이트/상태 | 수식 | 역할 |
|---|---|---|
| **Forget gate** | $\mathbf{f}^{(t)} = \sigma\!\left(\mathbf{W}_f \mathbf{h}^{(t-1)} + \mathbf{U}_f \mathbf{x}^{(t)}\right)$ | 이전 cell state를 얼마나 잊을지 |
| **Input gate** | $\mathbf{i}^{(t)} = \sigma\!\left(\mathbf{W}_i \mathbf{h}^{(t-1)} + \mathbf{U}_i \mathbf{x}^{(t)}\right)$ | 새 정보를 얼마나 쓸지 |
| **Output gate** | $\mathbf{o}^{(t)} = \sigma\!\left(\mathbf{W}_o \mathbf{h}^{(t-1)} + \mathbf{U}_o \mathbf{x}^{(t)}\right)$ | cell state를 얼마나 출력할지 |
| **New cell content** | $\tilde{\mathbf{c}}^{(t)} = \tanh\!\left(\mathbf{W}_c \mathbf{h}^{(t-1)} + \mathbf{U}_c \mathbf{x}^{(t)}\right)$ | 이번 스텝에 추가할 후보 정보 |
| **Cell state** | $\mathbf{c}^{(t)} = \mathbf{f}^{(t)} \circ \mathbf{c}^{(t-1)} + \mathbf{i}^{(t)} \circ \tilde{\mathbf{c}}^{(t)}$ | 기억 업데이트 (핵심!) |
| **Hidden state** | $\mathbf{h}^{(t)} = \mathbf{o}^{(t)} \circ \tanh\!\left(\mathbf{c}^{(t)}\right)$ | 출력 / 다음 스텝 입력 |
> **참고**: 모든 게이트의 입력은 $\mathbf{h}^{(t-1)}$과 $\mathbf{x}^{(t)}$뿐이며, $\mathbf{c}^{(t-1)}$은 **직접** 들어가지 않는다 (Peephole Connection은 변형 아키텍처에서 추가).
### 2.3 Cell state vs. Hidden state 비교

|       | $\mathbf{c}^{(t)}$ (Cell State) | $\mathbf{h}^{(t)}$ (Hidden State) |
| ----- | ------------------------------- | --------------------------------- |
| 비유    | 내부 장기 기억                        | 단기 기억 / 출력                        |
| 변환    | 덧셈 위주 (gradient highway)        | $\tanh$ 통과 후 출력                   |
| 외부 노출 | 노출 안 됨                          | 출력으로 사용                           |
## 3. LSTM이 Vanishing Gradient를 완화하는 방법
### 3.1 $\frac{dc^{(t)}}{dc^{(t-1)}}$ 분석
$\mathbf{c}^{(t)} = \mathbf{f}^{(t)} \circ \mathbf{c}^{(t-1)} + \mathbf{i}^{(t)} \circ \tilde{\mathbf{c}}^{(t)}$를 전미분하면 4개의 경로:
$$\frac{d\mathbf{c}^{(t)}}{d\mathbf{c}^{(t-1)}} = \underbrace{\frac{\partial \mathbf{c}^{(t)}}{\partial \mathbf{f}^{(t)}}\frac{\partial \mathbf{f}^{(t)}}{\partial \mathbf{h}^{(t-1)}}\frac{\partial \mathbf{h}^{(t-1)}}{\partial \mathbf{c}^{(t-1)}}}_{\text{forget gate 경로}} + \underbrace{\cdots}_{\text{input/cell 경로}} + \underbrace{\mathbf{f}^{(t)}}_{\text{직접 경로 (핵심!)}}$$
정리하면:
$$\frac{d\mathbf{c}^{(t)}}{d\mathbf{c}^{(t-1)}} = \text{diag}\Big(\mathbf{c}^{(t-1)} \circ \sigma'(\cdot)\, \mathbf{W}_f \circ \mathbf{o}^{(t-1)} \circ \tanh'(\mathbf{c}^{(t-1)}) + \cdots + \mathbf{f}^{(t)}\Big)$$
### 3.2 핵심: forget gate의 직접 경로
Cell state의 직접 gradient 경로는:
$$\frac{\partial}{\partial \mathbf{c}^{(t-1)}}\left[\mathbf{f}^{(t)} \circ \mathbf{c}^{(t-1)}\right] = \mathbf{f}^{(t)}$$
따라서 T 스텝을 거슬러 올라가면:
$$\frac{d\mathbf{c}^{(1)}}{d\mathbf{c}^{(0)}} \approx \prod_{t} \mathbf{f}^{(t)}$$

### 3.3 제어 메커니즘

| $\mathbf{f}^{(t)}$ 값 | 효과                                    |
| -------------------- | ------------------------------------- |
| $\approx 1$          | gradient가 거의 그대로 흘러 **vanishing 방지**  |
| $\approx 0$          | gradient 차단 → 의도적 망각                  |
| 수렴이 0으로 향하면          | 학습을 통해 $\mathbf{f}^{(t)}$를 높게 조정하여 보정 |
> **핵심 메시지**: 네트워크가 gradient를 *언제 vanish시킬지*, *언제 보존할지*를 **gate 값 조정으로 스스로 학습**한다. (RNN과 달리 계속 동일한 값이 곱해지지 않기 때문에 vanishing/exploding gradient 문제에 강인한 것으로 이해하였다.)
> 완전한 해결이 아닌 "too quickly vanishing을 방지"하는 것이 더 정확한 표현.
## 4. Fancy RNN Variants
### 4.1 Bidirectional RNN
![[Pasted image 20260624160014.png|400]]
- **동기**: 어떤 단어의 의미는 왼쪽 문맥뿐 아니라 **오른쪽 문맥**도 필요
  - 예: "I hate spiders but **my sister** loves them" → "sister"의 역할은 "loves"를 봐야 알 수 있음
- **구조**: forward RNN + backward RNN, hidden state를 concatenate
- **제한**: 미래 토큰을 볼 수 있으므로 **language modeling에는 사용 불가** (generation 불가)
- **적합**: 분류, 번역의 encoder, QA 등 전체 시퀀스를 한 번에 볼 수 있는 태스크
### 4.2 Multi-layer (Deep) RNN
![[Pasted image 20260624160058.png|400]]
- **동기**: 단일 레이어 RNN은 표현력이 제한됨
- 각 레이어가 **다른 수준의 추상적 특징** 학습
- 일반적으로 **2~4 레이어**가 실용적 (RNN은 깊이보다 길이 때문에 이미 비용이 큼)
- Skip connection (ResNet 방식) 추가 시 더 깊게 가능
## 5. RNN의 근본적 한계: Linear Interaction Distance
### 5.1 문제 정의
RNN은 시퀀스를 **왼쪽→오른쪽 순차적으로** 처리하므로:
$$\text{두 단어 간 상호작용에 필요한 step 수} = O(\text{sequence length})$$
### 5.2 두 가지 구체적 문제
**문제 ①: Long-distance dependency 학습의 어려움**
- 멀리 떨어진 단어 쌍은 $O(n)$ step의 직렬 전달 필요
- Gradient 소멸/폭발 위험
- 예: *"The **chef** who the guests who came talked about **cooked** the meal"*
  - *chef*와 *cooked* 사이에 긴 관계절이 끼어 있음
**문제 ②: 선형 순서가 구조에 "굳어버림" (Linear order is "baked in")**
- 언어의 의미 구조는 **계층적(hierarchical)** 인데, RNN은 선형 순서 처리가 구조 자체에 내재
- Dependency parsing에서 이미 알듯, 단어 간 관계는 선형 거리와 무관
## 6. Attention의 등장
### 6.1 핵심 개념
> **Attention**: 각 단어의 representation을 **query**로 사용해, value들의 집합으로부터 직접 정보를 가져오는 메커니즘

| 용어        | 의미                             |
| --------- | ------------------------------ |
| **Query** | 정보를 요청하는 현재 단어의 representation |
| **Key**   | 각 단어가 얼마나 relevant한지 비교용 벡터    |
| **Value** | 실제 정보를 담은 벡터                   |
### 6.2 이전 강의와의 차이
- 이전: **decoder → encoder** 방향의 attention (seq2seq)
- 이번: **단일 문장 내부** self-attention → Transformer의 핵심
### 6.3 RNN 대비 두 가지 장점
**장점 ①: 병렬화 가능 (Parallelizable)**

| | RNN | Attention |
|---|---|---|
| 연산 순서 | 순차적 (t-1 완료 후 t 계산) | 모든 위치 동시 계산 가능 |
| Unparallelizable ops | $O(n)$ | $O(1)$ |
**장점 ②: Interaction Distance = O(1)**
- RNN: 멀리 떨어진 두 단어 → $O(n)$ step 필요
- Attention: **매 레이어마다** 모든 단어가 모든 단어와 **직접** 상호작용
$$\text{maximum interaction distance} = O(1)$$
### 6.4 구조 시각화
![[Pasted image 20260624160240.png|500]]
- 매 레이어에서 각 단어는 이전 레이어의 **모든** 단어를 참조
## 7. RNN vs. Attention 최종 비교

| 항목 | Vanilla RNN | LSTM | Attention |
|---|---|---|---|
| Interaction distance | $O(n)$ | $O(n)$ (완화됨) | $O(1)$ |
| 병렬 처리 | 불가 | 불가 | 가능 |
| Long-range dependency | 어려움 | 부분 완화 | 직접 연결 |
| 선형 순서 의존 | 구조에 내재 | 구조에 내재 | 위치 인코딩으로 선택적 반영 |
| 메모리 | $O(n)$ 순차 | $O(n)$ 순차 | $O(n^2)$ attention map |
## 8. 핵심 요약
```
Vanilla RNN
  → 문제: Vanishing/Exploding Gradient, linear O(n) interaction
  
LSTM
  → 해결: Cell state + forget gate로 gradient 제어
  → 하지만: 여전히 순차적, O(n) interaction distance
  
Bidirectional / Multi-layer RNN
  → 보완: 양방향 문맥, 계층적 표현력
  → 하지만: 근본적 sequential 구조는 유지
  
Attention
  → 해결: O(1) interaction, 완전 병렬화
  → 다음 단계: Transformer = Attention만으로 구성된 모델
```