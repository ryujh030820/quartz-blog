---
title: Lecture 2
description: UNIST 나승훈 교수님의 수업 '자연어처리' 2강 PPT 내용에 대해 다룹니다.
date: 2026-06-07
tags:
  - AI
  - Lecture
---
## 1. 단어 표현 방법
### 1-1. Discrete Symbol (One-hot Vector)
전통적 NLP에서는 단어를 이산적 기호로 표현:
```
motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
hotel = [0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
```
**문제점:**
- 벡터 차원 = 어휘 크기 (500,000+) → **차원의 저주**
- 두 벡터의 내적 = 0 → **단어 간 유사도 표현 불가**
- `motel`과 `hotel`의 관계를 전혀 포착하지 못함
### 1-2. Distributional Semantics
> _"You shall know a word by the company it keeps"_ — J. R. Firth (1957)

**분포 가설 (Distributional Hypothesis):**
- 비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가짐
- 단어 $w$ 주변에 자주 등장하는 단어들로 $w$의 의미를 표현
### 1-3. Co-occurrence Matrix 기반 벡터
- 두 단어 $w_i$, $w_j$의 동시 출현 횟수 $C(w_i, w_j)$를 행렬로 구성
- 유사한 단어들은 벡터 공간에서 가깝게 위치
- **내적 $\approx$ 동시 출현 빈도**를 근사
## 2. 단어 벡터 (Word Embeddings)
각 단어를 **저차원 밀집 벡터(dense vector)** 로 표현:
![[Pasted image 20260607220545.png]]
- 차원 $d$는 보통 50~300
- 비슷한 의미의 단어들이 벡터 공간에서 **가까운 위치**에 클러스터링됨
## 3. Word2Vec
> Mikolov et al., 2013
### 3-1. 핵심 아이디어
Word2Vec은 **단어 벡터를 독립적으로 학습하는 프레임워크**. 전체 신경망 언어 모델 대신 단어 벡터 자체에 집중.
**학습 과정:**
1. 대규모 코퍼스에서 각 위치 $t$마다 **center word** $c$와 **context word** $o$ 정의
2. 두 벡터의 유사도로 $P(o|c)$ 계산
3. 확률을 최대화하도록 벡터 반복 조정
### 3-2. 두 종류의 단어 벡터
각 단어 $w$는 두 벡터를 가짐:

| 역할           | 표기             | 설명              |
| ------------ | -------------- | --------------- |
| Center word  | $\mathbf{v}_w$ | 해당 단어가 중심 단어일 때 |
| Context word | $\mathbf{u}_w$ | 해당 단어가 주변 단어일 때 |
**왜 두 벡터?** 최적화를 더 쉽게 하기 위함. 학습 후 두 벡터의 평균을 최종 임베딩으로 사용 가능.
### 3-3. 모델 파라미터 행렬
$$U = \begin{bmatrix} \mathbf{u}_1 \ \cdots \ \mathbf{u}_{|\mathcal{V}|} \end{bmatrix}, \quad V = \begin{bmatrix} \mathbf{v}_1 \ \cdots \ \mathbf{v}_{|\mathcal{V}|} \end{bmatrix}$$
- Shape: $|V| \times d$ (**행방향**으로 단어 벡터를 쌓음)
- "Rows not columns in actual DL packages!"
### 3-4. 두 가지 모델 변형

| 모델                 | 방향               | 설명                           |
| ------------------ | ---------------- | ---------------------------- |
| **Skip-gram (SG)** | center → context | center word가 주어졌을 때 주변 단어 예측 |
| **CBOW**           | context → center | 주변 단어들로 center word 예측       |
## 4. Word2Vec 목적함수 및 Gradient
### 4-1. 목적함수
각 위치 $t$에서 윈도우 크기 $m$ 안의 context word들을 예측하는 **likelihood 최대화**:
$$J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \ j \ne 0}} \log P(w_{t+j} | w_t; \theta)$$
### 4-2. 예측 함수: Naive Softmax
$$P(o|c) = \frac{\exp(\mathbf{u}_o^T \mathbf{v}_c)}{\sum_{w \in V} \exp(\mathbf{u}_w^T \mathbf{v}_c)}$$

| 단계                              | 역할                                            |
| ------------------------------- | --------------------------------------------- |
| ① $\mathbf{u}_o^T \mathbf{v}_c$ | center-context 간 유사도 (dot product). 클수록 확률 높음 |
| ② $\exp(\cdot)$                 | 값을 양수로 변환                                     |
| ③ $\sum_{w \in V}$ 정규화          | 전체 어휘에 대한 확률 분포 생성                            |
### 4-3. Gradient 계산
$\mathbf{v}_c$에 대한 gradient를 두 파트로 분해:
$$\frac{\partial J_{(c,o)}}{\partial \mathbf{v}_c} = \underbrace{\mathbf{u}_o}_{\text{Part A}} - \underbrace{\mathbb{E}[\mathbf{u}_x]}_{\text{Part B}}$$
**Part B: 기댓값 계산**

$$\mathbb{E}[\mathbf{u}_x] = \text{softmax}(\mathbf{v}_c \mathbf{U}^T), \mathbf{U} = \sum_{w \in V} P(w|c) \cdot \mathbf{u}_w$$
**최종 해석:**

$$\frac{\partial J}{\partial \mathbf{v}_c} = \mathbf{u}_o - \mathbb{E}[\mathbf{u}_x] = \text{"observed"} - \text{"expected"}$$

| 상황                  | Gradient | 의미          |
| ------------------- | -------- | ----------- |
| expected ≈ observed | → 0      | 모델이 잘 학습됨   |
| expected ≠ observed | → 크다     | 강하게 업데이트 필요 |
## 5. Negative Sampling
### 5-1. 동기
Naive softmax의 문제: 분모 $\sum_{w \in V}$ 계산이 **어휘 크기만큼 비쌈**.
**핵심 아이디어:** "전체 어휘에 대한 확률 분포" 문제를 **이진 분류** 문제로 전환.
### 5-2. 이진 분류기

| 쌍          | 레이블                | 의미            |
| ---------- | ------------------ | ------------- |
| $(c, o)$   | $C = 1$ (positive) | 실제 등장한 쌍      |
| $(c, w_i)$ | $C = 0$ (negative) | 랜덤 샘플링된 노이즈 쌍 |

$$P(C=1|(c,x)) = \sigma(\mathbf{u}_c^T \mathbf{v}_x)$$ $$P(C=0|(c,x)) = 1 - \sigma(\mathbf{u}_c^T \mathbf{v}_x) = \sigma(-\mathbf{u}_c^T \mathbf{v}_x)$$
### 5-3. Loss 함수
$$J = -\underbrace{\log \sigma(\mathbf{u}_o^T \mathbf{v}_c)}_{\text{positive pair}} - \underbrace{\sum_{i=1}^{K} \log \sigma(-\mathbf{u}_{j_i}^T \mathbf{v}_c)}_{\text{K negative pairs}}$$

| 항                                                | 의미                   |
| ------------------------------------------------ | -------------------- |
| $-\log \sigma(\mathbf{u}_o^T \mathbf{v}_c)$      | 진짜 쌍의 내적 ↑ → loss ↓  |
| $-\log \sigma(-\mathbf{u}_{j_i}^T \mathbf{v}_c)$ | 노이즈 쌍의 내적 ↓ → loss ↓ |
### 5-4. Noise 단어 샘플링

$$w_{i_1}, \ldots, w_{i_K} \sim U(w)^{3/4} / Z$$
- Unigram 분포의 **3/4승**: 희귀 단어가 더 자주 샘플링되도록 보정
- 보통 $K = 5 \sim 20$개 사용
### 5-5. Sparse Gradient와 효율적 업데이트
매 SGD 스텝에서 실제 non-zero gradient는 **(1 + K)개 행**만 해당:
- **Sparse matrix update**: 등장한 단어의 행(row)만 선택적 업데이트
- **Hash for word vectors**: 해시 테이블로 등장한 단어만 관리
- 분산 컴퓨팅에서 통신량을 수만 배 절감
## 6. Skip-gram as Implicit Matrix Factorization (Levy & Goldberg, 2014)
### 6-1. 핵심 발견
SGNS가 학습한 벡터는 사실 **PMI 행렬을 암묵적으로 분해**하고 있음:
$$\mathbf{v}_c \cdot \mathbf{u}_o \approx PMI(c, o) - \log k$$
**PMI (Pointwise Mutual Information):**
$$PMI(c, o) = \log \left( \frac{\#(c,o) \cdot |\mathcal{C}|}{\#(c) \cdot \#(o)} \right)$$
### 6-2. 행렬 분해 관점
$$\underbrace{U}_{|V| \times d} \cdot \underbrace{V^T}_{d \times |V|} \approx \underbrace{\text{Shifted PMI Matrix}}_{|V| \times |V|}$$

> 겉보기에 다른 두 접근법(신경망 예측 vs 통계 기반)이 **수학적으로 동치**임을 보임.
## 7. GloVe (Pennington, Socher, Manning, EMNLP 2014)
### 7-1. 동기: 확률 비율의 힘
동시 출현 확률의 **비율**이 raw 확률보다 의미 구분력이 뛰어남:

| $k$     | $P(k\|ice)$          | $P(k\|steam)$        | 비율           | 해석         |
| ------- | -------------------- | -------------------- | ------------ | ---------- |
| solid   | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | **8.9**      | ice에만 관련   |
| gas     | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | **0.085**    | steam에만 관련 |
| water   | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | **1.36 ≈ 1** | 둘 다 관련     |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | **0.96 ≈ 1** | 둘 다 무관     |
### 7-2. 목적함수 유도
**Step 1:** Skip-gram softmax loss를 동시출현 기반으로 재표현
$$J = -\sum_{i \in \mathcal{V}} \sum_{j \in \mathcal{V}} X_{ij} \log P(j|i)$$
**Step 2:** Softmax 정규화 비용 문제 → Least Squares로 전환
$$\hat{J} = \sum_{i,j} X_i \left( X_{ij} - \exp(\mathbf{u}_j^T \mathbf{v}_i) \right)^2$$
**Step 3:** $X_{ij}$가 매우 크면 최적화 불안정 → **log 취하기**
$$\hat{J} = \sum_{i,j} X_i \left( \mathbf{u}_j^T \mathbf{v}_i - \log X_{ij} \right)^2$$
**Step 4:** 일반적인 가중치 함수 $f(X_{ij})$ 도입 → **최종 GloVe 목적함수**
$$\hat{J} = \sum_{i \in \mathcal{V}} \sum_{j \in \mathcal{V}} f(X_{ij}) \left( \mathbf{u}_j^T \mathbf{v}_i + b_i + b_j - \log X_{ij} \right)^2$$
### 7-3. 핵심 등식
$$\mathbf{v}_i^T \mathbf{u}_k + b_i + b_k = \log X_{ik}$$

- 벡터 내적 + bias = **log 동시출현 횟수**를 만족하도록 학습
### 7-4. 의미의 선형 인코딩

$$\mathbf{v}_{ice} \cdot \mathbf{u}_{solid} - \mathbf{v}_{steam} \cdot \mathbf{u}_{solid} \approx \log \frac{P(solid|ice)}{P(solid|steam)}$$
→ **벡터 차이가 확률 비율의 log**에 대응 → 의미 관계가 벡터 공간에서 선형 구조로 인코딩
### 7-5. GloVe 특징
- **빠른 학습**: Least squares 형태로 효율적 최적화
- **대규모 코퍼스에 확장 가능**
- Word2Vec보다 전역적 통계 정보를 더 잘 활용
## 8. 단어 벡터 평가
### 8-1. Intrinsic (내재적) 평가
**Word Vector Analogy:**
$$\text{a:b :: c:?} \quad \Rightarrow \quad d = \arg\max_i \frac{(x_b - x_a + x_c)^T x_i}{|x_b - x_a + x_c|}$$
- 예: `man:woman :: king:?` → queen
- 입력 단어들은 검색에서 제외
- **한계**: 의미 정보가 있어도 선형이 아니면 포착 불가
**Meaning Similarity (WordSim353):**
- 단어 쌍 간 거리와 인간 판단의 상관관계 측정
### 8-2. Extrinsic (외재적) 평가
실제 NLP 태스크 성능으로 측정:

| Model     | Dev      | Test | ACE      | MUC7     |
| --------- | -------- | ---- | -------- | -------- |
| Discrete  | 91.0     | 85.4 | 77.4     | 73.4     |
| SVD       | 90.8     | 85.7 | 77.3     | 73.7     |
| CBOW      | 93.1     | 88.2 | 82.2     | 81.1     |
| **GloVe** | **93.2** | 88.3 | **82.9** | **82.2** |
→ **NER(Named Entity Recognition)** 태스크에서 GloVe가 전반적으로 우수
## 9. Word Senses & Polysemy
### 9-1. 문제: 다의어
대부분의 단어는 여러 의미를 가짐. 하나의 벡터로 모든 의미를 표현하는 것이 충분한가?
예: `pike` = 창, 물고기, 고속도로, ...
### 9-2. 해결: Linear Superposition (Arora et al., TACL 2018)
서로 다른 의미들이 표준 word embedding 공간에서 **선형 중첩(superposition)** 으로 존재:
$$\mathbf{v}_{pike} = \alpha_1 \mathbf{v}_{pike_1} + \alpha_2 \mathbf{v}_{pike_2} + \alpha_3 \mathbf{v}_{pike_3}$$
$$\alpha_i = \frac{f_i}{f_1 + f_2 + f_3} \quad \text{(빈도 기반 가중치)}$$

**핵심 결과**: Sparse Coding 아이디어를 이용하면 각 의미 벡터를 **분리(disentangle)** 할 수 있음 (의미가 충분히 자주 등장하는 경우에 한해).
## 요약: Word2Vec vs GloVe 비교

|항목|Word2Vec (Skip-gram)|GloVe|
|---|---|---|
|**학습 방식**|예측 기반 (SGD)|통계 기반 (Least Squares)|
|**정보 활용**|지역적 (window 기반)|전역적 (전체 동시출현 행렬)|
|**Loss 함수**|Cross-entropy (Negative Sampling)|Weighted Least Squares|
|**목표값**|$P(o\|c)$ 최대화|$\mathbf{u}_j^T\mathbf{v}_i \approx \log X_{ij}$|
|**이론적 관계**|PMI 행렬의 암묵적 분해 (SGNS)|PMI와 명시적 연결|
|**계산 효율**|Negative Sampling으로 개선|정규화 불필요|
