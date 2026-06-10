---
title: Lecture 3
description: UNIST 나승훈 교수님의 수업 '자연어처리' 3강 PPT 내용에 대해 다룹니다.
date: 2026-06-10
tags:
  - AI
  - Lecture
---
## 1. Supervised Learning 개요
### 1.1 Supervised Learning
- 목표: 라벨된 입력-출력 쌍 $D = {(\mathbf{x}_i, y_i)}_{i=1}^N$ 으로부터 입력 $\mathbf{x}$ → 출력 $y$ 매핑 학습
- $D$: **training set**, $N$: 학습 예제 수
- $\mathbf{x}_i$: D차원 feature 벡터, 보통 $N \times D$ **design matrix** $\mathbf{X}$에 행 단위로 저장
- 두 가지 유형
    - **Classification**: $y \in {1, 2, \dots, K}$ (유한 클래스)
    - **Regression**: $y \in \mathbb{R}$
### 1.2 Unsupervised Learning
- $D = {\mathbf{x}_i}_{i=1}^N$ — 라벨 없음
- "흥미로운 패턴" 발견이 목표 (knowledge discovery), 명확한 error metric 없음
### 1.3 분류 접근법
- Naïve Bayes (생성 모델), Logistic Regression (판별 모델), SVM, Perceptron/MLP, Deep Learning
## 2. Matrix Calculus
### 2.1 기본 정의 (Jacobian formulation)
- $\mathbf{x} \in \mathbb{R}^{n\times 1}$ (열벡터)일 때
    - $y$가 **스칼라**: $\dfrac{\partial y}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times n}$ (행벡터)
    - $\mathbf{y} \in \mathbb{R}^{m\times 1}$ (벡터): $\dfrac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$ (**Jacobian**: m개의 식 → m개의 행)
- $y$가 스칼라이고 $\mathbf{X} \in \mathbb{R}^{n \times m}$ 행렬이면 $\dfrac{\partial y}{\partial \mathbf{X}} \in \mathbb{R}^{m \times n}$
### 2.2 Chain Rule
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{u}} \frac{\partial \mathbf{u}}{\partial \mathbf{x}}, \qquad (m\times n) = (m \times p)(p \times n)$$
- $p = |\mathbf{u}|$, $m = |\mathbf{z}|$, $n = |\mathbf{x}|$ — **차원이 정확히 맞물리는 행렬 곱**
### 2.3 Jacobian vs. Matrix derivative

| 형태                     | 정의                                                                  | 비고                           |
| ---------------------- | ------------------------------------------------------------------- | ---------------------------- |
| Jacobian form          | $\frac{\partial y}{\partial \mathbf{x}} \in \mathbb{R}^{1\times n}$ | chain rule이 자연스러움            |
| Matrix derivative form | Jacobian의 **전치** ($n \times 1$)                                     | gradient descent 업데이트에 자연스러움 |
문맥에 따라 둘 다 사용한다.
### 2.4 유용한 공식 (Jacobian form)
**(1) Matrix × column vector, 벡터에 대한 미분** $$\mathbf{z} = \mathbf{W}\mathbf{x} ;\Rightarrow; \frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \mathbf{W}$$
**(2) 선형변환 + Chain rule (x가 z의 함수일 때)** $$\mathbf{y} = \mathbf{A}\mathbf{x},;; \mathbf{x}=\mathbf{f}(\mathbf{z}) ;\Rightarrow; \frac{\partial \mathbf{y}}{\partial \mathbf{z}} = \mathbf{A}\frac{\partial \mathbf{x}}{\partial \mathbf{z}}$$
- 증명: $y_i = \sum_k A_{ik}x_k$ → $\dfrac{\partial y_i}{\partial z_j} = \sum_k A_{ik}\dfrac{\partial x_k}{\partial z_j}$ — 상수 행렬 $\mathbf{A}$는 미분 밖으로 나오는 순수 chain rule
**(3) 이차형식 (quadratic form)** $$\alpha = \mathbf{x}^T\mathbf{A}\mathbf{x} ;\Rightarrow; \frac{\partial \alpha}{\partial \mathbf{x}} = \mathbf{x}^T(\mathbf{A} + \mathbf{A}^T)$$
- 증명: $\alpha = \mathbf{x}'^T\mathbf{A}\mathbf{x}$ 로 두 변수처럼 취급 후 각각 미분, 마지막에 $\mathbf{x}'=\mathbf{x}$ 대입 $$\frac{\partial \alpha}{\partial \mathbf{x}} = \mathbf{x}^T\mathbf{A}^T + \mathbf{x}^T\mathbf{A}$$
- $\mathbf{A}$가 **대칭행렬**이면: $\dfrac{\partial \alpha}{\partial \mathbf{x}} = 2\mathbf{x}^T\mathbf{A}$
- ⚠️ **정오표**: 이 공식이 성립하려면 $\mathbf{A} \in \mathbb{R}^{n\times n}$ (정방행렬)이어야 함. 슬라이드의 $\mathbf{A}\in\mathbb{R}^{m\times n}$ 표기는 오타.
**(4) Elementwise 함수** $$\mathbf{z} = f(\mathbf{x}) ;(\text{원소별 적용}) ;\Rightarrow; \frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \text{diag}\big(f'(\mathbf{x})\big)$$
- 예: $f = \sigma$, $\exp$, $\log$ 등. $z_i$는 $x_i$에만 의존하므로 Jacobian이 대각행렬.
### 2.5 Matrix × column vector, **행렬**에 대한 미분
$\mathbf{z} = \mathbf{W}\mathbf{x}$, $\mathbf{W}\in\mathbb{R}^{m\times n}$일 때 $\dfrac{\partial \mathbf{z}}{\partial \mathbf{W}} = ?$
- **아이디어**: $\mathbf{W}$를 긴 벡터 $\text{vec}(\mathbf{W})$로 펼침 $$\frac{\partial \mathbf{z}}{\partial \mathbf{W}} = \frac{\partial \mathbf{z}}{\partial \text{vec}(\mathbf{W})} \in \mathbb{R}^{m \times mn}$$
- $z_i = \sum_k W_{ik}x_k$ 이므로 $W_{ij}$는 **오직 $z_i$에만** 영향: $$\frac{\partial \mathbf{z}}{\partial W_{ij}} = [0, \cdots, 0, \underbrace{x_j}_{i\text{번째}}, 0, \cdots, 0]^T$$
**Upstream gradient와 결합 (역전파)**
- $\delta^T := \dfrac{\partial J}{\partial \mathbf{z}} = [\delta_1,\cdots,\delta_m] \in \mathbb{R}^{1\times m}$ — Loss가 $\mathbf{z}$까지 흘러온 gradient
- chain rule 결과의 각 블록: $\dfrac{\partial J}{\partial \mathbf{W}_{*j}} = [\delta_1 x_j, \cdots, \delta_m x_j] = x_j,\delta^T$
- unvectorization하면 각 행이 $x_j \delta^T$의 형태 → **외적(outer product)** 으로 정리: $$\boxed{\frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}\cdot\delta^T \in \mathbb{R}^{n\times m} \quad\text{(matrix derivative form: } \delta\cdot\mathbf{x}^T \in \mathbb{R}^{m\times n})}$$
- 핵심: 각 행이 동일 벡터 $\delta^T$에 스칼라 $x_j$만 곱한 구조 → 외적으로 단번에 표현됨
### 2.6 Cross-entropy loss wrt logits
softmax 행렬 표현 ($\mathbf{1}$ = 모든 원소가 1인 벡터, $\mathbf{1}^T\exp(\mathbf{z})=\sum_i e^{z_i}$ 스칼라): $$\text{softmax}(\mathbf{z}) = \frac{\exp(\mathbf{z})}{\mathbf{1}^T\exp(\mathbf{z})}$$
Loss 전개 ($\mathbf{y}^T\mathbf{1}=1$ 이용 → 확률 분포이므로 모두 더하면 1): $$J = -\mathbf{y}^T\log\text{softmax}(\mathbf{z}) = -\mathbf{y}^T\mathbf{z} + \log\big(\mathbf{1}^T\exp(\mathbf{z})\big)$$
미분 (핵심 단계):
1. $\frac{\partial}{\partial\mathbf{z}}\log\mathbf{1}^T\exp(\mathbf{z})$ 에 chain rule → $\text{diag}(1/c)$, $c=\mathbf{1}^T\exp(\mathbf{z})$
2. $\frac{\partial}{\partial\mathbf{z}}\mathbf{1}^T\exp(\mathbf{z}) = \mathbf{1}^T\text{diag}(\exp(\mathbf{z})) = \exp(\mathbf{z})^T$
3. $\text{diag}(1/c)\cdot\mathbf{1}\mathbf{1}^T\cdot\text{diag}(\exp(\mathbf{z}))$ = **모든 행이 $\exp(\mathbf{z})^T/c$ 인 행렬**
4. $\mathbf{y}^T \times (\text{모든 행 동일 행렬}) = (\sum_i y_i)\cdot\text{공통 행} = 1\cdot\text{softmax}(\mathbf{z})^T$
$$\boxed{\frac{\partial J}{\partial \mathbf{z}} = -\mathbf{y}^T + \text{softmax}(\mathbf{z})^T = (\hat{\mathbf{p}} - \mathbf{y})^T}$$
> **직관**: gradient = 예측 확률 − 정답. 매우 깔끔.
### 2.7 Summary (4대 공식)

| 상황                                                  | 결과                                          |
| --------------------------------------------------- | ------------------------------------------- |
| $\mathbf{z}=\mathbf{W}\mathbf{x}$, wrt $\mathbf{x}$ | $\mathbf{W}$                                |
| $\mathbf{z}=\mathbf{W}\mathbf{x}$, wrt $\mathbf{W}$ | $\delta\mathbf{x}^T$ (matrix form)          |
| elementwise $f(\mathbf{x})$                         | $\text{diag}(f'(\mathbf{x}))$               |
| CE loss wrt logits                                  | $(\text{softmax}(\mathbf{z})-\mathbf{y})^T$ |
## 3. Logistic Regression
### 3.1 Binary Classification
- 모델 (선형 모델 + sigmoid): $$p = \sigma(\mathbf{w}^T\mathbf{x}), \qquad \sigma(t)=\frac{1}{1+e^{-t}}, \qquad \sigma' = \sigma(1-\sigma)$$
- 분류 규칙: $p \ge 0.5$ → 1 (positive), 아니면 0 (negative)
- Batch 표현: design matrix로 $\sigma(\mathbf{X}\mathbf{w})$
**Loss: Binary Cross-Entropy** $$J = -\sum_{(\mathbf{x},y)}\Big[y\log\sigma(\mathbf{w}^T\mathbf{x}) + (1-y)\log\big(1-\sigma(\mathbf{w}^T\mathbf{x})\big)\Big]$$
**Gradient 유도 (per-example)** — $s := \sigma(\mathbf{w}^T\mathbf{x})$
- $\frac{\partial}{\partial\mathbf{w}}\log s = (1-s)\cdot\mathbf{x}$
- $\frac{\partial}{\partial\mathbf{w}}\log(1-s) = -s\cdot\mathbf{x}$
- 합치면 ($\pm ys\mathbf{x}$ 상쇄): $$\boxed{\frac{\partial J}{\partial \mathbf{w}} = -\sum_{(\mathbf{x},y)}\big(y - \sigma(\mathbf{w}^T\mathbf{x})\big)\cdot\mathbf{x}}$$
**Matrix form 유도** ($\mathbf{X}$: design matrix) $$\frac{\partial J}{\partial\mathbf{w}} = -\mathbf{y}^T\text{diag}(1-\sigma(\mathbf{Xw}))\mathbf{X} + (\mathbf{1}-\mathbf{y})^T\text{diag}(\sigma(\mathbf{Xw}))\mathbf{X}$$
- $\log(1-\sigma)$ 미분 시 음수가 **두 번** 등장하여 두 번째 항이 양수가 됨
- $\text{diag}(1-\sigma)=I-\text{diag}(\sigma)$ 분배 후 $\pm\mathbf{y}^T\text{diag}(\sigma)\mathbf{X}$ 상쇄: $$= -\mathbf{y}^T\mathbf{X} + \mathbf{1}^T\text{diag}(\sigma(\mathbf{Xw}))\mathbf{X} = \boxed{-\big(\mathbf{y}-\sigma(\mathbf{Xw})\big)^T\mathbf{X}}$$
- 학습: SGD
### 3.2 Multi-class Classification
- Target design matrix의 각 행은 **one-hot 벡터**
- 모델: $\mathbf{p} = \text{softmax}(\mathbf{W}\mathbf{x})$, 분류: $k = \arg\max_i p_i(\mathbf{x})$
- Loss: cross-entropy → gradient는 §2.6 공식 **"CE loss wrt logits"** + **"matrix times vector wrt matrix"** 를 그대로 조합: $$\frac{\partial J}{\partial \mathbf{W}} = \big(\text{softmax}(\mathbf{W}\mathbf{x}) - \mathbf{y}\big)\cdot\mathbf{x}^T$$
## 4. Text Classification
### 4.1 응용 사례
- 스팸 분류, 저자 판별 (Federalist Papers — 1963년 Mosteller & Wallace가 베이지안 방법으로 해결), 감성 분석(영화 리뷰), 주제 분류
### 4.2 텍스트의 feature 표현
**(1) Bag of Words**
- 설명: 단어 등장 여부를 binary로 표현
- 한계: 빈도·순서 무시
**(2) Term Frequency**
- 설명: 단어 등장 빈도를 feature로 사용
- 한계: 순서 무시
**(3) Word Embedding 평균 (Average Pooling)**
- 설명: 문서 내 모든 단어의 임베딩 벡터를 평균하여 문서 벡터로 사용
$$\mathbf{x} = \frac{1}{|\mathcal{D}|}\sum_{w\in\mathcal{D}}\text{Emd}(w)$$
- 한계: 단어 순서·문맥 소실, 모든 단어를 동등하게 취급 (중요 단어 구분 없음)
### 4.3 Generative vs. Discriminative
|       | Discriminative          | Generative                             |
| ----- | ----------------------- | -------------------------------------- |
| 학습 대상 | $p(y\mid\mathbf{x})$ 직접 | $p(\mathbf{x}\mid y)$ 모델링 후 Bayes rule |
| 예시    | Logistic regression, NN | Bayes classifier, Naïve Bayes, GDA     |
### 4.4 Bayes Classifier
$$P(y=k\mid\mathbf{x}) = \frac{P(\mathbf{x}\mid y=k),P(y=k)}{P(\mathbf{x})}, \qquad \hat{y} = \arg\max_k P(y=k\mid\mathbf{x})$$
- **GDA(Gaussian discriminatvie analysis)**: class likelihood를 다변량 정규분포로 모델링
	- $P(\mathbf{x}\mid y=k) = \mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
- **Naïve Bayes 가정**: 클래스가 주어지면 feature들이 **조건부 독립**
    - Naïve Bayes GDA = 대각 공분산 행렬 사용
### 4.5 Multinomial Naïve Bayes (텍스트)
- 텍스트를 단어 시퀀스로 보고, 단어들을 독립적으로 생성: $$P(\mathbf{x}\mid y=k) = \prod_{w_i\in\mathbf{x}} P(w_i\mid y=k)$$
- 직관: 클래스별 "단어 주머니(bag of words)"에서 단어를 독립 추출 (자주 나오는 단어 = 큰 카드)
**MLE 추정과 zero-probability 문제**
- MLE: $P(w\mid y=k) = \frac{\text{count}(w,k)}{\sum_{w'}\text{count}(w',k)}$
- 학습 데이터에 없는 단어 → 확률 0 → **posterior 전체가 0**이 되는 문제
- 해결책 1: **Laplace (additive) smoothing** $$P(w\mid y=k)=\frac{\text{count}(w,k)+1}{N_k+|V|}$$
- 해결책 2: **Word embedding 기반 (semantic smoothing)** — Skip-gram처럼: $$P(w_i\mid y=k) = \frac{\exp(\mathbf{v}_i^T\mathbf{c}_k)}{\sum_{j\in\mathcal{V}}\exp(\mathbf{v}_j^T\mathbf{c}_k)}$$
    - $\exp$는 항상 양수 → **zero probability 원천 차단**
**학습 (conditional log-likelihood)** $$\text{Loss}_{condLL} = -\sum_{(\mathbf{x},k)\in\mathcal{D}}\sum_{w_i\in\mathbf{x}}\log P(w_i\mid y=k) = -\sum_{(\mathbf{x},k)}\sum_{w_i}\text{onehot}_{|\mathcal{V}|}(w_i)^T\log\text{softmax}(\mathbf{V}\mathbf{c}_k)$$
- $\mathbf{V}$: 단어 임베딩 행렬 (각 행 $\mathbf{v}_j^T$), one-hot 내적 = $i$번째 원소 선택
- ⚠️ **정오표**: 슬라이드 마지막 줄에는 $-$ 부호가 누락됨 (위 식이 올바른 형태)
- Gradient: §2의 **CE loss wrt logits** + **matrix × vector wrt matrix** 공식 조합으로 계산
### 4.6 Classifier 평가
**Confusion Matrix**

|              | gold + | gold − |
| ------------ | ------ | ------ |
| **system +** | TP     | FP     |
| **system −** | FN     | TN     |
$$\text{accuracy}=\frac{TP+TN}{TP+FP+TN+FN},\quad \text{precision}=\frac{TP}{TP+FP},\quad \text{recall}=\frac{TP}{TP+FN}$$
**Accuracy의 함정 (불균형 클래스)**
- 예: 100만 포스트 중 100개만 pie 관련 → "전부 not pie" 분류기의 accuracy = 99.99%
- 하지만 목표(파이 애호가 찾기)에는 **완전히 무용** → recall = 0, precision = 정의 불가
- → 불균형 클래스에서는 **precision/recall** 사용
**F-measure** $$F_1 = \frac{2PR}{P+R}$$

- 일반형 (precision/recall의 가중 조화평균): $$F = \frac{(\beta^2+1)PR}{\beta^2 P + R}, \qquad F_1: \beta=1;(\alpha=\tfrac{1}{2})$$
- 조화평균은 작은 값에 민감 → P, R 둘 다 좋아야 F1이 높음