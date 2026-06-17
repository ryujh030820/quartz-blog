---
title: Lecture 4
description: UNIST 나승훈 교수님의 수업 '자연어처리' 4강 PPT 내용에 대해 다룹니다.
date: 2026-06-17
tags:
  - AI
  - Lecture
---
## 1. Neural Classification
### 1.1 NER (Named Entity Recognition)
- 텍스트에서 **개체명을 찾고 분류**하는 태스크
- 레이블 예시: `PER` (인물), `LOC` (장소), `DATE` (날짜) 등
```
Samuel Quinn was arrested in the Hilton Hotel in Paris in April 1989.
PER    PER                   LOC LOC        LOC  DATE DATE
```
**활용 예:**
- 문서 내 특정 엔티티 추적
- Question Answering (정답은 대부분 개체명)
- Wikidata 같은 Knowledge Base와의 Entity Linking
### 1.2 Window Classification
- **아이디어**: 각 단어를 **주변 문맥 윈도우**와 함께 분류
- 윈도우 내 단어 벡터를 **concatenation** → 분류기 입력
$$\mathbf{x}_{\text{window}} = [x_{\text{museums}},\ x_{\text{in}},\ x_{\text{Paris}},\ x_{\text{are}},\ x_{\text{amazing}}]^T \in \mathbb{R}^{5d}$$
- 각 단어 위치마다 classifier를 실행
### 1.3 Neural Classification의 특징

| 구분    | 전통적 Softmax | Neural Network            |
| ----- | ----------- | ------------------------- |
| 학습 대상 | $W$ (가중치)만  | $W$ + **단어 벡터** (표현까지 학습) |
| 결정 경계 | 선형          | 비선형 (다층 구조)               |
| 입력 표현 | 희소 기호 특징    | 밀집 분산 표현                  |
- **Embedding layer**: $x = Le$ (one-hot → 밀집 벡터)
- 다층 구조가 선형 분리 불가능한 데이터를 분리 가능하게 변환
### 1.4 비선형 활성화 함수

|함수|수식|특징|
|---|---|---|
|Sigmoid|$\sigma(z) = \frac{1}{1+e^{-z}}$|출력 [0,1], 확률|
|tanh|$\tanh(z) = 2\sigma(2z) - 1$|출력 [-1,1], sigmoid의 2배 기울기|
|ReLU|$\max(z, 0)$|빠른 학습, gradient 흐름 우수|
|Leaky/Parametric ReLU|$\max(\alpha z, z)$|ReLU의 dead zone 완화|
|Swish|$x \cdot \sigma(x)$|arXiv:1710.05941|
|GELU|$x \cdot P(X \leq x),\ X \sim \mathcal{N}(0,1) \approx x \cdot \sigma(1.702x)$|Transformer에서 주로 사용 (BERT, RoBERTa)|
> **왜 비선형 함수가 필요한가?** 비선형성 없이는 $W_1 W_2 x = Wx$처럼 아무리 깊어도 선형 변환이 됨. 비선형 함수를 넣으면 임의의 복잡한 함수를 근사 가능 (Universal Approximation).
### 1.5 Cross Entropy Loss
**목표**: 정답 클래스 $y$의 확률을 최대화 = 음의 로그확률 최소화
$$H(p, q) = -\sum_{c=1}^{C} p(c) \log q(c)$$
- $p$: 정답 분포 (true), $q$: 모델 예측 분포 (predicted)
**One-hot $p$ 적용 시:**
$$p = [0, \ldots, 0, \underbrace{1}_{y_i}, 0, \ldots, 0]$$
- $p(c) = 0$인 항은 모두 소거 → **하나의 항만 생존**
$$\boxed{J = -\log p(y_i \mid x_i)}$$
> **직관**: 정답 클래스에 모델이 부여한 확률이 낮을수록 loss가 커짐. PyTorch에서 `nn.CrossEntropyLoss()`로 사용.
### 1.6 Deep Network의 필요성
- **경험적 근거**: 깊은 네트워크가 더 나은 일반화 성능 (Goodfellow et al. '14)
- **이론적 근거**: Rectifier Network는 층이 깊어질수록 **지수적으로 많은 선형 구역** 생성
    - 각 hidden unit이 입력 공간을 folding하는 역할
    - 마지막 hidden layer에서 선형 분리 가능해짐
## 2. Stochastic Gradient Descent (SGD)
### 2.1 학습 기본 구조
- **Training data**: ${x_i, y_i}_{i=1}^{N}$
- **Parameters** $\theta$: 가중치 행렬 + 데이터 표현 (word vectors)
- **Objective**: Loss function 최소화 (Cross-entropy / Squared error)
- **Regularization**: Overfitting 방지
$$J(\theta) = \frac{1}{N}\sum_{i=1}^{N} L(f(x_i; \theta), y_i) + \lambda R(\theta)$$
### 2.2 Gradient Descent 방식 비교

|방식|특징|단점|
|---|---|---|
|Batch GD|전체 데이터 사용|매우 느림|
|Mini-batch SGD|일부 $m$개 샘플 사용|추정 노이즈|
|SGD|샘플 1개씩|매우 노이지|
### 2.3 Mini-batch SGD 알고리즘
1. $m$개 샘플 $T_m = {(x, y)}_{i=1}^{m}$ 랜덤 샘플링
2. 미니배치 loss 계산 (NLL 기준):
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m} \log p(y_i \mid x_i; \theta)$$
3. 기울기 계산: $\nabla_\theta J(\theta)$
4. 파라미터 업데이트:
$$\theta^{\text{new}} = \theta^{\text{old}} - \alpha \nabla_\theta J(\theta^{\text{old}})$$
$$\theta_j^{\text{new}} = \theta_j^{\text{old}} - \alpha \frac{\partial J}{\partial \theta_j}\bigg|_{\theta^{\text{old}}}$$
- $\alpha$: learning rate (step size)
- $\theta$에는 word vectors도 포함됨!
## 3. Matrix Calculus — Gradient by Hand
### 3.1 기본 개념
**스칼라 → 스칼라:**
$$f(x) = x^3 \implies \frac{df}{dx} = 3x^2$$
**스칼라 → 벡터 (Gradient):**
$$\nabla_x f = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]^T$$
**벡터 → 벡터 (Jacobian):**
$$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m \implies \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n},\quad \left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)_{ij} = \frac{\partial f_i}{\partial x_j}$$
### 3.2 Chain Rule
**1변수:**
$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$
**다변수 (행렬):**
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \frac{\partial \mathbf{z}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

### 3.3 주요 Jacobian 공식
**Elementwise 활성화 함수** $\mathbf{h} = f(\mathbf{z})$:
$$\frac{\partial \mathbf{h}}{\partial \mathbf{z}} = \text{diag}(f'(z_1), f'(z_2), \ldots, f'(z_n))$$
- $n \times n$ 대각행렬
**행렬-벡터 곱** $\mathbf{z} = W\mathbf{x} + \mathbf{b}$:
$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = W, \quad \frac{\partial \mathbf{z}}{\partial \mathbf{b}} = I, \quad \frac{\partial \mathbf{z}}{\partial W} = ?$$
**내적** $s = \mathbf{u}^T \mathbf{h}$:
$$\frac{\partial s}{\partial \mathbf{u}} = \mathbf{h}^T, \quad \frac{\partial s}{\partial \mathbf{h}} = \mathbf{u}^T$$
### 3.4 NER 네트워크 예시 Gradient 계산
**네트워크 구조:**
$$s = \mathbf{u}^T \mathbf{h}, \quad \mathbf{h} = f(\mathbf{z}), \quad \mathbf{z} = W\mathbf{x} + \mathbf{b}$$
**Chain rule 적용 (upstream gradient 재사용):**
$$\delta = \frac{\partial s}{\partial \mathbf{z}} = \frac{\partial s}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial \mathbf{z}} = \mathbf{u}^T \cdot \text{diag}(f'(\mathbf{z})) = \mathbf{u}^T \odot f'(\mathbf{z})^T$$
$\delta$를 **error signal** (upstream gradient)이라 하면:
$$\frac{\partial s}{\partial \mathbf{b}} = \delta, \quad \frac{\partial s}{\partial W} = \delta^T \mathbf{x}^T, \quad \frac{\partial s}{\partial \mathbf{x}} = W^T \delta^T$$
> **핵심**: $\delta$를 한 번만 계산하고 재사용 → 중복 계산 제거!
### 3.5 Shape Convention

| 형태                   | 설명                                      |
| -------------------- | --------------------------------------- |
| **Jacobian form**    | Chain rule 계산에 편리                       |
| **Shape convention** | gradient 형태 = parameter 형태 (SGD 구현에 편리) |
- 최종적으로 **shape convention**에 맞게 transpose 조정
- $\frac{\partial s}{\partial W}$ → $n \times m$ (W와 동일한 형태)
- $\frac{\partial s}{\partial \mathbf{b}}$ → column vector (b와 동일한 형태)
> **Transpose 이유**: 각 입력이 각 출력에 영향 → outer product 형태
## 4. Backpropagation
### 4.1 핵심 원리
> 체인룰을 Computation Graph 위에서 **효율적으로** 재귀 적용
$$[\text{downstream gradient}] = [\text{upstream gradient}] \times [\text{local gradient}]$$
### 4.2 단일 노드의 Backprop
```
[downstream] ←─ [노드] ←─ [upstream]
                  ↕
              [local grad]
```
- **Local gradient**: 해당 노드의 출력을 입력으로 미분
- **Upstream gradient**: 최종 출력에서 이 노드의 출력까지의 gradient
- **Downstream gradient** = upstream × local
**입력이 여러 개인 경우**: 각 입력에 대한 local gradient 별도 계산
### 4.3 노드별 직관

| 노드       | 역할  | Gradient 행동                           |
| -------- | --- | ------------------------------------- |
| `+` (덧셈) | 분배  | upstream gradient를 **모든 입력에 동일하게 전달** |
| `max`    | 라우팅 | upstream gradient를 **최댓값 방향으로만** 전달   |
| `*` (곱셈) | 스위칭 | upstream gradient × **반대쪽 입력값**       |
### 4.4 Gradients Sum at Branches
노드의 출력이 여러 방향으로 나가는 경우:
$$\frac{\partial L}{\partial x} = \sum_{\text{branch}} \frac{\partial L}{\partial \text{branch}} \cdot \frac{\partial \text{branch}}{\partial x}$$
- 각 branch에서 오는 gradient를 **합산**
### 4.5 Feed-forward Network Backprop 전체 흐름
**Forward propagation:**
$$\mathbf{x} \to \mathbf{z}^{(1)} = W\mathbf{x} + \mathbf{b} \to \mathbf{h} = f(\mathbf{z}^{(1)}) \to \mathbf{o} = U\mathbf{h} \to \hat{\mathbf{y}} = \text{softmax}(\mathbf{o}) \to J$$
**Backward propagation:**
1. **초기화**: $\frac{\partial J}{\partial \mathbf{o}} = \hat{\mathbf{y}} - \mathbf{y}$ (softmax + cross-entropy)
2. **Output layer** $\to$ **Hidden layer**:

$$\boldsymbol{\delta}_z = \frac{\partial J}{\partial \mathbf{z}} = W^T \boldsymbol{\delta}_{\text{next}} \odot f'(\mathbf{z})$$
3. **가중치 gradient**:
$$\frac{\partial J}{\partial W} = \boldsymbol{\delta}_z \cdot \mathbf{h}^T, \quad \frac{\partial J}{\partial \mathbf{b}} = \boldsymbol{\delta}_z$$
4. **이전 층으로 전파**:
$$\boldsymbol{\delta}_h = W^T \boldsymbol{\delta}_z$$
### 4.6 효율성: 중복 계산 제거
**❌ 잘못된 방식**: $\frac{\partial s}{\partial \mathbf{b}}$와 $\frac{\partial s}{\partial W}$를 독립적으로 계산
- 두 경로 모두 $\bullet \to f$ 구간의 gradient를 중복 계산
**✅ 올바른 방식**: $\delta$ (upstream gradient)를 **한 번 계산 후 재사용**
- Forward → Backward 한 번만 통과하면서 모든 gradient 동시 수거
### 4.7 General Computation Graph에서의 Backprop
1. **Forward pass**: Topological sort 순서로 노드 방문 → 값 계산 및 중간값 저장
2. **Backward pass**: 역순으로 방문 → 각 노드에서 gradient 계산

$$\frac{\partial J}{\partial x_i} = \sum_{j \in \text{successors}} \frac{\partial J}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$
> **Big-O 복잡도**: Forward와 Backward 모두 동일!
## 5. 구현 및 자동 미분
### 5.1 Automatic Differentiation
- Forward pass의 symbolic expression에서 **자동으로 gradient 계산** 가능
- 각 노드 타입이 알아야 할 것:
    - 출력 계산 방법 (forward)
    - 입력에 대한 gradient 계산 방법 (backward)
- **PyTorch, TensorFlow** 등이 이를 자동 처리
### 5.2 Forward/Backward API
각 레이어/노드는:
- `forward()`: 입력 → 출력, 중간값 저장
- `backward()`: upstream gradient 받아 → local gradient 곱해 → downstream 반환
### 5.3 Numeric Gradient Check
$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + h \cdot e_i) - f(\theta - h \cdot e_i)}{2h}, \quad h \approx 10^{-4}$$
- 구현 검증에 유용
- 매우 느림 (파라미터마다 2번의 forward pass 필요)