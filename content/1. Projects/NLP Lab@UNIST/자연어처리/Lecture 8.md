---
title: Lecture 8
description: UNIST 나승훈 교수님의 수업 '자연어처리' 8강 PPT 내용에 대해 다룹니다.
date: 2026-07-02
tags:
  - AI
  - Lecture
---
## 0. 배경: 왜 RNN에서 Attention으로?
### 0.1 RNN의 두 가지 근본적 한계
**① Long-range dependency 문제 (linear locality)**
- RNN은 "왼쪽→오른쪽"으로 펼쳐진 순차 구조이므로, 서로 떨어진 두 단어가 상호작용하려면 $O(\text{sequence length})$ 만큼의 스텝(=layer 통과)이 필요함
- 이미지에는 locality(인접 픽셀이 관련성 높음)가 있지만, 언어는 반드시 그렇지 않음 (예: "The chef who … food"에서 chef와 food는 문장상 멀리 떨어져 있어도 강하게 연관)
**② 병렬화의 한계**
- $h_t$는 반드시 $h_{t-1}$ 이후에만 계산 가능 → forward/backward 모두 $O(\text{seq length})$ 만큼의 **순차적** (병렬화 불가능한) 연산이 필요
- GPU/TPU의 병렬 연산 능력을 제대로 활용하지 못함
### 0.2 Cross-Attention → Self-Attention
- 기존 Seq2Seq의 **cross-attention**: 디코더가 $y_t$를 생성할 때 인코더의 입력 $x$ 전체를 참조
- **Self-attention**: 시퀀스 자기 자신에 대해 attention을 적용 — $y_t$를 만들 때 $y_{<t}$(혹은 문맥 전체)를 참조
- 질문: "RNN을 완전히 없애고 attention만으로 시퀀스를 처리할 수 있을까?" → Transformer (Vaswani et al., 2017)
## 1. Transformer의 관점: 문장 = 단어의 "집합(Set)"
- Transformer는 입력을 순서가 있는 시퀀스가 아니라 **집합(set)** 으로 취급 (순서 정보는 나중에 위치 임베딩으로 별도 주입)
- 비유: 문장의 각 단어 = 이미지의 각 픽셀(혹은 패치)
- 각 층은 **Set-to-Set transformation**: 데이터 행렬 $X$를 받아 변환된 행렬 $\tilde X$를 출력
### Self-Attention vs Convolution

| | Self-Attention | Convolutional filter |
|---|---|---|
| 파라미터 공유 | O (모든 위치에 동일 가중치) | O |
| GPU 병렬화 | O | O |
| 연산 범위 | **전역(global)** — 집합 전체에 대해 한 번에 | **지역(local)** — receptive field 내에서만 |
| Long dependency | 한 층만으로 직접 처리 가능 | 여러 층을 쌓아야 간접적으로 처리 |
## 2. Self-Attention의 수식화
### 2.1 Naïve Self-Attention (파라미터 없는 버전)
입력: 토큰 표현 $\mathbf{x}_1, \dots, \mathbf{x}_N$ (열벡터). 세 단계로 구성:
**① Retrieval (관련도 계산)**
$$
e_{ij} = \mathbf{x}_i^\top \mathbf{x}_j
$$
$i$번째 토큰을 **query**로 사용했을 때, $j$번째 토큰(**key** 역할)과의 관련도를 내적으로 계산. $i$번째 행 전체 $[e_{i1},\dots,e_{iN}]$가 $i$번째 토큰이 다른 모든 토큰과 갖는 관련도 벡터(행벡터).
**② Softmax (정규화)**
$$
a_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^N \exp(e_{ik})}
$$
행(row) 단위로 정규화 — 즉 각 query 토큰에 대해 relevance score들이 합이 1이 되도록.
**③ Weighted sum**
$$
\mathbf{y}_i = \sum_{j=1}^N a_{ij}\,\mathbf{x}_j
$$
**행렬 표기**: $X \in \mathbb{R}^{N\times d}$ (각 행이 토큰 하나)라 하면
$$
E = XX^\top,\qquad A = \text{softmax}_{\text{row}}(E),\qquad Y = AX
$$
### 2.2 Standard Self-Attention: Q, K, V 프로젝션 도입
Naïve 버전은 학습 파라미터가 없어 유연성이 부족 → 토큰 유사도를 계산할 때 "어떤 특징에 더 집중할지"를 학습할 수 있도록, **query/key/value용 독립적인 선형변환**을 추가.
**Step 1) Projection**
$$
\mathbf{q}_i = W_Q\mathbf{x}_i, \qquad \mathbf{k}_i = W_K\mathbf{x}_i, \qquad \mathbf{v}_i = W_V\mathbf{x}_i
$$
**Step 2) Retrieval**: $e_{ij} = \mathbf{q}_i^\top \mathbf{k}_j$
**Step 3) Softmax**: $a_{ij} = \text{softmax}_j(e_{ij})$
**Step 4) Weighted sum**: $\mathbf{y}_i = \sum_j a_{ij}\mathbf{v}_j$
**행렬 형태**:
$$
Q = XW_Q,\quad K=XW_K,\quad V=XW_V
$$
$$
\text{output} = \text{softmax}\!\left(QK^\top\right)V = \text{softmax}\!\left(XW_QW_K^\top X^\top\right)XW_V
$$
### 2.3 Scaled Dot-Product Attention
**문제**: 차원 $d$가 커지면 내적 $\mathbf{q}^\top\mathbf{k}$의 값 자체가 (절댓값 기준) 커지는 경향 → softmax의 입력이 매우 커짐 → softmax는 tanh, sigmoid와 같은 squashing function이므로 입력이 클 때 **gradient가 지수적으로 작아짐** (softmax가 거의 one-hot에 가까워지며 saturate).
**증명 (분산 계산)**: $\mathbf{q}, \mathbf{k}\in\mathbb{R}^{d_k}$의 각 성분이 서로 독립이고 평균 0, 분산 1인 확률변수라고 하자. 그러면
$$
\mathbf{q}^\top\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i
$$
각 항 $q_ik_i$는 평균 0, 분산 $\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1$ (독립이고 각각 평균 0이므로 $\text{Var}(q_ik_i)=\mathbb{E}[q_i^2k_i^2]-0=1\cdot1=1$). $d_k$개의 독립항의 합이므로
$$
\text{Var}(\mathbf{q}^\top\mathbf{k}) = \sum_{i=1}^{d_k}\text{Var}(q_ik_i) = d_k
$$
즉 표준편차가 $\sqrt{d_k}$로 커짐 → $\sqrt{d_k}$로 나누어 분산을 다시 1로 되돌림:
$$
\boxed{\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
$$
## 3. Multi-Head Self-Attention
**동기**: 단일 head는 하나의 관련도 패턴만 포착 (예: 문법적 주어, 수식어, 공기(collocation) 관계, 의미적 함의 관계 등 다양한 패턴이 있는데 하나의 $W_Q,W_K$로는 이들을 동시에 담기 어려움).
**정의**: $h$개의 독립적인 projection 세트 $\{W_Q^{(\ell)}, W_K^{(\ell)}, W_V^{(\ell)}\}_{\ell=1}^h$를 둠. 전체 모델 차원 $d$를 $h$등분하여 각 head는
$$
W_Q^{(\ell)}, W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{d\times d/h}
$$
각 head는 독립적으로 attention 수행:
$$
\text{output}^{(\ell)} = \text{softmax}\!\left(\frac{XW_Q^{(\ell)}(W_K^{(\ell)})^\top X^\top}{\sqrt{d/h}}\right)XW_V^{(\ell)} \in \mathbb{R}^{N\times d/h}
$$
전체 head 출력을 **concat** 후, output projection $W_O\in\mathbb{R}^{d\times d}$로 다시 합침:
$$
\text{output} = \big[\text{output}^{(1)};\dots;\text{output}^{(h)}\big]\,W_O \in \mathbb{R}^{N\times d}
$$
**계산 효율성**: $h$개 head를 따로 계산해도 총 연산량은 single-head와 거의 동일 —
- $XW_Q\in\mathbb{R}^{N\times d}$를 계산한 뒤 $\mathbb{R}^{N\times h\times d/h}$로 reshape, 다시 $\mathbb{R}^{h\times N\times d/h}$로 transpose하면 head 축이 마치 배치(batch) 축처럼 작동
- 즉 $Q,K,V$ 전체를 한 번에 projection한 뒤 reshape만으로 $h$개 head를 만들 수 있어, 행렬 크기(파라미터 총량)는 single large head와 동일함
CNN에서 여러 필터를 쓰는 것과 유사한 아이디어: 각 head가 서로 다른 representation subspace에서 서로 다른 관계 패턴을 학습.
## 4. Feed-Forward Network (FFN) Layer
### 4.1 필요성
Self-attention은 (softmax 내부를 제외하면) 본질적으로 **value 벡터들의 가중합**일 뿐, elementwise nonlinearity가 없음 → self-attention만 여러 층 쌓아도 표현력이 제한적 (value의 재평균화만 반복).
**해결**: 각 토큰에 대해 **독립적으로(position-wise)** 적용되는 2층 MLP를 self-attention 뒤에 추가.
### 4.2 정의
$$
\text{FFN}(\mathbf{h}) = W_2\,\text{ReLU}(W_1\mathbf{h}+\mathbf{b}_1) + \mathbf{b}_2
$$

- $W_1: d_{model}\to d_{ff}$ (확장), $W_2: d_{ff}\to d_{model}$ (원래 차원으로 복원)
- 보통 $d_{ff}\gg d_{model}$ (예: GPT류에서 $d_{ff}=4d_{model}$)
## 5. Self-Attention의 3가지 근본 문제와 해결책 (요약표)

| 문제 (Barrier) | 해결 (Solution) |
|---|---|
| 순서(order) 정보가 없음 | 위치 표현(positional representation)을 입력에 추가 |
| Nonlinearity 부재 (그냥 가중평균) | 각 attention 출력 뒤에 FFN 적용 |
| 미래를 보면 안 됨 (MT, LM) | attention score를 $-\infty$로 masking |
## 6. Positional Encoding
### 6.1 필요성
Self-attention은 입력의 **순열(permutation)에 equivariant** — 즉 토큰 순서를 바꿔도 (같이 섞인 채로) 출력도 그대로 섞일 뿐, 순서 정보 자체를 내재적으로 담지 못함. 하지만 언어에서 어순은 중요하므로, 순서 정보를 별도로 주입해야 함.
**방법**: 각 위치 $i$에 대해 위치 벡터 $\mathbf{r}_i\in\mathbb{R}^d$를 정의하고, 단어 임베딩에 단순히 더함:
$$
\tilde{\mathbf{x}}_i = \mathbf{x}_i + \mathbf{r}_i
$$
(concat도 가능하지만 실무에서는 대부분 덧셈을 사용. 보통 첫 layer에서만 주입.)
### 6.2 Sinusoidal Positional Encoding (Vaswani et al., 2017)
서로 다른 주기(period)를 가진 sin/cos 함수들을 이어붙임:
$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$
($pos$는 위치 인덱스, $i$는 차원 인덱스, $L=10000$은 기준 주기 스케일)
**장점**:
- 주기성 → "절대 위치"보다 상대적 패턴을 학습할 여지
- 주기가 순환하므로 (이론적으로는) 학습 때보다 긴 시퀀스로 외삽(extrapolate) 가능해 보임
**단점**:
- 학습되지 않음 (고정된 함수)
- 실제로는 외삽이 잘 작동하지 않음
### 6.3 Learned Absolute Position Embedding
- $\mathbf{r}_i$ 자체를 학습 파라미터로 둠: 행렬 $R\in\mathbb{R}^{d\times n}$을 학습하고, $\mathbf{r}_i$는 그 $i$번째 열
- **장점**: 데이터에 맞게 유연하게 학습됨. 대부분의 실제 시스템이 이 방식 사용
- **단점**: 학습 시 본 위치 $1,\dots,n$ 밖의 인덱스로는 절대 외삽 불가능
- 그 외 대안: relative linear position attention (Shaw et al., 2018), dependency syntax 기반 위치 (Wang et al., 2019)
### 6.4 RoPE (Rotary Position Embedding) — Su et al., 2021 (RoFormer)
**목표**: relative position만을 반영하는 임베딩 함수 $f(\mathbf{x}, i)$를 만들고 싶음:
$$
\langle f(\mathbf{x},i),\, f(\mathbf{y},j)\rangle = g(\mathbf{x},\mathbf{y},\,i-j)
$$
즉 attention score가 **절대 위치가 아닌 상대 위치 $(i-j)$에만** 의존해야 함.
**기존 방법의 한계**:
- Sinusoidal: $i,j$ 각각에 대한 cross-term이 남아 순수하게 $(i-j)$만의 함수로 정리되지 않음
- Learned absolute: 애초에 내적($f(\mathbf{x},i)^\top f(\mathbf{y},j)$)이 $(i-j)$만의 함수가 되도록 보장할 방법이 없음
**RoPE의 아이디어**: 벡터에 상수를 더하는 대신, **위치에 비례하는 각도만큼 회전**시킴 (복소수의 곱셈이 회전에 대응한다는 점에서 착안).
- 2차원의 경우: $f(\mathbf{x},m) = R_{\theta,m}\mathbf{x}$, 여기서 $R_{\theta,m}$은 각도 $m\theta$만큼의 2D 회전행렬
- 회전행렬의 성질 $R_{\theta,m}^\top R_{\theta,n} = R_{\theta,\,n-m}$ 덕분에, 두 회전된 벡터의 내적은 정확히 $(n-m)$에만 의존:
$$
\langle R_{\theta,m}\mathbf{q},\, R_{\theta,n}\mathbf{k}\rangle = \mathbf{q}^\top R_{\theta,\,n-m}\,\mathbf{k}
$$
**고차원으로 확장**: $d$차원 벡터를 2차원씩 짝지어($(x_1,x_2), (x_3,x_4),\dots$) 각 쌍마다 서로 다른 주파수 $\theta_i = 10000^{-2i/d}$ (sinusoidal PE와 같은 주파수 스케일)로 독립적으로 회전 → block-diagonal 회전행렬
**효율적 구현**: 전체 회전행렬을 직접 곱하는 대신(희소행렬이라 낭비), $\cos,\sin$ 값을 미리 계산해두고 원소별(element-wise) 곱셈+덧셈만으로 동일한 결과를 얻음 → $O(d)$ 연산 (matmul 대신).
## 7. Residual Connections (He et al., 2016)
### 7.1 동기: Degradation Problem
- 단순히 층을 깊게 쌓으면 오히려 **training error**가 올라가는 현상 관찰 (overfitting이 아님 — training error 자체가 높아짐)
- He et al.의 사고실험: shallow model에 identity mapping층을 추가해 만든 deep model은 이론적으로 shallow model과 정확히 동일한 함수를 표현할 수 있어야 함 → 이런 deep model의 training error는 shallow model보다 높을 수 없어야 함
- 하지만 실제로는 SGD가 이 identity mapping 구성을 **찾아내지 못함** → 순수 표현력이 아니라 **최적화(optimization)의 문제**
- 증거: plain 56-layer network가 plain 20-layer network보다 **training error도 더 높음** (그래프로 확인 — overfitting이었다면 test error만 높고 training error는 낮아야 함)
### 7.2 Residual Learning
기존: $X^{(i)} = \text{Layer}(X^{(i-1)})$ — 목표 매핑 $H(x)$를 직접 학습
Residual: $X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})$ — 차이(residual) $F(x)=H(x)-x$만 학습
- Identity mapping을 원할 경우, 단순히 $\text{Layer}(\cdot)\equiv 0$이 되도록만 학습하면 됨 → 훨씬 쉬운 최적화 목표
- Residual 경로를 통한 gradient는 정확히 **1** (덧셈의 미분이므로) → gradient vanishing 문제 완화, identity function으로의 편향(bias) 부여
- Loss landscape 시각화(Li et al., 2018)에서도 residual이 있는 네트워크가 훨씬 매끄러운(convex에 가까운) loss surface를 가짐
### 7.3 Transformer에서의 적용
Transformer의 두 서브층(MHSA, FFN) 각각에 residual connection 적용:
$$
X \leftarrow X + \text{MHSA}(X), \qquad X \leftarrow X + \text{FFN}(X)
$$
## 8. Normalization
### 8.1 Batch Normalization (Ioffe & Szegedy, 2015)
**배경 — Internal Covariate Shift**: 학습 중 이전 층의 파라미터 $\Theta_1$이 계속 업데이트되면서, 다음 층(하위 네트워크 $F_2$, 파라미터 $\Theta_2$)에 들어오는 입력 $x$의 분포도 계속 변함. $F_2$는 $x$의 분포가 고정되어 있을 때 더 효율적으로 학습되므로, 이 분포 변화는 학습을 방해함.
**Whitening (LeCun et al., 1998)**: 입력을 평균 0, 분산 1로(가능하면 decorrelate까지) 만들면 수렴이 빨라진다는 아이디어.
#### Naïve Whitening의 함정

$$
\tilde{\mathbf{x}} = \frac{\mathbf{x}-sg(\mathbb{E}[\mathbf{x}])}{sg(\sqrt{\text{Var}[\mathbf{x}]})+\epsilon}
$$
여기서 $sg(\cdot)$는 **stop-gradient**: forward에서는 값을 그대로 쓰지만, $\partial\, sg(v)/\partial\theta \equiv 0$으로 취급 — 즉 평균·분산 같은 통계량이 사실은 파라미터의 함수임에도 이를 "상수"로 간주.
**문제**: 이렇게 하면 gradient descent 업데이트가 "정규화가 일어난다"는 사실 자체를 반영하지 못함 → 파라미터를 업데이트해도, whitening 통계량이 그 변화를 그대로 다시 상쇄해버려서 **whitened activation 값이 업데이트 전후로 거의 변하지 않는** 현상 발생 (gradient step이 사실상 무효화됨).
#### 진짜 Batch Normalization
$\mu,\sigma$가 파라미터에 의존한다는 사실을 **computational graph에 포함**시켜, 그 경로로도 gradient가 흐르게 함 (stop-gradient 제거). 또한 항등함수(identity)를 표현할 수 있도록 학습 가능한 scale/shift 파라미터 $\gamma,\beta$를 추가:
$$
\hat{x}_k = \frac{x_k - \mathbb{E}[x_k]}{\sqrt{\text{Var}[x_k]+\epsilon}}, \qquad y_k = \gamma_k\hat{x}_k+\beta_k
$$
**미니배치 근사**: SGD에서는 전체 데이터셋이 아닌 미니배치 단위로 학습하므로, 각 미니배치의 평균 $\mu_\mathcal{B}$, 분산 $\sigma_\mathcal{B}^2$로 $\mathbb{E}[x],\text{Var}[x]$를 근사. 학습 중엔 미니배치마다 새로 계산하고, **테스트 시**엔 여러 미니배치에 대해 집계한 값(전역 통계)을 사용.
#### 추론 시(inference) 분산 보정 — Bessel's Correction
$$
\mathbb{E}[x]\leftarrow \mathbb{E}_\mathcal{B}[\mu_\mathcal{B}], \qquad \text{Var}[x]\leftarrow \frac{m}{m-1}\mathbb{E}_\mathcal{B}[\sigma_\mathcal{B}^2]
$$

미니배치 분산 $\sigma_\mathcal{B}^2=\frac1m\sum_i(x_i-\mu_\mathcal{B})^2$는 **biased estimator**로, $\mathbb{E}[\sigma_\mathcal{B}^2]=\frac{m-1}{m}\sigma^2$ ($\mu_\mathcal{B}$ 자신이 표본으로부터 추정된 값이라 자유도를 하나 잃음). 이를 보정하기 위해 역수 $\frac{m}{m-1}$을 곱해 unbiased estimator로 만듦.
**Inference-time BN 변환**: $N^{\inf}_{BN}$에서 $y=\text{BN}_{\gamma,\beta}(x)$를 다음으로 대체:
$$
y = \frac{\gamma}{\sqrt{\text{Var}[x]+\epsilon}}\cdot x + \left(\beta-\frac{\gamma\,\mathbb{E}[x]}{\sqrt{\text{Var}[x]+\epsilon}}\right)
$$
(이는 각 배치마다 통계량을 계산할 필요 없이, $x$에 대한 하나의 고정된 affine 변환으로 정리한 형태.)
### 8.2 Layer Normalization (Ba et al., 2016)
#### RNN에서 BN을 쓰기 어려운 이유
- FNN에서는 각 층마다 통계량을 따로 저장하면 되지만, **RNN은 시퀀스 길이가 가변적**이라 시점(timestep)마다 통계량이 달라야 함
- 학습 시 본 것보다 긴 테스트 시퀀스에서 문제 발생 (해당 timestep의 통계량이 없음)
- 미니배치 내에서도 시퀀스 길이가 제각각이라, 각 timestep에서 통계량을 계산하는 데 쓰이는 샘플 수가 들쭉날쭉함
#### 정의
배치가 아니라 **같은 층의 hidden unit들 전체**에 대해 평균·분산을 계산 (한 샘플 내에서 완결됨 → 배치/시퀀스 길이와 무관):
주어진 활성화 벡터 $\mathbf{x}=[x_1,\dots,x_D]^\top$에 대해
$$
\mu = \frac1D\sum_{i=1}^D x_i, \qquad \sigma = \sqrt{\frac1D\sum_{i=1}^D (x_i-\mu)^2}
$$
Parametric한 최종형태 (gain $\mathbf{g}$, bias $\mathbf{b}$는 학습 파라미터):
$$
\mathbf{y} = f\!\left[\frac{\mathbf{g}}{\sigma}\odot(\mathbf{x}-\mu)+\mathbf{b}\right]
$$
#### 왜 잘 되는가 — Gradient Normalization (Xu et al., 2019)
기존 통념(forward normalization 자체가 원인)과 달리, Xu et al.은 **backward pass에서 $\mu,\sigma$를 통해 흐르는 gradient**가 핵심이라고 보임:
- **Re-centering**: $x$ 전체에 상수 $c$를 더해도 $\mu,\sigma$가 자동으로 상쇄하므로 $L$이 불변 → $\dfrac{d}{dc}L(x+c\mathbf1)\big|_{c=0}=\sum_i\dfrac{\partial L}{\partial x_i}=0$. 즉 $x$에 대한 gradient는 **항상 정확히 평균 0**이 되도록 자동 강제됨.
- **Re-scaling**: chain rule에서 $1/\sigma$ 인자가 곱해져, activation 크기가 커질수록 gradient가 자동으로 작아짐 (gradient explosion 억제).
- **실증(DetachNorm)**: forward는 LayerNorm과 동일하되 backward에서만 $\mu,\sigma$에 stop-gradient를 거는 ablation을 하면 (= naive whitening과 동일한 아이디어), forward 분포가 같음에도 성능이 눈에 띄게 나빠짐 → forward normalization 자체보다 backward의 gradient normalization이 핵심이라는 증거.
- 실험적으로는 re-scaling 효과가 re-centering보다 성능 기여가 더 큼.
- (추가 발견) LayerNorm의 gain/bias 파라미터는 오히려 과적합을 유발하는 경우가 많아, 이를 제거한 "LayerNorm-simple"이 더 나은 성능을 보이기도 함.
#### Invariance 성질
**① Weight re-scaling & re-centering**: $W'=\delta W+\mathbf{1}\boldsymbol\gamma^\top$ (전체 가중치 행렬을 동일하게 스케일링+shift)에 대해 불변:
$$
\mathbf{h}' = f\!\left(\frac{\mathbf{g}}{\sigma'}(W'\mathbf{x}-\boldsymbol\mu')+\mathbf{b}\right) = f\!\left(\frac{\mathbf{g}}{\sigma}(W\mathbf{x}-\boldsymbol\mu)+\mathbf{b}\right) = \mathbf{h}
$$
(단, **개별 유닛 하나만** 스케일링하는 경우는 불변이 아님 — $\mu,\sigma$가 $D$개 유닛 전체의 공통 통계량이므로, 전체를 동일하게 바꿔야만 정확히 상쇄됨.)
**② Data re-scaling & re-centering**: 한 학습 샘플 $\mathbf{x}\to\delta\mathbf{x}$에 대해 불변 (LayerNorm의 $\mu,\sigma$는 **그 샘플 자신**의 통계량이라, 샘플이 스케일되면 통계량도 같이 스케일되어 상쇄됨):
$$
h_i' = f\!\left(\frac{g_i}{\sigma'}(\mathbf{w}_i^\top\mathbf{x}'-\mu')+b_i\right) = h_i
$$
- **BatchNorm은 이 성질이 없음** — $\mu,\sigma$가 미니배치 내 **여러** 샘플에 걸쳐 계산되므로, 한 샘플만 스케일링하면 그 샘플의 통계량이 배치 전체 통계량과 어긋나 상쇄되지 않음. 이 차이가 LayerNorm이 가변 길이 시퀀스(RNN, Transformer)에 더 적합한 이유 중 하나.
### 8.3 BatchNorm vs LayerNorm 비교

|                           | BatchNorm              | LayerNorm             |
| ------------------------- | ---------------------- | --------------------- |
| 통계량 계산 축                  | 배치 내 여러 샘플 (같은 채널/유닛)  | 한 샘플 내 여러 hidden unit |
| 가변 길이 시퀀스                 | 부적합 (timestep별 통계 불안정) | 적합                    |
| 테스트 시                     | 별도의 저장된 전역 통계 필요       | 불필요 (샘플 자체로 완결)       |
| 데이터 re-scaling invariance | 없음                     | 있음                    |
### 8.4 Transformer sublayer 적용 (Add & Norm)
Residual connection과 LayerNorm을 함께 적용:
$$
X \leftarrow \text{LayerNorm}\big(X + \text{Sublayer}(X)\big)
$$
## 9. Transformer Layer 조립 및 Pre-Norm vs Post-Norm
- 하나의 Transformer layer = **MHSA (+ Add & Norm) → FFN (+ Add & Norm)**
- **Post-Norm** (원 논문 방식): $X \leftarrow \text{LayerNorm}(X+\text{Sublayer}(X))$ — 서브층 출력 이후에 정규화
- **Pre-Norm** (현대 LLM에서 더 흔히 쓰임): $X \leftarrow X + \text{Sublayer}(\text{LayerNorm}(X))$ — 서브층에 들어가기 *전*에 정규화
- **동기**: Post-Norm은 층이 깊어질수록 학습이 불안정해지는 경향이 있는 반면, Pre-Norm은 residual stream이 정규화 없이 그대로 유지되어(gradient가 identity 경로로 그대로 흐름) 훨씬 안정적으로 깊은 모델을 학습할 수 있음 (참고: arXiv:2002.04745, arXiv:2412.13795)
## 10. Masking: 미래를 보지 못하게 하기 (Causal Attention)
**필요성**: 디코더(기계번역, 언어모델)에서 $y_t$를 예측할 때 $y_{>t}$를 미리 봐서는 안 됨.
**비효율적인 방법**: 매 timestep마다 과거 단어들로만 key/query 집합을 다시 구성 — 병렬화 불가능.
**해결 (Masked / Causal Self-Attention)**: 전체 시퀀스를 한 번에 병렬로 처리하되, attention score 행렬에서 미래 위치에 해당하는 pre-softmax 값을 $-\infty$로 설정:
$$
e_{ij} = \begin{cases} \mathbf{q}_i^\top\mathbf{k}_j/\sqrt{d_k} & j\le i \\ -\infty & j>i\end{cases}
$$
softmax를 취하면 $j>i$인 위치의 attention weight는 정확히 0이 됨 → 각 토큰이 이전 토큰들에 대해서만(그리고 자기 자신까지만) attend.
## 11. Transformer 아키텍처 3종

| 구조 | 대표 모델 | Self-attention masking | 용도 |
|---|---|---|---|
| **Decoder-only** | GPT | Causal (masked) | 자기회귀 언어모델, 생성 |
| **Encoder-only** | BERT | 없음 (fully-visible) | 양방향 문맥 이해, 분류/태깅 |
| **Encoder-Decoder** | T5, 원조 Transformer | 인코더: 없음 / 디코더: causal + cross-attn | 번역, 요약 등 seq2seq |
### 11.1 Decoder Transformer (GPT류)
- Transformer layer들을 순수하게 쌓은 자기회귀 언어모델: $\mathbf{h}^{(l)} = f^{(l)}(\mathbf{h}^{(l-1)})$
- 최종 층에서 vocabulary 크기 $K$에 대한 다항 로지스틱회귀(=softmax classifier)로 다음 토큰 확률 예측
- 손실함수: cross-entropy
$$
J = -\sum_{k=1}^K y_k\log\hat y_k, \qquad \hat{\mathbf y}=\text{softmax}(\text{LSM}(\mathbf h))
$$
('LSM' = Linear-Softmax; 모든 위치에서 파라미터를 공유하는 linear layer + softmax)
### 11.2 Encoder Transformer (BERT류)
- Decoder와 유일한 차이: **masking을 제거** — 모든 출력 위치가 입력 전체(양방향)에 attend 가능
- Fully-visible mask: 매 timestep에서 전체 입력을 볼 수 있음
### 11.3 Encoder-Decoder Transformer
- **인코더**: 입력 문장을 self-attention(마스킹 없음) 층들로 인코딩 → $Z$ (또는 $H=[\mathbf h_1;\dots;\mathbf h_T]$)
- **디코더**: $Z$가 주어졌을 때 조건부 언어모델 $P(\mathbf y|\mathbf x)$ — self-attention(causal) + **cross-attention**으로 구성
#### Cross-Attention 상세
self-attention은 query/key/value가 모두 같은 출처에서 나오지만, cross-attention은 출처가 다름:
- **Key, Value**: 인코더 출력에서 (기억(memory) 역할): $\mathbf k_i = K\mathbf h_i,\ \mathbf v_i=V\mathbf h_i$
- **Query**: 디코더 내부에서: $\mathbf q_i = Q\mathbf z_i$
행렬형태 ($H=[\mathbf h_1;\dots;\mathbf h_T]\in\mathbb R^{T\times d}$, $Z=[\mathbf z_1;\dots;\mathbf z_T]\in\mathbb R^{T\times d}$):
$$
\text{output} = \text{softmax}\big(ZQ(HK)^\top\big)\,HV \ \in\mathbb R^{T\times d}
$$

- Cross-attention도 multi-head로 확장 가능 (query는 decoder에서, key/value는 encoder에서 유래)
- 디코더 블록 구성: **1) Masked Multi-head Self-Attention → 2) Multi-head Cross-Attention → 3) FFN** (각각 residual + LayerNorm 수반)
## 12. Transformer의 성과
- **기계번역**: 원 논문에서 기존 대비 더 높은 BLEU 점수 + 더 효율적인 학습 (WMT 2014 En-De, En-Fr)
- **장문 생성**: WikiSum 데이터셋 문서 생성에서 기존 방법 대비 우수 (Liu et al., 2018)
- **Pretraining과의 결합**: Transformer의 병렬화 가능성 덕분에 대규모 pretraining이 효율적으로 가능해짐 → 사실상 현재 NLP 벤치마크 상위권 모델 대부분이 Transformer+pretraining 기반
## 13. Transformer의 한계와 최신 변형
### 13.1 개선하고 싶은 두 가지
1. **학습 불안정성** (Pre-norm vs Post-norm 논의, 위 9절 참고)
2. **Self-attention의 이차(quadratic) 연산 비용**
### 13.2 Quadratic Cost 상세
$$
XW_Q(XW_K)^\top \in \mathbb R^{n\times n}
$$
모든 위치쌍(all pairs) 간의 상호작용을 계산해야 하므로 총 연산량 $O(n^2 d)$ ($n$=시퀀스 길이, $d$=차원)
- RNN은 $O(n)$ (순차적이라 병렬화는 안 되지만 연산량은 선형)
- $d\approx 1000$ 정도라 할 때, 짧은 문장($n\le 30$)에서는 $n^2\le 900$이라 문제 없지만, 실무에서는 $n=512$ 정도로 제한하는 경우가 많음. 문서 단위($n\ge50{,}000$)로 가면 이차 비용이 크게 부담됨.
## Backlinks
- [[[논문 리뷰] Attention Is All You Need]]
- [[[개념 정리] Rotary Positional Embedding(RoPE)]]