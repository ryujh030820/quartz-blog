---
title: Lecture 6
description: UNIST 나승훈 교수님의 수업 '자연어처리' 6강 PPT 내용에 대해 다룹니다.
date: 2026-06-22
tags:
  - AI
  - Lecture
---
## 1. Language Modeling
### 정의
**Language Model (LM)** 은 다음 단어를 예측하는 확률 분포를 출력하는 시스템이다.
$$P(\mathbf{x}^{(t+1)} \mid \mathbf{x}^{(t)}, \ldots, \mathbf{x}^{(1)})$$
여기서 $\mathbf{x}^{(t+1)}$은 어휘 $V = {\mathbf{w}_1, \ldots, \mathbf{w}_{|V|}}$ 중 임의의 단어일 수 있다.
### 텍스트에 확률 부여
LM을 달리 보면 **텍스트 전체에 확률을 부여하는 시스템**이다.
$$P(\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(T)}) = \prod_{t=1}^{T} P(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}, \ldots, \mathbf{x}^{(1)})$$
각 인수가 바로 LM이 한 스텝에서 제공하는 확률이다.
### 왜 중요한가?
LM은 NLP의 **핵심 부품**이자 **벤치마크 태스크**이다.
- 응용: 예측 타이핑, 음성 인식, 철자·문법 교정, 기계번역, 요약, 대화 등
- ChatGPT를 비롯한 현대 NLP 시스템의 근간
> 충분히 강력한 LM은 trivia, 구문, 공참조, 감성 분석, 추론, 기초 산술까지 next-word prediction만으로 처리할 수 있다.
## 2. n-gram Language Models
### 마르코프 가정 (Markov Assumption)
$\mathbf{x}^{(t+1)}$이 직전 $n-1$개 단어에만 의존한다고 가정한다.

$$P(\mathbf{x}^{(t+1)} \mid \mathbf{x}^{(t)}, \ldots, \mathbf{x}^{(1)}) \approx P(\mathbf{x}^{(t+1)} \mid \mathbf{x}^{(t)}, \ldots, \mathbf{x}^{(t-n+2)})$$
조건부 확률의 정의에 의해:
$$= \frac{P(\mathbf{x}^{(t+1)},\ \mathbf{x}^{(t)},\ \ldots,\ \mathbf{x}^{(t-n+2)})}{P(\mathbf{x}^{(t)},\ \ldots,\ \mathbf{x}^{(t-n+2)})} \approx \frac{\text{count}(\mathbf{x}^{(t+1)},\ \ldots,\ \mathbf{x}^{(t-n+2)})}{\text{count}(\mathbf{x}^{(t)},\ \ldots,\ \mathbf{x}^{(t-n+2)})}$$
코퍼스에서 **빈도를 세어** 추정한다.
### 모델 종류

| 모델               | 수식                                                           | 특징            |
| ---------------- | ------------------------------------------------------------ | ------------- |
| Unigram          | $P(w_1 \cdots w_n) \approx \prod_i P(w_i)$                   | 문맥 무시         |
| Bigram           | $P(w_i \mid w_1 \cdots w_{i-1}) \approx P(w_i \mid w_{i-1})$ | 직전 1단어만 조건    |
| Trigram / 4-gram | 직전 2~3단어 조건                                                  | 품질 향상, 희소성 악화 |
### Bigram 최대우도 추정 (MLE)
$$\hat{P}(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1},\ w_i)}{\text{count}(w_{i-1})}$$
### 희소성 문제 (Sparsity Problems)
**문제 1** — n-gram이 코퍼스에 전혀 등장하지 않으면 확률 = 0  
 → **해결(부분적)**: 스무딩(Smoothing) — 모든 $w \in V$의 카운트에 작은 $\delta$ 추가
**Add-one (Laplace) Smoothing:**
$$P_{\text{Add-1}}(w_i \mid w_{i-1}) = \frac{c(w_{i-1},\ w_i) + 1}{c(w_{i-1}) + V}$$
**문제 2** — (n-1)-gram 자체가 코퍼스에 없으면 어떤 단어도 확률 계산 불가  
 → **해결(부분적)**: 백오프(Backoff) — 더 짧은 문맥으로 후퇴
> $n$을 늘릴수록 희소성 악화. 실제로 $n > 5$는 사용 불가.
### 저장 문제 (Storage Problem)
코퍼스에서 관찰된 모든 n-gram의 카운트를 저장해야 한다. $n$ 증가 또는 코퍼스 증가 → 모델 크기 폭증.
### n-gram LM의 한계
- 장거리 의존성(long-distance dependency) 포착 불가
    - 예: "The computer which I had just put into the machine room on the fifth floor **crashed**."
## 3. Neural Language Models
### 언어 모델 평가: Perplexity
LM의 표준 평가 지표는 **perplexity**이다.
$$\text{perplexity} = \prod_{t=1}^{T} \left(\frac{1}{P_{\text{LM}}(\mathbf{x}^{(t+1)} \mid \mathbf{x}^{(t)}, \ldots, \mathbf{x}^{(1)})}\right)^{1/T}$$
이는 **cross-entropy 손실의 지수**와 동일하다.
$$= \prod_{t=1}^{T}\left(\frac{1}{\hat{\mathbf{y}}^{(t)}_{\mathbf{x}_{t+1}}}\right)^{1/T} = \exp!\left(\frac{1}{T}\sum_{t=1}^{T} -\log \hat{\mathbf{y}}^{(t)}_{\mathbf{x}_{t+1}}\right) = \exp(J(\theta))$$
> **낮을수록 좋다.**
### Fixed-window Neural LM (Bengio et al., 2000/2003)
$$\mathbf{e} = [\mathbf{e}^{(1)};, \mathbf{e}^{(2)};, \mathbf{e}^{(3)};, \mathbf{e}^{(4)}], \quad \mathbf{h} = f(\mathbf{W}\mathbf{e} + \mathbf{b}_1), \quad \hat{\mathbf{y}} = \text{softmax}(\mathbf{U}\mathbf{h} + \mathbf{b}_2)$$
**장점** — 희소성 없음, 관찰된 n-gram 저장 불필요  
**단점** — 고정 윈도우로 문맥 제한, 윈도우 확장 시 $W$ 비례 확장, 입력 위치별 대칭성 없음
→ **임의 길이 입력을 처리할 수 있는 구조가 필요하다.**
## 4. Recurrent Neural Networks (RNN)
### 핵심 아이디어
![[Pasted image 20260622121158.png]]
> **같은 가중치 $W$를 매 타임스텝마다 반복 적용한다.**
### 바닐라 RNN 수식

| 변수         | 수식                                                                                 | 설명                      |
| ---------- | ---------------------------------------------------------------------------------- | ----------------------- |
| 입력 임베딩     | $\mathbf{e}^{(t)} = E\mathbf{x}^{(t)}$                                             | 원-핫 → 밀집 벡터             |
| 프리액티베이션    | $\mathbf{z}^{(t)} = W_h \mathbf{h}^{(t-1)} + W_e \mathbf{e}^{(t)} + \mathbf{b}_1$  | —                       |
| 히든 상태      | $\mathbf{h}^{(t)} = \sigma(\mathbf{z}^{(t)}) = f(\mathbf{z}^{(t)})$                | $\mathbf{h}^{(0)}$은 초기값 |
| 출력 프리액티베이션 | $\mathbf{o}^{(t)} = V\mathbf{h}^{(t)}$                                             | —                       |
| 출력 분포      | $\hat{\mathbf{y}}^{(t)} = \text{softmax}(\mathbf{o}^{(t)}) \in \mathbb{R}^{\|V\|}$ | —                       |
### RNN LM의 장단점
**장점**
- 임의 길이 입력 처리 가능
- 이론상 먼 과거 정보까지 사용 가능
- 입력 길이가 길어져도 모델 크기 불변
- 매 타임스텝 동일 가중치 → 입력 처리의 대칭성
**단점**
- 순환 연산이 느림 (병렬화 어려움)
- 실제로는 먼 과거 정보에 접근하기 어려움 (그래디언트 소실)
### RNN LM 학습
**손실 함수**: 스텝 $t$의 cross-entropy
$$J^{(t)}(\theta) = CE(\mathbf{y}^{(t)},, \hat{\mathbf{y}}^{(t)}) = -\sum_{w \in V} \mathbf{y}^{(t)}_w \log \hat{\mathbf{y}}^{(t)}_w = -\log \hat{\mathbf{y}}^{(t)}_{\mathbf{x}_{t+1}}$$
**전체 손실**:
$$J(\theta) = \frac{1}{T}\sum_{t=1}^{T} J^{(t)}(\theta) = \frac{1}{T}\sum_{t=1}^{T} -\log \hat{\mathbf{y}}^{(t)}_{\mathbf{x}_{t+1}}$$
**Teacher Forcing**: 학습 시 이전 스텝의 예측값이 아닌 **정답 토큰**을 다음 스텝 입력으로 사용한다.
**실용적 학습**: 전체 코퍼스를 한 번에 처리하면 메모리 과부하 → **SGD**로 배치(문장 단위) 처리
## 5. RNN 학습: BPTT (Backpropagation Through Time)
### 핵심 원칙
> **반복 등장하는 가중치에 대한 그래디언트 = 각 등장 위치에서의 그래디언트 합**

$$\frac{\partial J^{(t)}}{\partial W_h} = \sum_{i=1}^{t} \frac{\partial J^{(t)}}{\partial W_h}\Bigg|_{(i)}$$
이는 **다변수 연쇄법칙(Multivariable Chain Rule)** 의 귀결이다. $W_h$가 여러 노드에 공유되므로, 각 노드에서의 기여를 모두 합산해야 한다.
### $W$에 대한 그래디언트 전체 유도
$$\frac{\partial J}{\partial W} = \sum_{t=1}^{T} \frac{\partial J}{\partial \mathbf{h}^{(t)}} \frac{\partial \mathbf{h}^{(t)}}{\partial W} = \sum_{t=1}^{T} \frac{\partial J_{\geq t}}{\partial \mathbf{h}^{(t)}} \frac{\partial \mathbf{h}^{(t)}}{\partial W}$$
$$= \sum_{t=1}^{T} \left(\mu_t^{\mathbf{h}} + \delta_t^{\mathbf{h}}\right) \frac{\partial \mathbf{h}^{(t)}}{\partial \mathbf{z}^{(t)}} \frac{\partial \mathbf{z}^{(t)}}{\partial W}$$
$$= \sum_{t=1}^{T} \mathbf{h}^{(t-1)} \left(\mu_t^{\mathbf{h}} + \delta_t^{\mathbf{h}}\right) \operatorname{diag}\left(f'(\mathbf{z}^{(t)})\right)$$
### 주요 기호 정리

| 기호                                                                             | 정의                                              | 의미                                            |
| ------------------------------------------------------------------------------ | ----------------------------------------------- | --------------------------------------------- |
| $\delta_t^{\mathbf{o}} = \frac{\partial J_t}{\partial \mathbf{o}^{(t)}}$       | $(\hat{\mathbf{y}}^{(t)} - \mathbf{y}^{(t)})^T$ | 출력 프리액티베이션에 대한 국소 그래디언트                       |
| $\delta_t^{\mathbf{h}} = \frac{\partial J_t}{\partial \mathbf{h}^{(t)}}$       | $\delta_t^{\mathbf{o}} V$                       | $J_t$ 하나만의 히든 상태 그래디언트                        |
| $\mu_t^{\mathbf{h}} = \frac{\partial J_{\geq t+1}}{\partial \mathbf{h}^{(t)}}$ | 아래 재귀식                                          | $t$ 이후 미래 손실이 $\mathbf{h}^{(t)}$에 주는 누적 그래디언트 |
### $\mu$의 재귀식 (BPTT의 핵심)
$$\mu_t^{\mathbf{h}} = \left(\mu_{t+1}^{\mathbf{h}} + \delta_{t+1}^{\mathbf{h}}\right) \frac{\partial \mathbf{h}^{(t+1)}}{\partial \mathbf{h}^{(t)}}$$
여기서 히든-히든 야코비안은:
$$\frac{\partial \mathbf{h}^{(t+1)}}{\partial \mathbf{h}^{(t)}} = \frac{\partial \mathbf{h}^{(t+1)}}{\partial \mathbf{z}^{(t+1)}} \frac{\partial \mathbf{z}^{(t+1)}}{\partial \mathbf{h}^{(t)}} = \operatorname{diag}\left(f'(\mathbf{z}^{(t+1)})\right) W$$
**직관**: 시점 $t+1$의 히든 상태에 모이는 그래디언트는 두 출처의 합 — 자기 시점의 국소 손실 신호($\delta$)와 더 먼 미래에서 흘러온 신호($\mu$) — 을 합산한 뒤 야코비안을 통해 한 스텝 더 과거로 전달한다.
### $V$, $U$에 대한 그래디언트
$$\frac{\partial J}{\partial V} = \sum_{t=1}^{T} \mathbf{h}^{(t)} \delta_t^{\mathbf{o}}$$
$$\frac{\partial J}{\partial U} = \sum_{t=1}^{T} \mathbf{x}^{(t)} \left(\mu_t^{\mathbf{h}} + \delta_t^{\mathbf{h}}\right) \operatorname{diag}\left(f'(\mathbf{z}^{(t)})\right)$$
### 실용적 BPTT
전체 시퀀스를 역방향으로 전파하면 비용이 크므로, 실제로는 **약 20 타임스텝에서 절단(truncated BPTT)** 한다.
## 6. RNN의 문제: 그래디언트 소실 · 폭발
### 그래디언트 소실 (Vanishing Gradient)
BPTT에서 시점 $j$의 히든 상태에 대한 그래디언트는:

$$\frac{\partial \mathbf{h}^{(k)}}{\partial \mathbf{h}^{(t)}} = \prod_{j=t+1}^{k} \frac{\partial \mathbf{h}^{(j)}}{\partial \mathbf{h}^{(j-1)}} = \prod_{j=t+1}^{k} \operatorname{diag}\left(f'(\mathbf{z}^{(j)})\right) W$$
$|W| \leq \beta_W$, $|\operatorname{diag}(f')| \leq \beta_h$라 하면:
$$\left|\frac{\partial \mathbf{h}^{(k)}}{\partial \mathbf{h}^{(t)}}\right| \leq (\beta_W \beta_h)^{k-t}$$
- $\beta_W \beta_h < 1$ → **그래디언트 소실** (충분조건)
- $\beta_W \beta_h > 1$ → **그래디언트 폭발** (필요조건)
#### 선형 케이스에서의 증명 스케치 (고유분해)
$\sigma = \text{id}$이면 야코비안이 $W_h$로 단순화된다. $W_h$를 고유분해하면:
$$\frac{\partial J^{(i)}(\theta)}{\partial \mathbf{h}^{(i)}} W_h^\ell = \sum_{i=1}^{n} c_i \lambda_i^\ell \mathbf{q}_i \approx \mathbf{0} \quad (\text{모든 } |\lambda_i| < 1 \text{ 일 때, 큰 } \ell \text{ 에서})$$
$|\lambda_i| < 1$이면 $\lambda_i^\ell \to 0$으로 신호 소멸. 이 조건은 소실의 충분조건이지 필요조건은 아님.
비선형 활성함수의 경우에도 $|\lambda_i| < \gamma$ ($\gamma$는 차원과 $\sigma$에 의존)이면 유사한 결과.
- 증명
	![[Pasted image 20260622121325.png]]
### 왜 문제인가?
- **소실**: 먼 과거에서 오는 그래디언트 신호가 가까운 신호에 압도 → **장기 의존성 학습 불가**
    - 예: "When she tried to print her tickets … she finally printed her **tickets**" — 모델이 먼 거리의 'tickets'를 참조하지 못함
- **폭발**: SGD 업데이트 스텝이 과도하게 커져 파라미터 발산 (Inf/NaN)
### 해결책

| 문제  | 해결책                                 | 비고                                                                                                                                           |
| --- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 폭발  | **Gradient Clipping**               | $\|\hat{\mathbf{g}}\| \geq \text{threshold}$ 이면 $\hat{\mathbf{g}} \leftarrow \frac{\text{threshold}}{\|\hat{\mathbf{g}}\|} \hat{\mathbf{g}}$ |
| 소실  | **LSTM**                            | 별도의 셀 메모리(cell memory)를 통해 정보를 덧셈적으로 보존                                                                                                      |
| 소실  | **Attention, Residual connections** | 직접적이고 선형적인 경로 추가 (Transformer 등)                                                                                                             |
**Gradient Clipping 알고리즘:**
```
g ← ∂ℰ/∂θ
if ‖g‖ ≥ threshold then
    g ← (threshold / ‖g‖) · g
end if
```
직관: 같은 방향으로, 더 작은 보폭으로 이동.