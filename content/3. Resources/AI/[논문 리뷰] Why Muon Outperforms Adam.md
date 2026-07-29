---
title: "[논문 리뷰] Why Muon Outperforms Adam"
date: 2026-07-28
tags:
  - AI
  - Paper
---
## Introduction
Muon은 거대 언어 모델(LLM) 사전 학습에서 Adam의 강력한 대안으로 부상했다.
이 알고리즘은 기울기 모멘텀 행렬을 스펙트럼 정규화하여 매개변수의 행렬 구조를 활용하며, 결과적으로 0이 아닌 특이값들을 동일한 규모로 설정한다.
이러한 행렬 인식 설계 덕분에 Muon은 다양한 모델 규모의 LLM 사전 학습에서 Adam보다 최대 약 2배 빠른 학습 속도를 달성할 수 있다.

본 연구는 최적화 환경의 관점에서 Muon의 이점을 이해하기 위한 첫걸음을 내딛는다. 구체적으로 다음과 같은 질문을 던진다.
1. 어떤 손실 지형(landscape) 특성이 Adam 대비 Muon의 이점을 뒷받침하는가?
2. 학습 데이터 및 모델 구조와 같은 사전 학습 요인들이 이 특성에 어떤 영향을 미치는가?

첫 번째 질문에 답하기 위해, 본 연구는 학습 손실의 **2차 테일러 전개**를 통해 Muon과 Adam의 1단계 손실 감소량을 분석한다.
이 전개가 실현된 손실 감소량을 정확하게 예측함을 확인했으며, 두 최적화 알고리즘이 유사한 1차 이득을 얻는 반면, Muon은 훨씬 더 작은 곡률 패널티를 발생시킨다는 점을 밝혀냈다.
또한 곡률 패널티를 <strong>업데이트 노름(update norm)</storng>과 정규화된 <strong>방향성 선명도(Normalized Directional Sharpness, NDS)</strong>의 기여도로 추가 분해하였으며, Muon의 더 작은 곡률 패널티는 업데이트 방향에 의해 결정되는 더 낮은 NDS에서 비롯됨을 보여주었다.

두 번째 질문에 답하기 위해, 본 연구는 학습 데이터와 모델 구조가 Muon의 더 낮은 NDS에 어떻게 기여하는지 조사한다.
먼저 제어된 불균형 수준을 가진 Zipf-PCFG에서 생성된 합성 데이터를 학습시켜 데이터 불균형의 역할을 검토한다. 데이터셋이 더 불균형해질수록 Adam 대비 Muon의 NDS 이점이 더 커짐을 발견하였다.
데이터 관점을 보완하기 위해, 모델 구조 관점으로 전환하여 NDS를 **레이어 내(within-layer)** 기여도와 **레이어 간(cross-layer)** 기여도로 분해한다.
이 분해 결과는 Muon의 레이어 간 기여도가 학습 중에 빠르게 감소함을 보여주며, 따라서 사전 학습의 중후반 단계에서 Muon의 더 작은 NDS는 주로 더 작은 레이어 내 NDS에 의해 유지된다.

이러한 관찰 결과를 이론적으로 이해하기 위해, 본 연구는 LLM 학습의 지형적 특성을 반영하도록 설계된 정형화된 이차 최적화 문제에서 Muon의 동작을 연구한다.
이러한 문제들에서 Adam과 GD는 경험적으로 유사한 NDS와 손실 감소를 보이므로, 단순화를 위해 GD와 Muon에 대한 분석에 집중한다.
이질적인 곡률과 고곡률 방향으로의 기울기 정렬 조건 하에서, Muon의 업데이트가 고곡률 방향과 저곡률 방향을 더 균등하게 균형 잡기 때문에 GD보다 평균 NDS가 더 작음을 증명한다.
또한, 곡률 이질성이 충분히 강할 때 Muon은 동일한 단계 수 이후 GD보다 더 작은 손실을 달성한다.
## Preliminaries
**Adam**은 지난 10년간 LLM 학습의 기본 옵티마이저로 자리 잡아 왔다. 행렬 파라미터 $W_t^{\text{Adam}} \in \mathbb{R}^{m\times n}$에 대해, 미니배치 $D_t$ 상의 기울기를 $G_t^{\text{Adam}} = \nabla_W \mathcal{L}_{D_t}(W_t^{\text{Adam}})$라 하면, Adam은 1차 및 2차 모멘트의 지수이동평균을 다음과 같이 유지한다.
$$M_t = \beta_1 M_{t-1}+(1-\beta_1)G_t^{\text{Adam}}, \quad V_t = \beta_2V_{t-1}+(1-\beta_2)(G_t^{\text{Adam}}\odot G_t^{\text{Adam}})$$
편향 보정 후 $M_t' = M_t/(1-\beta_1^t)$, $V_t' = V_t/(1-\beta_2^t)$를 얻고, 업데이트 방향은
$$Z_t^{\text{Adam}} = \eta_t M_t'/(\sqrt{V_t'}+\epsilon)$$
로 좌표별(element-wise)로 정규화된다. 즉 Adam은 파라미터의 각 성분을 자신의 과거 기울기 통계에 따라 독립적으로 스케일링하는 **좌표별 적응적(coordinate-wise adaptive)** 옵티마이저다.

**Muon**은 행렬 파라미터의 스펙트럼 구조를 명시적으로 활용하는 옵티마이저다. 모멘텀 누적값 $B_t = \mu B_{t-1}+G_t^{\text{Muon}}$을 유지하고, $B_t$의 SVD $B_t = U_tS_tV_t^\top$에서 $O_t = U_tV_t^\top$로 스펙트럼 정규화(spectral normalization)한 뒤 $Z_t^{\text{Muon}} = \eta_tO_t$를 업데이트 방향으로 사용한다. 이는 $B_t$의 0이 아닌 모든 특이값을 1로 맞춰, **모든 특이 방향에 동일한 크기의 업데이트를 부여**하는 것과 같다. 실무에서는 $O_t$를 정확한 SVD 대신 Newton–Schulz 반복으로 근사한다. 이 업데이트는 스케일 불변(scale-invariant)이라는 성질을 가진다 — $B_t$에 임의의 양의 스칼라를 곱해도 $Z_t^{\text{Muon}}$은 변하지 않는다.

**표기법**: $\langle A,B\rangle = \text{tr}(A^\top B)$는 Frobenius 내적, $\|A\|_F=\sqrt{\langle A,A\rangle}$는 그 노름이다. $G=\nabla_W\mathcal{L}_D(W)$는 기울기, $\mathcal{H}=\nabla_W^2\mathcal{L}_D(W)$는 행렬 섭동에 작용하는 Hessian 연산자로, $\mathcal{H}[Z] = \frac{d}{d\epsilon}\nabla_W\mathcal{L}_D(W+\epsilon Z)\big|_{\epsilon=0}$로 정의된다. $\text{mat}(\mathcal{H})$는 벡터화 기준의 행렬 표현이다.
## Main Results
### Muon Incurs a Smaller Second-Order Curvature Penalty Than Adam
곡률 관점에서 Muon의 우월성을 연구하기 위해, 한 단계 최적화 과정의 국소 분해부터 시작한다.
구체적으로, 파라미터 행렬 $W$와 업데이트 $Z$에 대하여, 미니 배치 $D$에서의 경험적 손실 감소는 다음과 같이 근사할 수 있다.
$$
\Delta_{D}(W, Z) \approx \langle G, Z \rangle - \frac{1}{2} \langle Z, \mathcal{H}[Z] \rangle = I_{D}^{(1)}(W, Z) - I_{D}^{(2)}(W, Z).
$$
1차 항은 업데이트 방향을 따라 이동함으로써 유도되는 손실 감소를 측정하는 반면, 곡률 패널티는 이러한 감소를 상쇄하는 2차 손실 증가를 포착한다.
위 식의 우변을 예측된 손실 감소라고 하며, 좌변을 실현된 손실 감소라고 한다.
![[Pasted image 20260728142459.png]]
실험 결과를 살펴보면 Muon이 동일한 Validation Loss 지점에서 Adam보다 항상 더 큰 손실 감소($\Delta D$)를 달성함을 보여준다.
실선(예측값)과 점선(실제값)이 상당히 일치하여, 2차 근사 모델이 옵티마이저의 성능 차이를 잘 설명하고 있음을 입증한다.

또한 위 그림의 (b)에서 볼 수 있듯이, Adam과 Muon은 최적화 과정 전반에 걸쳐 유사한 일차 감소량을 보인다.
반면, 그림 (c)는 곡률 항에서 명확한 차이를 보여준다.
Adam의 곡선은 Muon의 곡선보다 일관되게 위에 위치하며, 이는 Muon이 업데이트 방향을 따라 훨씬 더 작은 Hessian 이차 형식 패널티를 발생시킨다는 것을 나타낸다.
> **관찰 1**: Muon은 검증 손실 정렬 하에서 Adam보다 더 큰 일단계 손실 감소를 달성한다. 이 격차는 주로 더 작은 이차 곡률 비용 때문이다.
### Muon’s Smaller Curvature Penalty Comes from Its Update Direction
관찰 1은 한 단계 손실 감소에 있어 Muon의 이점이 주로 더 작은 2차 곡률 비용으로 설명됨을 보여준다.
곡률 패널티 $I_D^{(2)}(W,Z) = 1/2 \cdot \langle Z, \mathcal{H}[Z] \rangle$는 업데이트의 규모와 해당 방향의 곡률 모두에 의존한다는 점에 유의하자.
이제 우리는 Muon의 더 작은 곡률 비용이 더 작은 스텝을 취하기 때문인지, 아니면 업데이트 방향이 더 적은 곡률을 마주하기 때문인지 질문한다.
구체적으로, 0이 아닌 업데이트 $Z$에 대한 <strong>정규화된 방향성 선명도(Normalized Directional Sharpness, NDS)</strong>를 다음과 같이 정의한다.
$$
\mathcal{S}_F(W; Z) = \langle Z, \mathcal{H}[Z] \rangle / \|Z\|_F^2.
$$
![[Pasted image 20260728143623.png]]
위 그림의 (a)에서 볼 수 있듯이, Adam의 곡선은 Muon의 곡선보다 일관되게 위에 위치하며, 이는 Muon이 학습 전반에 걸쳐 Adam보다 낮은 NDS를 가짐을 나타낸다.
반면, (b)는 두 최적화 도구가 비슷한 업데이트 노름을 가짐을 보여준다. 두 곡선 모두 모든 검증 손실 수준에서 거의 평평하고 가깝게 유지된다.
> **관찰 2**: Muon과 Adam은 비슷한 업데이트 노름을 가지므로, Muon의 더 작은 곡률 패널티는 눈에 띄게 더 작은 NDS에 의해 주도된다.
### Dataset Imbalance Widens the NDS Gap between Adam and Muon
선행 연구들은 학습 데이터의 꼬리 구조(tail structure)와 불균형이 두 가지 방식으로 최적화 알고리즘의 동작과 강하게 상호작용할 수 있음을 시사한다.
첫째, 신경망의 Hessian 분석은 Hessian 스펙트럼이 데이터 혼합에 민감하게 의존함을 보여준다.
둘째, 최근 연구들은 Muon이 특히 꼬리가 긴(heavy-tailed) 데이터에서 Adam보다 뛰어난 성능을 보일 수 있음을 발견했다.
이러한 발견에 동기를 얻어, 본 연구는 데이터셋 불균형이 위에서 식별된 NDS 격차를 증폭시키는지 연구한다.

전체 학습 궤적을 따라 NDS를 평가하기 위해, 각 최적화 알고리즘 $\text{opt} \in \{\text{Muon}, \text{Adam}\}$에 불균형 수준 $s$에서의 궤적 평균 NDS를 다음과 같이 정의한다.
$$
\bar{S}_{\text{opt}}(s) = \sum_{t \in \mathcal{T}} S_F(W_t^{\text{opt},s}; Z_t^{\text{opt},s}) / |\mathcal{T}|,
$$
여기서 $\mathcal{T}$는 학습 단계의 집합을 나타내고, $s \in \{0, 0.5, 1\}$은 불균형 수준이며, $W^{\text{opt, s}}_t$와 $Z^{\text{opt, s}}_t$는 단계 $t$에서 불균형 수준 $s$ 하에 옵티마이저 $\text{opt}$에 의해 유도된 파라미터의 업데이트를 나타낸다.
Adam과 Muon의 차이를 강조하기 위해 $\tilde{S}_{\text{opt}}(s)$를 $s = 0$일 때 Muon의 값으로 정규화한다.
![[Pasted image 20260728151529.png]]
위 그림 (a)에서는 정규화된 궤적 평균 NDS $\tilde{s}_{opt}(s)$를 Zipf 지수 $s \in \{0, 0.5, 1\}$에 대해 도식화하였다.
위 그림에서 볼 수 있듯이, 두 옵티마이저의 NDS는 모두 불균형에 따라 단조롭게 증가하지만, 그 효과는 Adam에서 훨씬 더 강력하다.
Adam의 정규화된 NDS는 $s$가 0에서 1로 증가함에 따라 1.63에서 2.38로 상승하는 반면, Muon은 1.00에서 1.25로만 증가한다.
(b)에서는 이 격차를 직접적으로 정량화한다. $\Delta(s)$는 $s = 0$일 때 0.63에서 $s = 1$일 때 1.13으로 단조롭게 넓어지며, 데이터가 더 불균형해짐에 따라 1.8배 증가한다.
> 관찰 3: 데이터셋의 불균형 수준을 높이는 것은 Muon과 Adam 모두의 NDS를 증폭시킬 뿐만 아니라, 두 옵티마이저 간의 NDS 격차를 넓힌다.
### Muon’s NDS increasingly shifts toward within-layer Hessian blocks
관찰 2는 모든 모델 파라미터에 걸쳐 Muon과 NDS 격차를 확립한다. 이 섹션에서는 서로 다른 레이어가 이 격차에 어떻게 기여하는지 연구한다.

$L$개의 레이어로 구성된 모델에서 전체 파라미터를 $W_t=(W_{t,1},\ldots,W_{t,L})$, 업데이트를 $Z_t=(Z_{t,1},\ldots,Z_{t,L})$로 레이어별로 나눌 수 있다.
Hessian 연산자도 레이어 블록 $\mathcal{H}_{\ell\ell'}$로 분해되는데, 대각 블록 $\mathcal{H}_{\ell\ell}$은 레이어 내부(within-layer) 곡률을, 비대각 블록 $\mathcal{H}_{\ell\ell'}\ (\ell\ne\ell')$은 레이어 간(cross-layer) 상호작용을 나타낸다.
이를 바탕으로 NDS를 다음과 같이 두 성분의 순수한 합으로 분해한다.
$$\mathcal{S}_F^{\text{within}}(W_t;Z_t) = \sum_{\ell=1}^{L}\langle Z_{t,\ell},\mathcal{H}_{\ell\ell}[Z_{t,\ell}]\rangle/\|Z_t\|_F^2,\quad \mathcal{S}_F^{\text{cross}}(W_t;Z_t) = \sum_{\ell\ne\ell'}\langle Z_{t,\ell},\mathcal{H}_{\ell\ell'}[Z_{t,\ell'}]\rangle/\|Z_t\|_F^2$$
이 분해를 바탕으로 상대적 레이어 내 기여도 $\rho_t^{\text{within}}=\mathcal{S}_F^{\text{within}}(W_t;Z_t)/\mathcal{S}_F(W_t;Z_t)$를 정의한다.
![[Pasted image 20260728161408.png]]
위 그림의 (a)는 Adam과 Muon 각각의 $\mathcal{S}_F^{\text{within}}$(실선)과 $\mathcal{S}_F^{\text{cross}}$(점선)을 학습 스텝에 대해 그린다.
네 곡선 모두 학습이 진행됨에 따라 감소하며, within/cross 두 성분 모두 Muon이 Adam보다 일관되게 작다.
Adam은 두 성분이 비슷한 속도로 줄어들어 그 비율이 대체로 안정적으로 유지되는 반면, Muon은 $\mathcal{S}_F^{\text{cross}}$가 $\mathcal{S}_F^{\text{within}}$보다 훨씬 빠르게 감소하여 학습이 진행될수록 두 곡선이 서로 수렴한다.
이는 레이어 내 성분이 Muon의 NDS에서 점점 더 지배적인 역할을 차지하게 됨을 의미한다.

(b)는 레이어 내 비중 $\rho_t^{\text{within}}$을 직접 보여준다.
Muon의 곡선은 학습 초반 약 14%에서 후반 약 44%로 가파르게 상승하여 레이어 내 비중이 거의 세 배가 되는 반면, Adam의 곡선은 약 27%에서 34%로 완만하게 오르내리며 대체로 30% 부근에서 안정적이다.
이는 학습 중후반 단계에서 Muon이 전체 모델 NDS를 낮게 유지하는 데 있어 작은 $\mathcal{S}_F^{\text{within}}$의 역할이 중요해짐을 시사하며, Adam은 레이어 내/간 성분 간의 균형을 비교적 일정하게 유지함을 보여준다.
> **관찰 4**: 학습이 진행됨에 따라 Muon의 방향성 선명도는 점점 더 레이어 내(within-layer) Hessian 블록 쪽으로 이동하는 반면, Adam의 선명도 구성은 상대적으로 안정적으로 유지된다. 레이어 내/간 성분 모두 Muon이 Adam보다 항상 작다.
## A Case Study of Structured Matrix-Block Quadratic Models
4절의 실증적 관찰(Observation 1–4)을 이론적으로 뒷받침하기 위해, 본 절에서는 하나의 가중치 행렬 블록을 고립시켜 국소적인 이차 모델(quadratic model) 상에서 Muon과 GD가 마주치는 곡률을 비교한다.
관찰 4가 보여주듯 학습 중후반에는 레이어 내(within-layer) 곡률이 NDS의 지배적 성분이 되므로, 여러 레이어에 걸친 상호작용은 무시하고 단일 블록 안에서의 곡률 구조에 집중한다.
### Structured Quadratic Model
고정된 파라미터 $W_0 \in \mathbb{R}^{d_1\times d_2}$, 그 기울기 $G=\nabla\mathcal{L}(W_0)$, Hessian 연산자 $\mathcal{H}=\nabla^2\mathcal{L}(W_0)$에 대해, 업데이트 $Y\in\mathbb{R}^{d_1\times d_2}$에 대한 국소 이차 모델을 다음과 같이 정의한다.
$$\mathcal{Q}(Y) = \mathcal{L}(W_0) - \langle G,Y\rangle + \frac{1}{2}\langle Y,\mathcal{H}[Y]\rangle$$
이 모델을 LLM 사전 학습에 대해 대표성 있게 만들기 위해 네 가지 가정을 도입하며, 각 가정은 실제 사전 학습 동역학에서 실증적으로 검증된다.
**가정 5.1 (Hessian의 낮은 Kronecker rank성)**. 국소 Hessian $\mathcal{H}$가 작은 Kronecker rank를 가진다고 가정한다. 즉 $\text{mat}(\mathcal{H})\in\mathbb{R}^{d_1d_2\times d_1d_2}$에 대해, $r\ll\min\{d_1^2,d_2^2\}$인 정수 $r$과 대칭행렬 $A_k\in\mathbb{R}^{d_1\times d_1}$, $B_k\in\mathbb{R}^{d_2\times d_2}$가 존재하여
$$\text{mat}(\mathcal{H}) = \sum_{k=1}^r B_k^\top \otimes A_k$$
가 성립한다. 이는 K-FAC(Martens and Grosse, 2015)에서 영감을 받은 것으로, Fisher 행렬이 Kronecker 구조로 잘 근사된다는 결과가 Hessian에도 확장됨을 시사한다.
Van Loan rearrangement를 통해 이 근사를 재배열된 행렬의 rank-$r$ SVD 문제로 환원할 수 있으며, attention 행렬 $W_Q,W_K,W_V,W_O$에서 rank $r=4$만으로도 잘 설명됨이 확인된다.
**가정 5.2 (동시 대각화)**. 가정 5.1의 Kronecker 인수들이 공통의 직교기저를 공유한다고 가정한다. 즉 직교행렬 $U\in\mathbb{R}^{d_1\times d_1}$, $V\in\mathbb{R}^{d_2\times d_2}$가 존재하여 모든 $k$에 대해
$$A_k = U\,\text{Diag}(a_k^{(1)},\ldots,a_k^{(d_1)})\,U^\top,\qquad B_k = V\,\text{Diag}(b_k^{(1)},\ldots,b_k^{(d_2)})\,V^\top$$
이 가정이 성립하면 $\text{mat}(\mathcal{H})$ 전체가 $U\otimes V$라는 단일 직교기저로 완전히 대각화된다.
JADE(Joint Approximate Diagonalization of Eigenmatrices) 알고리즘으로 측정한 동시대각화 점수는 $\{A_k\}$에서 0.892, $\{B_k\}$에서 0.845로, 최댓값 1에 근접하여 이 가정을 뒷받침한다.
이로써 $u_i,v_i$($U,V$의 열벡터)로 만든 rank-1 행렬 $M_i=u_{\pi(i)}v_{\pi(i)}^\top$이 Hessian의 고유행렬이 되고, $\mathcal{H}[M_i]\approx w_iM_i$가 성립한다.
**가정 5.3 (곡률 이질성)**. paired curvature $w_i=\sum_{k=1}^r a_k^{(i')}b_k^{(i')}$ 중 정확히 $q$개가 양수이며, 이들이 2단계 구조를 가진다고 가정한다: $w_i=w_H\ (i\in[m])$, $w_i=w_L\ (i\in\{m+1,\ldots,q\})$, $w_H>w_L$, $\alpha=m/q<1/2$.
실제 attention 파라미터에서 양의 paired curvature는 6자리 수 이상($w_1/w_{88}\approx 2.59\times10^6$)에 걸친 강한 long-tail 분포를 보여, 소수의 고곡률 모드가 존재한다는 이 가정을 뒷받침한다.
**가정 5.4 (기울기 정렬)**. 기울기 $G$가 $\{M_i\}_{i=1}^q$의 span 안에 놓이며($G=\sum_{i=1}^q\sigma_iM_i$), 그 계수 $\sigma_i$도 곡률과 동일한 2단계 구조($\sigma_i=\sigma_H,\ i\in[m]$; $\sigma_i=\sigma_L$, 나머지)를 가진다고 가정한다.
즉 **곡률이 큰 방향일수록 기울기 에너지도 크게 실린다**는 gradient–Hessian alignment를 반영한다.
누적 기울기 에너지 비율 $\zeta(i)=\|\Pi_{\mathcal{M}_i}G\|_F^2/\|G\|_F^2$이 $i\approx 30$에서 이미 0.8을 넘고 전체 양의 곡률 부분공간에서 $\zeta(q)=0.871$에 도달함이 이를 뒷받침한다.

네 가정을 종합하면, $\mathcal{H}[M_i]\approx w_iM_i$이고 $G\approx\sum_{i=1}^q\sigma_iM_i$이므로, $Y=\sum_i y_iM_i$로 전개할 때 이차 모델이 완전히 분리 가능한(separable) 스칼라 문제로 환원된다.
$$\mathcal{Q}(Y) = \mathcal{L}(W_0) - \sum_{i=1}^q\sigma_iy_i + \frac{1}{2}\sum_{i=1}^q w_iy_i^2$$
이 위에서 GD와 (모멘텀을 0으로 둔) Muon의 업데이트 규칙을 각각 다음과 같이 정의한다.
- **GD**: $Y_{t+1}^{\text{GD}} = Y_t^{\text{GD}} + \eta_t^{\text{GD}}Z_t^{\text{GD}}$, 여기서 $Z_t^{\text{GD}}$는 잔차(residual) $R_t=G-\mathcal{H}[Y_t]$ 그 자체.
- **Muon**: $Y_{t+1}^{\text{Muon}} = Y_t^{\text{Muon}} + \eta_t^{\text{Muon}}Z_t^{\text{Muon}}$, 여기서 $Z_t^{\text{Muon}} = \text{spec}(\nabla\mathcal{Q}(Y_t^{\text{Muon}}))$은 잔차의 모든 0이 아닌 특이값을 1로 정규화한 방향.
### Theoretical Results
두 옵티마이저 모두 $Y_0=0$에서 시작하고, 공정한 비교를 위해 exact line-search 스텝 크기 $\eta_t^{\text{opt}}=\arg\max_{\eta\ge0}\{\mathcal{Q}(Y_t^{\text{opt}})-\mathcal{Q}(Y_t^{\text{opt}}+\eta Z_t^{\text{opt}})\}$를 사용한다.
$\alpha=m/q$(고곡률 그룹의 상대적 크기), $\rho=w_H/w_L>1$(곡률 비)이라 할 때, 다음이 성립한다.
**정리 5.5**. 가정 5.1–5.4 하에서,
- **Muon의 더 작은 NDS**: 모든 유한한 horizon $T\ge1$에 대해, Muon의 시간평균 NDS가 GD보다 항상 작다: $\bar{\mathcal{S}}_T^{\text{Muon}} < \bar{\mathcal{S}}_T^{\text{GD}}$.
- **Muon의 더 큰 손실 감소**: $\rho+1 > 1/\alpha > 1+\sigma_H/\sigma_L$이면, 모든 $T\ge1$에 대해 Muon이 GD보다 더 낮은 손실을 달성한다: $\mathcal{Q}(Y_T^{\text{Muon}}) < \mathcal{Q}(Y_T^{\text{GD}})$.
두 번째 조건 $\rho>1/\alpha-1$은 최고 곡률 값이 나머지보다 훨씬 커야 함을 요구하며(Hessian 고유값의 이상치 현상과 부합), $\alpha<\sigma_L/(\sigma_L+\sigma_H)$는 고곡률 그룹이 양의 곡률 부분공간에서 충분히 작은 비중을 차지해야 함을 요구한다(실제로 관찰되는 이상치 곡률 방향의 희소성과 부합).
이 결과의 핵심 메커니즘은, Muon의 스펙트럼 정규화가 모든 직교 곡률 고유모드에 걸쳐 업데이트 크기를 균등화하여 고곡률·저곡률 방향에 에너지를 고르게 분배하는 반면, GD의 업데이트는 기울기에 비례하므로(가정 5.4에 의해) 고곡률 방향에 더 많은 에너지를 집중시킨다는 데 있다.
이 집중이 GD로 하여금 더 큰 방향성 선명도와, 곡률 이질성이 충분히 강할 때는 1차 이득을 상쇄하는 더 큰 곡률 페널티를 초래하게 만든다.
### Proof Sketch
가정 5.1–5.4 하에서 국소 이차 모델은 orthonormal한 rank-1 모드 $\{M_i\}_{i=1}^q$의 span 위에서 대각화되므로, $Y=\sum_iy_iM_i$에 대해 잔차 $r_i^{\text{opt}}=\sigma_i-w_iy_i^{\text{opt}}$로 동역학이 완전히 특징지어진다.
Muon은 $Z_t^{\text{Muon}}=\sum_i\text{sgn}(r_{i,t}^{\text{Muon}})M_i$로 모든 활성 모드에 동일한 진폭을 부여하는 반면, GD는 $Z_t^{\text{GD}}=\sum_ir_{i,t}^{\text{GD}}M_i$로 업데이트 에너지가 잔차 에너지에 비례한다.
**NDS 비교**. Muon의 NDS는 상수 $\mathcal{S}_F(Z_t^{\text{Muon}})=\alpha w_H+(1-\alpha)w_L$이다.
반면 GD의 NDS는 잔차 에너지 가중평균 $\mathcal{S}_F(Z_t^{\text{GD}})=P_t^{\text{GD}}w_H+(1-P_t^{\text{GD}})w_L$이며, 고곡률 에너지 비중 $P_t^{\text{GD}}$는 초기값 $p=m\sigma_H^2/(m\sigma_H^2+(q-m)\sigma_L^2)>\alpha$에서 시작해 $P_{t+1}^{\text{GD}}=1-P_t^{\text{GD}}$로 진동한다.
$\sigma_H>\sigma_L$로 인해 GD의 시간평균 고곡률 비중이 항상 $\alpha$보다 크므로 $\bar{\mathcal{S}}_T^{\text{Muon}}<\bar{\mathcal{S}}_T^{\text{GD}}$가 성립한다.
**손실 비교**. Suboptimality는 $\Phi_t^{\text{opt}}=\mathcal{Q}(Y_t^{\text{opt}})-\mathcal{Q}(Y^\star)=\frac{1}{2}\sum_i(r_{i,t}^{\text{opt}})^2/w_i$로 잔차만으로 표현된다. Muon은 스케일된 잔차 $|r_{H,t}^{\text{Muon}}|/w_H$와 $|r_{L,t}^{\text{Muon}}|/w_L$을 매 스텝 다음 비율로 서로에게 수렴시킨다.
$$\Gamma = \frac{|mw_H-(q-m)w_L|}{mw_H+(q-m)w_L}$$
GD는 대신 두 곡률 그룹 사이에서 잔차 에너지를 교대로 오가며, 그 축소율은 2스텝 인수 $R=C(P_0^{\text{GD}})C(1-P_0^{\text{GD}})$로 주어진다. 여기서
$$C(x) = \frac{(w_H-w_L)^2x(1-x)}{(w_L+(w_H-w_L)x)^2}$$
는 고곡률 에너지 비중이 $x$일 때 한 스텝 GD가 전체 잔차 에너지를 얼마나 축소시키는지를 나타낸다.
이로부터 $\Phi_T^{\text{Muon}}=\Phi_1^{\text{Muon}}\Gamma^{2(T-1)}$, $\Phi_T^{\text{GD}}=\Phi_1^{\text{GD}}R^{(T-1)/2}$라는 닫힌 형태를 얻는다.
정리 5.5의 조건 하에서 Muon은 첫 스텝의 gap이 더 작을 뿐 아니라($\Phi_1^{\text{Muon}}<\Phi_1^{\text{GD}}$) 이후의 축소율도 더 빠름($\Gamma^2<\sqrt{R}$)이 증명되어, 모든 $T\ge1$에 대해 $\mathcal{Q}(Y_T^{\text{Muon}})<\mathcal{Q}(Y_T^{\text{GD}})$가 성립한다.
요컨대 이 절의 증명은, 기울기가 편향적으로 고곡률 모드에 실려있을 때 GD는 그 편향을 그대로 따라가 고곡률 방향에 과도한 업데이트 에너지를 반복적으로 소모하는 반면, Muon의 스펙트럼 정규화는 활성 특이 모드 전체에 에너지를 균등하게 분산시켜 방향성 선명도를 낮추고 가파른 방향과 완만한 방향 모두에서 더 균형 잡힌 진전을 이룬다는 직관을 공식화한 것이다.
## References
- https://arxiv.org/pdf/2606.04662
## Backlinks
- [[[개념 정리] Muon + MuonClip]]