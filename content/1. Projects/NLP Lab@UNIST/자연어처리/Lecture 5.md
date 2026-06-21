---
title: Lecture 5
description: UNIST 나승훈 교수님의 수업 '자연어처리' 5강 PPT 내용에 대해 다룹니다.
date: 2026-06-20
tags:
  - AI
  - Lecture
---
## 1. 문장의 언어 구조
### 1.1 두 가지 관점

| 관점           | 이름                             | 설명                     |
| ------------ | ------------------------------ | ---------------------- |
| Constituency | Phrase Structure Grammar / CFG | 단어들을 중첩된 구(phrase)로 조직 |
| Dependency   | Dependency Structure           | 단어들 간의 의존 관계를 나타냄      |
**Constituency 예시**
```
the cuddly cat → NP
by the door    → PP
the cuddly cat by the door → 더 큰 NP
```
**Dependency 예시**
```
Look in the large crate in the kitchen by the door
→ 각 단어가 어떤 단어에 의존(수식/논항)하는지 표시
```
### 1.2 왜 문장 구조가 필요한가?
- 인간은 단어를 조합해 복잡한 의미를 전달
- 모델은 어떤 단어가 어떤 단어를 수식하는지 파악해야 언어를 올바르게 해석 가능
- 예: "Scientists count whales from space" → "from space"가 "count"에 걸리는지 "whales"에 걸리는지 모호
### 1.3 PP Attachment Ambiguity
전치사구(PP)가 어디에 붙느냐에 따라 의미가 달라지는 문제.
예시:
> "The board approved ==[its acquisition]== ==[by Royal Trustco Ltd.]== ==[of Toronto]== ==[for $27 a share]== ==[at its monthly meeting]==."

PP가 $n$개 있을 때 가능한 파스 트리의 수는 **카탈란 수(Catalan Number)**를 따름:
$$C_n = \frac{(2n)!}{(n+1)!n!}$$
- 지수적으로 증가하는 수열
- 다각형의 삼각분할, 트리 구조 등 다양한 맥락에서 등장
## 2. Probabilistic Context-Free Grammar (PCFG)
### 2.1 CFG 정의
CFG는 4-tuple $\langle N, \Sigma, S, R \rangle$:
- $N$: **비단말(Non-terminals)** 집합
    - Phrasal categories: S, NP, VP, ADJP 등
    - Parts-of-speech (pre-terminals): NN, JJ, DT, VB 등
- $\Sigma$: **단말(Terminals)** 집합 (실제 단어들)
- $S$: **시작 기호(Start Symbol)** — 보통 ROOT 또는 TOP
- $R$: **규칙(Rules)** 집합
    - 형태: $X \rightarrow Y_1 Y_2 \ldots Y_n$, where $X \in N$, $Y_i \in (N \cup \Sigma)$
    - 예: $S \rightarrow NP\ VP$, $VP \rightarrow VP\ CC\ VP$
### 2.2 PCFG
PCFG는 CFG에 확률 분포 $q$를 추가:
$$\sum_{X \rightarrow \beta \in R} q(X \rightarrow \beta) = 1 \quad \forall X \in N$$
트리 $t$의 확률:
$$P(t) = \prod_{i=1}^{n} q(\alpha_i \rightarrow \beta_i)$$
**학습**: Treebank(예: Penn WSJ Treebank, 50,000 문장)에서 규칙 빈도를 세어 MLE로 추정
**추론**: 주어진 문장 $s$에 대해
$$t^* = \arg\max_{t \in T(s)} P(t)$$
### 2.3 Chomsky Normal Form (CNF)
모든 규칙을 다음 두 형태로 제한:
- $X \rightarrow Y\ Z$ (비단말 2개)
- $X \rightarrow w$ (단말 1개)
**변환 방법**:
- N-ary 규칙 → 새 비단말 도입해 이진화
- Unary/empty 규칙 → "promote" 처리
CNF를 쓰면 **파싱 알고리즘(특히 CKY)이 단순해짐**
변환 예시:
```
VP → VBD NP PP PP
↓
VP → [VP→VBD NP PP •]  PP
[VP→VBD NP PP •] → [VP→VBD NP •]  PP
[VP→VBD NP •] → VBD  NP
```
### 2.4 CKY 알고리즘
**입력**: 문장 $s = x_1 \ldots x_n$, PCFG $= \langle N, \Sigma, S, R, q \rangle$
**DP 테이블**: $\pi(i, j, X)$ = 스팬 $[i,j]$를 헤드 $X$로 파싱한 최적 확률
**초기화** ($i = 1 \ldots n$, 모든 $X \in N$):
$$\pi(i, i, X) = \begin{cases} q(X \rightarrow w_i) & \text{if } X \rightarrow w_i \in R \ 0 & \text{otherwise} \end{cases}$$
**점화식** ($l = 1 \ldots n-1$, $i = 1 \ldots n-l$, $j = i+l$):
$$\pi(i, j, X) = \max_{\substack{X \rightarrow YZ \in R \ i \leq k \leq j-1}} q(X \rightarrow YZ) \cdot \pi(i, k, Y) \cdot \pi(k+1, j, Z)$$
**Back Pointer** (파스 트리 복원용):
$$bp(i, j, X) = \underset{\substack{X \rightarrow YZ \in R \ i \leq k \leq j-1}}{\text{argmax}}\ q(X \rightarrow YZ) \cdot \pi(i,k,Y) \cdot \pi(k+1,j,Z)$$
**시간 복잡도**: $O(n^3 |N|)$
### 2.5 PCFG 확장 모델
#### PCFG with Latent Annotation (Matsuzaki et al. '05)
- 기본 PCFG는 `NP` 같은 범주 하나에 단일 확률만 부여 → 맥락 구분 불가
	- 예: 주어 자리의 NP vs 목적어 자리의 NP
- 각 비단말에 **잠재 하위 범주** 도입: $NP \rightarrow NP^{(1)}, NP^{(2)}, \ldots$
- EM 알고리즘으로 잠재 범주 자동 학습
- Berkeley Parser의 핵심 아이디어
#### Compositional Vector Grammars (Socher et al. '13)
- PCFG의 구조적 뼈대 + 신경망의 표현력 결합
- 각 스팬에 대해 **스칼라 점수** + **벡터 표현** 동시 계산
- 자식 노드 $Y$, $Z$로부터 부모 $X$의 벡터 합성:
$$\mathbf{p} = f\left(W \begin{bmatrix} \mathbf{y} \ \mathbf{z} \end{bmatrix} + \mathbf{b}\right)$$
$$s(X, i, j) = \mathbf{v}^\top \mathbf{p}$$
- $\mathbf{v}$: 벡터를 스칼라 점수로 압축하는 **학습 가능한 파라미터**
- 언어학적 합성성 원리(Principle of Compositionality)를 직접 모델링
#### Span-Based Neural Constituency Parsing (Kitaev et al. '18)
- 규칙 기반 PCFG 없이 신경망으로 직접 스팬 점수 예측
- BiLSTM 또는 BERT로 문장 인코딩
- 스팬 $(i,j)$ 표현: $\mathbf{v}_{ij} = \mathbf{h}_j - \mathbf{h}_i$
- MLP로 비단말 $X$별 점수 계산, CKY로 최적 트리 탐색
- BERT 기반 모델은 state-of-the-art 수준
## 3. Dependency Grammar
### 3.1 기본 개념
- 문장 구조를 **어휘 항목 간의 이진 비대칭 관계(호, arc)** 로 표현
- 호 $h \rightarrow d$: $h$ = head(헤드), $d$ = dependent(의존어)
- 호에는 문법 관계 레이블 부착 (subject, object, nmod 등)
- 의존 구조는 보통 **rooted tree** (연결, 비순환, 단일 루트)
- 관례: ROOT 노드를 추가해 모든 단어가 정확히 하나의 헤드를 가지도록 함
### 3.2 의존 파싱의 정보 출처
1. **Bilexical affinities**: 특정 단어 쌍의 의존 관계 선호도
2. **Dependency distance**: 대부분의 의존 관계는 인접 단어 사이
3. **Intervening material**: 동사나 구두점을 넘는 의존 관계는 드묾
4. **Valency of heads**: 헤드가 보통 몇 개의 의존어를 갖는지
### 3.3 두 가지 의존 트리 유형
![[Pasted image 20260620200820.png]]

| 유형                 | 설명        | 특징                      |
| ------------------ | --------- | ----------------------- |
| **Projective**     | 교차하는 호 없음 | CFG 트리와 대응, DP 적용 가능    |
| **Non-projective** | 교차하는 호 존재 | 자유 어순 언어(한국어, 터키어)에서 빈번 |
**Projective 정의**: 단어들을 선형 순서로 나열했을 때, 모든 의존 호가 서로 교차하지 않음
**Non-projective가 필요한 이유**: 추방된 구성요소(displaced constituents) 처리
- 예: "Who did Bill buy the coffee from yesterday?"
- `Who`는 의미상 `from`의 목적어이지만 wh-이동으로 앞으로 이동 → 교차 호 발생
- Non-projective 없이는 올바른 의미 해석 불가
## 4. Transition-Based Dependency Parsing
### 4.1 기본 아이디어
- 파서 상태(configuration)를 정의하고, **전이(transition) 시퀀스**로 파스 트리 구축
- 각 전이는 **분류기(classifier)** 가 결정
- **시간 복잡도: $O(n)$** — 매우 빠름
### 4.2 파서 상태
$$c = (\sigma,\ \beta,\ A)$$
- $\sigma$: **스택** — 처리 중인 단어들 (top이 오른쪽)
- $\beta$: **버퍼** — 아직 처리 안 된 단어들 (front가 왼쪽)
- $A$: **호 집합** — 확정된 의존 관계들
**초기 상태**: $c_x = ([ROOT],\ [w_1, \ldots, w_n],\ \emptyset)$ **종료 상태**: $C_x = (\sigma,\ [],\ A)$ (버퍼가 빈 상태)
### 4.3 Arc-Standard vs Arc-Eager
#### Arc-Standard (Nivre 2003)

|전이|동작|
|---|---|
|**SHIFT**|버퍼 front → 스택으로 push|
|**LEFT-ARC$_r$**|스택 top2 ← top 호 추가, top2 pop|
|**RIGHT-ARC$_r$**|스택 top2 → top 호 추가, top pop|
예시: "I ate fish"
![[Pasted image 20260620201337.png|500]]
#### Arc-Eager (Nivre 2008)

|전이|동작|수식|
|---|---|---|
|**LEFT-ARC$_l$**|$j \rightarrow i$ 호 추가, $i$ pop|$(\sigma\|i,\ j\|\beta,\ A) \Rightarrow (\sigma\|j,\ \beta,\ A \cup {(j,i,l)})$|
|**RIGHT-ARC$_l$**|$i \rightarrow j$ 호 추가, $j$ push|$(\sigma\|i,\ j\|\beta,\ A) \Rightarrow (\sigma\|i\|j,\ \beta,\ A \cup {(i,j,l)})$|
|**REDUCE**|스택 top pop|$(\sigma\|i,\ \beta,\ A) \Rightarrow (\sigma,\ \beta,\ A)$|
|**SHIFT**|버퍼 front → 스택 push|$(\sigma,\ i\|\beta,\ A) \Rightarrow (\sigma\|i,\ \beta,\ A)$|
**Preconditions**:

| 전이            | 조건                                                                                           |
| ------------- | -------------------------------------------------------------------------------------------- |
| LEFT-ARC$_l$  | $\lnot[i=0]$ (ROOT는 의존어 불가), $\lnot\exists k\exists l'[(k,i,l') \in A]$ ($i$가 이미 헤드를 가지면 불가) |
| RIGHT-ARC$_l$ | $\lnot\exists k\exists l'[(k,j,l') \in A]$ ($j$가 이미 헤드를 가지면 불가)                              |
| REDUCE        | $\exists k\exists l[(k,i,l) \in A]$ ($i$가 반드시 헤드를 가져야 함)                                     |
**Arc-Standard vs Arc-Eager 비교**:

|             | Arc-Standard | Arc-Eager       |
| ----------- | ------------ | --------------- |
| RIGHT-ARC 후 | $j$ 바로 pop   | $j$ 스택에 유지      |
| 호 생성 시점     | 나중에          | 즉시(eager)       |
| REDUCE      | 불필요          | 필요 (헤드 받은 후 제거) |
### 4.4 MaltParser
- 분류기: softmax classifier
- 특징(feature): 스택 top 단어/POS, 버퍼 front 단어/POS 등
- Beam search 옵션으로 정확도 향상 가능
- **매우 빠른 선형 시간 파싱**
### 4.5 평가 지표
- **UAS (Unlabeled Attachment Score)**: 헤드만 맞으면 정답
- **LAS (Labeled Attachment Score)**: 헤드 + 레이블 모두 맞아야 정답
$$\text{UAS} = \frac{\text{\# correct heads}}{\text{\# of deps}}, \quad \text{LAS} = \frac{\text{\# correct (head, label) pairs}}{\text{\# of deps}}$$
## 5. Graph-Based Dependency Parsing
### 5.1 기본 아이디어
$$G^* = \arg\max_{G \in D(G_x)} s(G)$$
가능한 모든 의존 트리 중 **총 점수가 가장 높은 트리** 탐색
- **Arc-factored model**: 트리 점수 = 각 호 점수의 합
- 전역(global), 완전(exhaustive) 탐색
### 5.2 방법론

| 언어 유형          | 알고리즘                     | 복잡도      |
| -------------- | ------------------------ | -------- |
| Projective     | Eisner 알고리즘              | $O(n^3)$ |
| Non-projective | Chu-Liu-Edmonds MST 알고리즘 | $O(n^2)$ |
### 5.3 Graph-Based Projective Parsing의 DP
**DP 테이블**: $s(l, r, h)$ = 스팬 $[l,r]$을 헤드 $h$로 파싱한 최적 점수
**점화식**:
$$s(l,r,h) = \max_{l \leq m \leq r-1} \Bigg( \max_{\substack{l \leq h' \leq m \ m < h}} \big(s(l,h',m) + s(m+1,r,h) + s(h' \leftarrow h)\big),$$
$$\max_{\substack{m+1 \leq h' \leq r \ m \geq h}} \big(s(l,h,m) + s(m+1,r,h') + s(h \rightarrow h')\big) \Bigg)$$
Naive 방법: $(l, r, h, m, h')$ 5개 파라미터 → $O(n^5)$
### 5.4 Eisner 알고리즘
**핵심 아이디어**: 왼쪽/오른쪽 의존어를 **독립적으로** 수집 → 3단계 분해
#### DP 아이템 4종류 (`[min, max, direction, complete]`)

|아이템|도형|의미|
|---|---|---|
|`[min,max,left,no]`|직각삼각형 (빗변 아래-오른쪽)|Incomplete, 헤드=왼쪽, 호 열린 상태|
|`[min,max,right,no]`|직각삼각형 (빗변 아래-왼쪽)|Incomplete, 헤드=오른쪽, 호 열린 상태|
|`[min,max,left,yes]`|사다리꼴 (빗변 위-오른쪽)|Complete, 헤드=왼쪽, 헤드 방향 기억|
|`[min,max,right,yes]`|사다리꼴 (빗변 위-왼쪽)|Complete, 헤드=오른쪽, 헤드 방향 기억|
|`[s,m]`|직사각형|Complete, 헤드 방향 소멸, 순수 스팬|
- **Incomplete(no)**: 호가 확정됐지만 의존어 서브트리가 아직 흡수되지 않은 상태
- **Complete(yes)**: 해당 스팬의 모든 의존어 흡수 완료, 봉인된 서브트리
- **사다리꼴 vs 직사각형**: 사다리꼴은 헤드 방향 정보를 유지, 직사각형은 형제 처리 완료 후 헤드 방향 불필요
#### 3단계 연산
**Step 1 — 호 생성 (Incomplete 생성)**
```
Complete(h) + Complete(h') → Incomplete(h, h')
```
호를 확정하고 열린 상태 아이템 생성. 파라미터: $(h, q, h')$ → 3 positions
**Step 2 — Complete 아이템 확장 (사다리꼴)**
```
Incomplete(h, h') + Complete(h', j) → Complete(h, j)
```
완성된 오른쪽 서브트리를 흡수해 닫힘. 파라미터: $(h, h', j)$ → 3 positions
**Step 3 — 두 Complete 아이템 병합 (직사각형 생성)**
```
Complete(s, r) + Complete(r+1, m) → Rectangle(s, m)
```
두 완성 스팬을 이어붙여 형제 스팬 생성. 파라미터: $(s, r, m)$ → 3 positions
#### 복잡도 비교

| 방법     | 단계당 파라미터          | 복잡도      |
| ------ | ----------------- | -------- |
| Naive  | 5 positions       | $O(n^5)$ |
| Eisner | 3 positions × 3단계 | $O(n^3)$ |
#### Pseudo Code
```
for each i from 0 to n and all d,c do
    C[i][i][d][c] = 0.0

for each m from 1 to n do
    for each i from 0 to n-m do
        j = i+m

        # Step 1: Incomplete 생성
        C[i][j][←][1] = max_{i≤q<j}(C[i][q][→][0] + C[q+1][j][←][0] + score(w_j, w_i))
        C[i][j][→][1] = max_{i≤q<j}(C[i][q][→][0] + C[q+1][j][←][0] + score(w_i, w_j))

        # Step 2/3: Complete 생성
        C[i][j][←][0] = max_{i≤q<j}(C[i][q][←][0] + C[q][j][←][1])
        C[i][j][→][0] = max_{i≤q<j}(C[i][q][→][1] + C[q][j][→][0])

return [0][n][→][0]
```
- `q`: 스팬 $[i,j]$를 $[i,q]$와 $[q+1,j]$로 나누는 **분할점**
### 5.5 고차원 파서 (Koo & Collins '09)
기본 Eisner(1차)는 호 하나의 점수 $s(h \to e)$만 고려.
**2차(Second-order) Sibling Parser**: 형제(sibling) 관계 추가 고려
$$s(h \to e,\ h \to \text{prev\_sibling})$$
DP 아이템이 형제 정보를 포함하도록 확장. 복잡도: $O(n^4)$
**3차(Third-order) Parser**: 조부모(grandparent) 관계까지 추가
$$s(g \to h \to e)$$
DP 아이템에 조부모 인덱스 $g$ 추가. 복잡도: $O(n^4)$

| 차수                  | 고려 관계        | 복잡도      |
| ------------------- | ------------ | -------- |
| 1st (Eisner)        | 호 하나         | $O(n^3)$ |
| 2nd (Sibling)       | 호 + 형제       | $O(n^4)$ |
| 3rd (Koo & Collins) | 호 + 형제 + 조부모 | $O(n^4)$ |
## 6. Neural Dependency Parsing
### 6.1 기존 방법의 문제
기존 특징(feature) 표현의 문제점:
- **Sparse**: 희소 이진 벡터 (차원 $10^6 \sim 10^7$)
- **Incomplete**: 보지 못한 조합은 처리 불가
- **Expensive**: 파싱 시간의 95% 이상이 특징 계산에 소요
### 6.2 Neural Dependency Parser (Chen & Manning 2014)
**두 가지 핵심 이점**:
1. **Distributed Representations (분산 표현)**
    - 단어, POS 태그, 의존 레이블을 모두 $d$차원 dense 벡터로 표현
    - 유사한 단어/태그는 가까운 벡터 (NNS ↔ NN, nummod ↔ amod 등)
2. **Non-linear Classifier (비선형 분류기)**
    - Softmax만으로는 선형 결정 경계만 가능
    - 은닉층을 통해 훨씬 복잡한 비선형 경계 학습
**모델 구조**:
```
입력층 x       ← 스택/버퍼의 단어, POS, 의존 레이블 벡터 concat
      ↓
은닉층 h = ReLU(Wx + b₁)
      ↓
출력층 y = softmax(Uh + b₂)
      ↓
{SHIFT, LEFT-ARC_r, RIGHT-ARC_r}
```
**성능 비교** (PTB WSJ, Stanford Dependencies):

| Parser              | UAS  | LAS  | 속도(sent/s) |
| ------------------- | ---- | ---- | ---------- |
| MaltParser          | 89.8 | 87.2 | 469        |
| MSTParser           | 91.4 | 88.1 | 10         |
| TurboParser         | 92.3 | 89.6 | 8          |
| Chen & Manning 2014 | 92.0 | 89.7 | **654**    |
### 6.3 이후 발전
**SyntaxNet / Parsey McParseFace (Google, 2016)**:
- 더 크고 깊은 신경망
- Beam search
- CRF 스타일 글로벌 추론

| 방법                  | UAS   | LAS   |
| ------------------- | ----- | ----- |
| Chen & Manning 2014 | 92.0  | 89.7  |
| Weiss et al. 2015   | 93.99 | 92.05 |
| Andor et al. 2016   | 94.61 | 92.79 |
### 6.4 Neural Graph-Based Parser (Dozat & Manning 2017)
- **Biaffine scoring model** 사용
- 신경망 시퀀스 모델(LSTM) 기반 문맥 표현
- 각 단어 쌍에 대해 의존 점수 계산 → MST 알고리즘으로 최적 트리
- Transition-based보다 느리지만 더 높은 정확도

| 방법                       | UAS       | LAS       |
| ------------------------ | --------- | --------- |
| Andor et al. 2016        | 94.61     | 92.79     |
| **Dozat & Manning 2017** | **95.74** | **94.08** |
## 핵심 정리

| 파서 유형                    | 방법                    | 복잡도      | 장단점                  |
| ------------------------ | --------------------- | -------- | -------------------- |
| Transition-based         | MaltParser, Arc-Eager | $O(n)$   | 빠르지만 탐욕적(greedy)     |
| Graph-based (projective) | Eisner 알고리즘           | $O(n^3)$ | 전역 최적, projective만   |
| Graph-based (non-proj.)  | Chu-Liu-Edmonds       | $O(n^2)$ | non-projective 처리 가능 |
| Neural transition        | Chen & Manning        | $O(n)$   | 빠르고 정확               |
| Neural graph             | Dozat & Manning       | $O(n^2)$ | 가장 높은 정확도            |
