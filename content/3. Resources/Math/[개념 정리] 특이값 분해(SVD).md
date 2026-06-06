---
title: "[개념 정리] 특이값 분해(SVD)"
description: 특이값 분해(SVD)에 대해 다룹니다.
date: 2026-06-05
tags:
  - Math
  - Concept
---
## 정의
특이값 분해(Singular Value Decomposition, SVD)는 임의의 $m \times n$ 차원의 행렬 $A$에 대하여 다음과 같이 행렬을 분해할 수 있는 ‘행렬 분해(decomposition)’ 방법 중 하나이다.
$$
A = U \Sigma V^T
$$
여기서 네 행렬($A, U, \Sigma, V$)의 크기와 성질은 다음과 같다.
- $A: m \times n$ rectangular matrix
- $U: m \times m$ orthogonal matrix
- $\Sigma: m \times n$ diagonal matrix
- $V: n \times n$ orthogonal matrix
## 특이값 분해의 기하학적 의미
특이값 분해는 다음과 같은 의미를 갖는다.
“직교하는 벡터 집합에 대하여, 선형 변환 후에 그 크기는 변하지만 여전히 직교할 수 있게 되는 그 직교 집합은 무엇인가? 그리고 선형 변환 후의 결과는 무엇인가?”
### 2차원 벡터공간에서의 예제
설명을 단순화시키고, 시각적인 설명을 할 수 있도록 행렬 $A$가 $2 \times 2$차원인 경우에 한정하여 생각해보자.
우선은 2차원 실수 벡터 공간에서 하나의 벡터가 주어지면 언제나 그 벡터에 직교하는 벡터를 찾을 수 있을 것이다.
그런데 직교하는 두 벡터에 대해 동일한 선형 변환 $A$를 취해준다고 했을 때, 그 변환 후에도 여전히 직교한다고 보장할 수는 없다.
아래 그림은 행렬 A에 대하여,
$$
A =
\begin{pmatrix}
0.25 & 0.75 \\
1 & 0.5
\end{pmatrix}
$$
임의의 벡터 $\vec{x}$를 선형변환 시켰을 때의 결과($A\vec{x}$)를 보여주고 있다.
![[pic1.gif|400]]
그렇다면 직교하는 두 벡터에 대해 동시에 선형 변환을 시켜본다면, 선형 변환 후의 결과가 직교하는 경우를 찾을 수 있을까?
아래의 그림은 동일한 행렬 $A$에 대하여 직교하는 두 벡터 $\vec{x}$와 $\vec{y}$에 대한 선형 변환 결과(각각 $A\vec{x}, A\vec{y}$)를 보여준다.
![[pic2.gif|400]]
위 그림에서 주목할 것은 크게 두 가지이다.
1. $A\vec{x}$와 $A\vec{y}$가 직교하게 되는 경우는 단 한 번만 있는 것이 아니다.
2. $A\vec{x}$와 $A\vec{y}$는 $A$라는 행렬(즉, 선형변환)을 통해 변환되었을 때, 길이가 조금씩 변했다. 이 값들을 scaling factor라고 할 수 있지만, 일반적으로는 signular value라고 하고 크기가 큰 값부터 $\sigma_1, \sigma_2, \dots$ 등으로 부른다.
처음으로 돌아가서, 임의의 $m \times n$ 행렬 $A$는 다음과 같이 분해된다고 했다.
$$
A = U \Sigma V^T
$$
위의 예시에서 보여준 선형 변환 전의 직교하는 벡터 $\vec{x}, \vec{y}$는 다음과 같이 열벡터의 모음으로 생각할 수 있으며, 이것이 $A = U \Sigma V^T$에서 $V$ 행렬에 해당된다.
$$
V= \begin{pmatrix} |  & | \\\\ \vec x & \vec y \\\\ |  & | \end{pmatrix}
$$
또 위 예시에서 보여준 선형변환 후의 직교하는 벡터 $A\vec{x}, A\vec{y}$에 대하여 각각의 크기를 1로 정규화한 벡터를 $\vec{u}_1, \vec{u}_2$라 한다면 이 둘의 열벡터의 모음이 $A = U \Sigma V^T$에서 $U$ 행렬에 해당된다.
$$
U= \begin{pmatrix} |  & | \\\\ \vec{u}_1 & \vec{u}_2 \\\\ |  & | \end{pmatrix}
$$
마지막으로 singular value(즉, scaling factor)는 다음과 같이 $\Sigma$ 행렬로 묶어서 생각할 수 있다.
$$
\Sigma= \begin{pmatrix} \sigma_1 & 0 \\\\ 0 & \sigma_2 \end{pmatrix}
$$
선형 변환의 관점에서 네 개의 행렬($A, V, \Sigma, U$)의 관계를 생각하면 다음과 같다. ($V$는 직교행렬이므로 $V^T = V^{-1}$이므로 성립)
$$
AV=U\Sigma
$$
즉 “$V$에 있는 열벡터를 행렬 $A$를 통해 선형변환할 때, 그 크기는 $\sigma_1, \sigma_2$만큼 변하지만, 여전히 직교하는 벡터들 $\vec{u}_1, \vec{u}_2$를 찾을 수 있겠느냐?”라고 묻는 것이다.
## 특이값 분해의 목적
특이값 분해의 공식을 다시 풀어 써보자면 다음과 같다.
$$
\begin{gathered}
A = U\Sigma V^T \\

= \begin{pmatrix} | & | & & | \\ \vec{u}_1 & \vec{u}_2 & \cdots & \vec{u}_m \\ | & | & & | \end{pmatrix} \begin{pmatrix} \sigma_1 & & & 0 \\ & \sigma_2 & & 0 \\ & & \ddots & 0 \\ & & & \sigma_m & 0 \end{pmatrix} \begin{pmatrix} - & \vec{v}_1^T & - \\ - & \vec{v}_2^T & - \\ & \vdots & \\ - & \vec{v}_n^T & - \end{pmatrix} \\

= \sigma_1 \vec{u}_1 \vec{v}_1^T + \sigma_2 \vec{u}_2 \vec{v}_2^T + \cdots + \sigma_m \vec{u}_m \vec{v}_m^T
\end{gathered}
$$
여기서 $\vec{u}_1\vec{v}_1^T$ 등은 $m \times n$ 행렬이 된다. 또 $\vec{u}$와 $\vec{v}$는 정규화된 벡터이기 때문에 $\vec{u}_1\vec{v}_1^T$ 내의 성분의 값은 -1에서 1 사이의 값을 가진다.
따라서 $\sigma_1\vec{u}_1\vec{v}_1^T$라는 부분만을 놓고 보면, 이 행렬의 크기는 $\sigma_1$의 값에 의해 정해지게 된다.
즉 우리는 SVD라는 방법을 이용해 $A$라는 임의의 행렬을 여러개의 $A$ 행렬과 동일한 크기를 갖는 여러개의 행렬로 분해해서 생각할 수 있는데, 분해된 각 행렬의 원소의 값의 크기는 $\sigma$의 값의 크기에 의해 결정된다.
다시 말해, SVD는 임의의 행렬 $A$를 정보량에 따라 여러 layer로 쪼개서 생각할 수 있게 해준다.
## References
- https://angeloyeo.github.io/2019/08/01/SVD.html