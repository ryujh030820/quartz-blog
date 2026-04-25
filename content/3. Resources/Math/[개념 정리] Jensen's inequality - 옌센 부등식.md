---
title: "[개념 정리] Jensen's inequality - 옌센 부등식"
description: Jensen's inequality에 대해 다룹니다.
date: 2026-02-07
tags:
  - Math
  - Concept
---
## 한 줄 요약
> [!quote] 볼록(또는 오목) 함수에 기대값을 넣는 것과, 함수에 넣은 뒤 기대값을 취하는 것 사이의 부등식을 보장하는 정리
## 핵심 아이디어
- 함수의 **볼록성(convexity)** 이 평균/기대값과 만났을 때 방향성이 생긴다  
- 평균을 먼저 넣느냐, 함수를 먼저 적용하느냐가 결과를 바꾼다  
- 확률변수뿐 아니라 가중평균에도 동일하게 적용된다  
## 정의 / 정리
- **정의:**  
  - 함수 $f$가 **볼록 함수(convex)** 이면, 임의의 확률변수 $X$에 대해  
    - $f(\mathbb{E}[X]) \;\le\; \mathbb{E}[f(X)]$
  - 함수 \( f \)가 **오목 함수(concave)** 이면 부등호 방향이 반대이다  
    - $f(\mathbb{E}[X]) \;\ge\; \mathbb{E}[f(X)]$
- **중요 성질/포인트:**  
  - 볼록성의 정의:
    - $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$
  - 연속형/이산형 확률변수 모두에 적용 가능  
  - 가중합(확률)의 합이 1이면 기대값이 아니어도 동일하게 성립  
- **헷갈리기 쉬운 점:**  
  - **볼록 ↔ 오목**에 따라 부등호 방향이 바뀐다  
  - $\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$ 인 경우가 대부분  
  - equality는 보통 $X$가 상수일 때만 성립  
## 직관 / 예시
- **기하학적 직관:**  
  - 볼록 함수에서는 **그래프 위의 평균점**이 **그래프 아래**에 있다  
- **간단한 예시:**  
  - $f(x) = x^2$ (볼록 함수)  
  - $X \in \{-1, 1\}$ with equal probability  
	$$
	\begin{gathered}    
    \mathbb{E}[X] = 0,\quad f(\mathbb{E}[X]) = 0
    \\
    \mathbb{E}[f(X)] = \mathbb{E}[X^2] = 1
    \\
    → f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
    \end{gathered}
    $$
## 증명 / 알고리즘
![[R1280x0.png|500]]
## References
- https://www.youtube.com/watch?v=10xgmpG_uTs