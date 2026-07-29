---
title: "[개념 정리] 점수 함수(Score Function)와 피셔 정보(Fisher information)"
date: 2026-07-22
tags:
  - Math
  - Concept
---
## Likelihood
![[Pasted image 20260722153554.png|400]]
**베이즈 정리(Bayes Theorem)** 에 의하면 사후 확률(Posterior Probability)은 정규화 상수(Normalizing Constant), 사전 확률(Prior Probability), 그리고 가능도(Likelihood)로 분리가 된다.
여기서 우도(가능도, Likelihood)는 다음과 같이 표현된다.
$$
L(\theta) = f(x|\theta) = \prod_{i=1}^{n} f(x_i|\theta)
$$
곱셈의 형태를 계산하기 쉽게 바꿔주기 위해 **로그(Log)** 를 취해 덧셈으로 바꿔준다.
우도 또한 확률이므로 1보다 작은 수이기에 계속해서 곱해주면 결국 0에 근사해버리는 문제도 있다.
따라서 **로그 우도(로그 가능도, Log Likelihood)** 는 다음과 같다.
$$
l(\theta) = \log L(\theta) = \log f(x|\theta) = \log \prod_{i=1}^{n} f(x_i|\theta) = \sum_{i=1}^{n} \log f(x_i|\theta)
$$
## Score Function
$$
s(\theta) = l'(\theta) = \frac{\partial l(\theta)}{\partial \theta} = \frac{\partial}{\partial \theta} \log f(x|\theta) = \frac{1}{f(x|\theta)} \cdot \frac{\partial}{\partial \theta} f(x|\theta) = \frac{f'(x|\theta)}{f(x|\theta)}
$$

**점수 함수(스코어 함수, Score Function)** 는 위와 같이 로그 우도를 모수로 미분한 것을 의미한다.
![[Pasted image 20260722154011.png]]
결국 Score Function은 각 Likelihood의 **기울기(Slope)** 를 의미하게 된다.
Score Function의 평균 즉, 기댓값을 구해보면 다음과 같다.
$$
\begin{align*}
\mathrm{E}_f\big[s(\theta)\big] &= \int s(\theta) f(x|\theta)\,dx = \int \left( \frac{1}{f(x|\theta)} \frac{\partial}{\partial \theta} f(x|\theta) \right) f(x|\theta)\,dx \\
&= \int \frac{\partial}{\partial \theta} f(x|\theta)\,dx = \frac{\partial}{\partial \theta} \int f(x|\theta)\,dx = \frac{\partial}{\partial \theta} 1 = 0
\end{align*}
$$
중간에 미분과 적분을 교환한 것은 **라이프니츠 적분 규칙(Leibniz Integral Rule)** 에 의한 것이다.
모든 확률의 합은 1이므로, 1을 미분하면 결과적으로 0이 된다.
따라서, Score Function의 평균 즉, 기댓값은 항상 ‘0’이 된다.
## Fisher Information
**피셔 정보(Fisher Information)** 는 Score Function의 **분산(Variance)** 을 의미한다.
![[Pasted image 20260722161616.png|500]]
뒤에서 살펴보겠지만 Fisher Information은 결국 스코어 함수의 미분의 평균 또는 로그 우도의 이계 미분, 즉 두 번 미분한 결과의 평균(기댓값)의 음수와 일치하게 된다.
수식으로 알아보자. Score Function의 분산을 구하는 식은 다음과 같다.
$$
I(\theta) = \mathrm{Var}\big[s(\theta)\big] = \mathrm{E}_f\big[s(\theta)^2\big] - \mathrm{E}_f\big[s(\theta)\big]^2
$$
위에서 스코어 함수의 평균, 즉 Expected Score Function이 0임을 밝혔다. 따라서 다음과 같다.
$$
I(\theta) = \mathrm{E}_f\big[s(\theta)^2\big] = \mathrm{E}_f\big[l'(\theta)^2\big] = \mathrm{E}_f\left[\left(\frac{\partial}{\partial \theta}\log f(x|\theta)\right)^2\right] = \int \left(\frac{\partial}{\partial \theta}\log f(x|\theta)\right)^2 f(x|\theta)\,dx
$$
이제 스코어 함수의 미분을 구해보자.
$$
\begin{align*}
s'(\theta) = l''(\theta) &= \frac{\partial^2}{\partial \theta^2} \log f(x|\theta) = \frac{\partial}{\partial \theta} \frac{f'(x|\theta)}{f(x|\theta)} \\
&= \frac{f''(x|\theta)f(x|\theta) - f'(x|\theta)^2}{f(x|\theta)^2} = \frac{f''(x|\theta)}{f(x|\theta)} - \left(\frac{f'(x|\theta)}{f(x|\theta)}\right)^2 \\
&= \frac{f''(x|\theta)}{f(x|\theta)} - \left(\frac{\partial}{\partial \theta}\log f(x|\theta)\right)^2
\end{align*}
$$
이 값의 평균(Expectation)을 구하면 어떻게 될까? 기댓값을 구해보면 다음과 같다.
$$
\mathrm{E}_f\big[s'(\theta)\big] = \mathrm{E}_f\left[\frac{f''(x|\theta)}{f(x|\theta)} - \left(\frac{\partial}{\partial \theta}\log f(x|\theta)\right)^2\right] = \mathrm{E}_f\left[\frac{f''(x|\theta)}{f(x|\theta)}\right] - \mathrm{E}_f\left[\left(\frac{\partial}{\partial \theta}\log f(x|\theta)\right)^2\right]
$$
마지막 식의 첫 번째 항을 구해보면 Expected Score Function을 구하는 방법과 마찬가지로 미분과 적분의 순서를 바꾸면 결국 0이 된다.
$$
\begin{align*}
\mathrm{E}_f\left[\frac{f''(x|\theta)}{f(x|\theta)}\right] &= \int \frac{f''(x|\theta)}{f(x|\theta)} \cdot f(x|\theta)\,dx = \int f''(x|\theta)\,dx \\
&= \int \frac{\partial^2}{\partial \theta^2} f(x|\theta)\,dx = \frac{\partial^2}{\partial \theta^2} \int f(x|\theta)\,dx = \frac{\partial^2}{\partial \theta^2} 1 = 0
\end{align*}
$$
따라서 Fisher Information은 다음과 같이 구해진다.
$$
\therefore I(\theta) = -\mathrm{E}_f\left[\frac{\partial^2}{\partial \theta^2}\log f(x|\theta)\right] = \mathrm{E}_f\left[\left(\frac{\partial}{\partial \theta}\log f(x|\theta)\right)^2\right]
$$
참고로 Fisher Information은 항상 0보다 크다.
$$
I(\theta) \geq 0
$$
여기서는 1차원으로 한정했지만, 차원을 높여 스칼라(Scalar)가 아닌 벡터(Vector)로 취급하면 Multi-dimension이 되어 분산은 **공분산(Covariance)** 이 되고, gradient는 **Hessian**이 된다.
![[Pasted image 20260722164052.png]]
## References
- https://blog.naver.com/ycpiglet/223109892874