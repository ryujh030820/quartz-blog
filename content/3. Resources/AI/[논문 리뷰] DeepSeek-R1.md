---
title: "[논문 리뷰] DeepSeek-R1"
date: 2026-07-20
tags:
  - AI
  - Paper
---
## Background
### DeepSeek-V3
DeepSeek-V3는 2024년 12월 공개된 DeepSeek의 오픈소스 LLM으로, OpenAI의 GPT-4나 Meta의 Llama 3.1과 견줄 수 있는 성능을 목표로 개발되었다.
MoE(Mixture-of-Experts) 아키텍처를 기반으로 총 6,710억 개의 파라미터를 가지며, 토큰당 370억 개만 활성화되어 효율성과 성능을 동시에 확보한다.
14.8조 개의 고품질 토큰으로 사전 학습된 후, 지도 미세 조정과 강화학습을 거쳐 다양한 영역에서의 능력을 향상시켰다.
효율적인 추론을 위한 **MLA(Multi-head Latent Attention)**, **보조 손실이 없는 부하 분산(auxiliary-loss-free load-balancing) 전략**, 성능 향상을 위한 **MTP(Multi-Token Prediction)** 등의 특징을 포함한다.

본 논문에서는 사전 학습만 마친 기반 모델을 **DeepSeek-V3-Base**, 이를 지도 미세 조정 및 강화학습으로 정렬한 모델을 **DeepSeek-V3**로 표기하며, DeepSeek-R1과 DeepSeek-R1-Zero는 모두 DeepSeek-V3-Base를 기반으로 학습된다.
### 기존 Post-Training 패러다임
사전 학습된 LLM을 특정 성능 목표에 맞게 정교화하는 후속 학습(post-training) 단계에서는 일반적으로 **지도 미세 조정(SFT)** 이후 **강화학습(RL)** 을 적용하는 2단계 프레임워크가 널리 사용된다.
SFT는 입력-출력 쌍으로 구성된 정제된 데이터셋으로 모델을 학습시켜 특정 작업에 대한 정밀한 정렬을 달성하지만, 데이터셋의 품질과 다양성에 성능이 크게 좌우되며 고정된 출력을 최적화하는 정적인 특성상 진화하는 인간의 선호도를 반영하기 어렵다.
RL은 보상 신호를 최대화하도록 모델의 출력을 최적화하는 단계로, RLHF가 대표적이다. SFT처럼 전체 입력-출력 쌍에 대한 라벨링이 필요하지 않아 주석 부담이 상대적으로 적다.
SFT와 RL의 순차적 적용은 각각의 장점을 결합한다. SFT가 정제된 예시를 기반으로 견고한 과제별 기준선을 확립하면, RL은 이를 더 넓은 인간 중심적 목표에 맞게 정교화한다.

본 연구는 SFT 단계가 오히려 모델의 효과적인 추론 전략 탐색을 저해할 수 있음을 지적한다. 인간이 제공한 응답은 명시적인 성찰 및 검증 단계와 같은 핵심 추론 요소를 종종 생략하기 때문이다. 이에 따라 DeepSeek-R1-Zero는 인간의 사전 지식과 무관하게 모델 스스로 추론 패턴을 직접 탐색하도록 설계된다.
### GRPO와 PPO의 비교
![[Pasted image 20260721094521.png]]
GRPO는 PPO의 학습 과정을 단순화하고 자원 소비를 줄이기 위해 제안된 강화학습 알고리즘이다.
PPO는 어드밴티지를 계산할 때 보상뿐만 아니라 정책 모델과 비슷한 크기의 학습된 **가치 모델(value model)** 을 기반으로 한 GAE(Generalized Advantage Estimation)를 사용하며, 이는 상당한 메모리 및 연산 오버헤드를 유발한다. 특히 긴 CoT를 학습할 때는 부분 응답만으로 최종 보상을 예측하기 어렵다는 근본적인 한계도 있다.
반면 GRPO는 별도의 가치 모델 없이 그룹 내 보상 점수로부터 직접 어드밴티지를 추정하여 이러한 오버헤드를 제거한다.
KL divergence를 다루는 방식에도 차이가 있다. GRPO는 비편향 추정량을 손실 함수에 직접 더하는 반면, PPO는 토큰별 KL 페널티를 밀집 보상(dense reward) 형태로 추가한다.
MATH 과제 실험에서 PPO는 GAE의 $\lambda$ 계수 튜닝에 민감하여 적절히 튜닝하지 않으면 GRPO보다 성능이 크게 저하되는 반면, GRPO는 추가적인 하이퍼파라미터 튜닝 없이도 우수한 성능을 보인다.
## Introduction
인공지능 분야의 최근 발전은 LLM이 충분한 규모로 확장되었을 때 추론 능력을 포함한 창발적 행동을 보일 수 있음을 입증했다.
그러나 사전 학습 단계에서 이러한 능력을 달성하려면 일반적으로 막대한 컴퓨팅 자원이 필요하다.
또한 기존의 접근법들은 효과적임에도 불구하고 몇 가지 주목할 만한 한계를 보인다.
1. 인간이 주석을 단 추론 흔적에 대한 의존성은 확장성을 저해하고 인지적 편향을 유발한다.
2. 모델이 인간의 사고 과정을 복제하도록 제한함으로써 모델의 성능은 본질적으로 인간이 제공한 예제에 의해 제한되며, 이는 인간과 다른 방식의 더 우수한 추론 경로를 탐색하는 것을 방해한다.

이러한 문제를 해결하기 위해, 본 연구는 인간의 라벨링 노력에 대한 의존도를 최소화하면서 강화 학습(RL) 프레임워크 내에서 자기 진화(self-evolution)를 통해 LLM의 추론 능력을 개발할 가능성을 탐구하고자 한다.
구체적으로, 본 연구는 DeepSeek-V3-Base를 기반으로 구축하고 **GRPO(Group Relative Policy Optimization)** 를 RL 프레임워크로 채택한다.
보상 신호는 추론 과정 자체에 제약을 가하지 않고, 정답(ground-truth)에 대한 최종 예측의 정확성에만 전적으로 기반한다.
특히, 본 연구는 RL 학습 전의 일반적인 지도 미세 조정 단계(SFT) 단계를 생략한다.
이러한 설계 선택은 인간이 정의한 추론 패턴이 모델의 탐색을 제한할 수 있는 반면, 제한없는 RL 학습은 LLM에서 새로운 추론 능력의 발현을 더 잘 촉진할 수 있다는 가설에서 비롯되었다.

DeepSeek-R1-Zero는 뛰어난 추론 능력을 보여주지만, 가독성이 떨어지고 언어가 혼용되는 등의 문제에 직면한다.
또한, DeepSeek-R1-Zero의 규칙 기반 RL 학습 단계는 추론 작업에만 좁게 초점을 맞추고 있어, 글쓰기나 오픈 도메인 질의응답과 같은 더 넓은 영역에서는 제한된 성능을 보인다.
이러한 문제를 해결하기 위해, 본 연구는 **기각 샘플링(rejection sampling)**, 강화 학습, 그리고 지도 미세 조정을 통합한 다단계 학습 프레임워크를 통해 훈련된 모델인 DeepSeek-R1을 도입한다.
## DeepSeek-R1-Zero
### Group Relative Policy Optimization
GRPO는 DeepSeek-R1-Zero와 DeepSeek-R1을 학습시키기 위해 채택한 강화 학습 알고리즘이다.
이는 원래 LLM의 RL 단계에서 널리 사용되는 PPO(Proximal Policy Optimization)의 학습 과정을 단순화하고 자원 소비를 줄이기 위해 제안되었다.

각 질문 $q$에 대해, GRPO는 이전 정책 $\pi_{\theta_{old}}$로부터 출력 그룹 $\left\{o_1, o_2, \dots, o_G\right\}$를 샘플링한 다음, 다음 목적 함수를 최대화하여 정책 모델 $\pi_\theta$를 최적화한다.
$$
\begin{gathered}
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)] \\
\frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta \mathbb{D}_{KL} (\pi_\theta || \pi_{ref}) \right),
\end{gathered}
$$
$$
\mathbb{D}_{KL} (\pi_{\theta} || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,
$$
여기서 $\pi_{ref}$는 참조 정책이고, $\epsilon$과 $\beta$는 하이퍼파라미터이며, $A_i$는 각 그룹 내 출력에 해당하는 보상 그룹 $\left\{r_1, r_2, \dots, r_G\right\}$를 사용하여 계산된 어드밴티지(advantage)이다.
$$
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}.
$$
### Reward Design
보상은 RL 최적화의 방향을 결정하는 학습 신호의 원천이다.
DeepSeek-R1-Zero의 경우, 수학, 코딩 및 논리적 추론 영역의 데이터에 대해 정확한 피드백을 제공하기 위해 규칙 기반 보상을 사용한다.
본 연구의 규칙 기반 보상 시스템은 주로 정확도 보상과 형식 보상이라는 두 가지 유형의 보상으로 구성된다.

**정확도 보상**은 응답이 올바른지 평가한다. 예를 들어, 수학 문제의 경우, 모델은 최종 답을 지정된 형식(예: 상자 안)으로 제공해야 하며, 이를 통해 정확성에 대한 신뢰할 수 있는 규칙 기반 검증이 가능하다.
**형식 보상**은 특정 형식 요구 사항을 강제함으로써 정확도 보상 모델을 보완한다.
특히, 모델은 추론 과정을 ‘\<think\>’ 및 ‘\</think\>‘와 같은 지정된 태그 내에 포함하도록 유도된다.
$$
Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}
$$
정확도 보상과 형식 보상은 동일한 가중치로 결합된다. 특히, 본 연구는 추론 작업에 신경망 보상 모델을 적용하지 않는다.
이러한 결정은 대규모 강화학습 과정에서 신경망 보상 모델이 **보상 해킹(reward hacking)** 이 취약하다는 관찰에 근거한다.
또한, 이러한 모델을 재학습하는 데는 상당한 컴퓨팅 자원이 필요하며 학습 파이프라인에 추가적인 복잡성을 도입하여 전반적인 최적화 과정을 어렵게 만든다.
### Incentivize Reasoning Capability in LLMs
구체적으로, 본 연구는 DeepSeek-V3 베이스 모델에 강화학습 기법을 적용하여 DeepSeek-R1-Zero를 학습시킨다.
학습 과정에서 본 연구는 DeepSeek-R1-Zero가 먼저 추론 과정을 생성하고 그 뒤에 최종 답변을 제시하도록 요구하는 간단한 템플릿을 설계한다.

아래 그림은 강화학습 과정 전반에 걸쳐 AIME 2024 벤치마크에서 DeepSeek-R1-Zero의 성능 궤적을 보여주며, AIME 2024의 평균 pass@1 점수는 초기 15.6%에서 77.9%로 크게 상승했다.
또한, 자기 일관성 디코딩(self-consistency decoding)을 활용함으로써 모델의 성능을 더욱 향상시켜 86.7%의 정확도를 달성할 수 있었다.
![[Pasted image 20260720170733.png]]
DeepSeek-R1-Zero의 자기 진화(self-evolution)는 RL이 어떻게 모델의 추론 능력을 자율적으로 향상시킬 수 있는지를 잘 보여준다.
위 그림의 오른쪽에서 나타난 바와 같이, DeepSeek-R1-Zero는 외부 수정이 아닌 오직 내재적 적응에 의해 주도되어 학습 과정 전반에 걸쳐 사고 시간이 꾸준히 증가하는 모습을 보인다.

사고 시간의 증가는 정교한 행동의 자율적 발달을 촉진한다.
구체적으로, DeepSeek-R1-Zero는 성찰적 추론 및 대안적 해결책에 대한 체계적 탐색과 같은 고급 추론 전략을 점점 더 많이 보여주며, 수학 및 코딩과 같은 검증 가능한 작업에서 성능을 크게 향상시킨다.
특히 학습 도중 DeepSeek-R1-Zero는 성찰 과정에서 “wait”이라는 단어 사용이 갑자기 증가하는 것으로 특징지어지는 “아하 모먼트(aha moment)”를 보인다. (아래 참조)
![[Pasted image 20260720171212.png|500]]
## DeepSeek-R1
DeepSeek-R1-Zero는 강력한 추론 능력을 보여주지만, 몇 가지 문제점에 직면해 있다.
DeepSeek-V3-Base가 여러 언어, 특히 영어와 중국어로 학습되었기 때문에 DeepSeek-R1-Zero는 가독성 저하 및 언어 혼용과 같은 문제로 어려움을 겪는다.
이러한 문제를 해결하기 위해 본 연구는 DeekSeek-R1을 개발했으며, 그 파이프라인은 아래 그림에 나타나 있다.
![[Pasted image 20260720171957.png]]
초기 단계에서 본 연구는 대화형이며 인간의 사고 과정에 부합하는 수천 개의 콜드 스타트(cold-start) 데이터를 수집한다.
이후 강화 학습을 적용하여 대화형 사고 과정과 언어 일관성을 바탕으로 모델 성능을 향상한다. 그다음, 기각 샘플링과 지도 미세 조정을 한 번 더 적용한다.
이 단계에서는 추론 데이터셋과 비추론 데이터셋을 모두 SFT 과정 통합하여, 모델이 추론 작업에서 뛰어난 성능을 발휘할 뿐만 아니라 고급 작문 능력까지 갖추도록 한다.
모델을 인간의 선호도에 더욱 맞추기 위해, 모델의 유용성과 무해성을 강화하는 동시에 추론 능력을 정교화하도록 설계된 2차 강화 학습 단계를 구현한다.

본 섹션의 나머지 부분에서는 이 파이프라인의 핵심 구성 요소를 자세히 설명한다.
### Model-based Rewards
일반적인 데이터의 경우, 복잡하고 미묘한 시나리오에서 인간의 선호도를 포착하기 위해 보상 모델을 사용한다.
**유용성(helpfulness)** 의 경우, 최종 요약에만 집중하여 평가가 사용자에 대한 응답의 유용성과 관련성을 강조하도록 하고, 근본적인 추론 과정에 대한 간섭을 최소화한다.
**무해성(harmlessness)** 의 경우, 생성 과정에서 발생할 수 있는 잠재적 위험, 편향 또는 유해 콘텐츠를 식별하고 완화하기 위해 추론 과정과 요약을 포함한 모델의 전체 응답을 평가한다.

**유용성 보상 모델(Helpful Reward Model)** 과 관련하여, 본 연구는 arena-hard 프롬프트 형식을 사용하여 DeepSeek-V3에 프롬프트를 입력함으로써 선호도 쌍을 생성하며, 각 쌍은 사용자 질의와 두 개의 후보 응답으로 구성된다.
각 선호도 쌍에 대해, 위치 편향을 완화하기 위해 응답을 무작위로 Response A 또는 Response B로 할당하여 DeepSeek-V3에 네 번 질의한다.
최종 선호도 점수는 네 번의 독립적인 판단을 평균하여 결정되며, 의미 있는 구분을 보장하기 위해 점수 차이($\Delta$)가 1을 초과하는 쌍만 유지한다.
또한, 길이와 관련된 편향을 최소화하기 위해 전체 데이터셋에서 선택된 응답과 거부된 응답의 길이가 비슷하도록 보장한다.
$$
Reward_{helpful} = RM_{helpful}(Response_{A}, Response_{B})
$$
**안전성 보상 모델(Safety Reward Model)** 을 평가하고 개선하기 위해, 본 연구는 사전 정의된 안전 가이드라인에 따라 “안전(safe)” 또는 “불안전(unsafe)”으로 주석이 달린 모델 생성 응답을 포함하는 프롬프트 데이터셋을 구축했다.
유용성 보상 모델에 사용된 쌍별(pair-wise) 손실과 달리, 안전성 보상 모델은 포인트별(point-wise) 방법론을 사용하여 학습되었다.
$$
Reward_{safety} = RM_{safety}(Response)
$$
### Training Details
#### Training Details of the First RL Stage
첫 번째 RL 단계에서 언어 혼용 문제를 완화하기 위해 RL 학습 중에 언어 일관성 보상을 도입하며, 이는 CoT 내 타겟 언어 단어의 비율로 계산된다.
$$
Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}
$$
Ablation experiments에서 이러한 정렬이 모델 성능을 다소 저하시키는 것으로 나타났지만, 이 보상은 인간의 선호도와 일치하여 가독성을 높여준다.
언어 일관성 보상을 추론 데이터와 비추론 데이터 모두에 최종 보상으로 직접 추가하여 적용한다.

클립 비율(clip ratio)이 학습에 중요한 역할을 한다는 점에 유의해야 한다.
값이 낮으면 상당수의 토큰에 대해 그래디언트가 잘릴 수 있어 모델 성능이 저하될 수 있으며, 값이 높으면 학습 중 불안정성이 발생할 수 있다.
#### Training Details of the First RL Stage
추론 데이터의 경우, 수학, 코딩 및 논리적 추론 영역에서 학습을 유도하기 위해 규칙 기반 보상을 사용하는 DeepSeek-R1-Zero의 방법론을 따른다.
학습 과정에서 CoT가 종종 언어 혼용을 보이는 것을 관찰했으며, 특히 RL 프롬프트가 여러 언어를 포함할 때 이러한 현상이 두드러진다.
일반 데이터의 경우, 보상 모델을 활용하여 학습을 유도한다. 궁극적으로, 보상 신호와 다양한 데이터 분포의 통합을 통해 추론 능력뿐만 아니라 유용성과 무해함을 우선시하는 모델을 개발할 수 있다.
$$
\begin{gathered}
&Reward = Reward_{\text{reasoning}} + Reward_{\text{general}} + Reward_{\text{language}} \\
&\text{where, } Reward_{\text{reasoning}} = Reward_{\text{rule}} \\
&Reward_{\text{general}} = Reward_{\text{reward\_model}} + Reward_{\text{format}}
\end{gathered}
$$
## References
- https://arxiv.org/pdf/2501.12948
## Backlinks
- [[[개념 정리] MoE(Mixture of Experts)]]
- [[[개념 정리] Multi-head Latent Attention]]