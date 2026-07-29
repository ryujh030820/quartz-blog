---
title: "[논문 리뷰] Survey on Knowledge Distillation for Large Language Models"
date: 2026-07-23
tags:
  - AI
  - Paper
---
## Introduction
LLM은 뛰어난 성능을 보이지만, 파라미터 규모와 연산량이 커서 실제 배포 환경에서는 추론 비용과 메모리 요구량이 큰 문제가 된다.
예를 들어 GPT-3는 float16 기준 약 350GB의 저장 공간과, 각 80GB 메모리를 가진 A100 GPU 5개 이상을 필요로 한다.
이러한 문제를 완화하기 위한 모델 압축(model compression) 기법 중 하나가 **Knowledge Distillation(KD)** 이며, 큰 teacher 모델이 학습한 지식을 작은 student 모델로 전이시켜 성능 저하를 최소화하면서 추론 속도를 높이는 방법이다.

기존의 KD 서베이들은 대부분 일반적인 모델 압축 관점에 머물러 있었는데, LLM의 등장은 KD에 다음과 같은 새로운 과제를 제기한다.
1. LLM은 단일 태스크가 아니라 다양한 태스크와 미지의 데이터에 대한 폭넓은 일반성(generality)을 목표로 설계되므로, 압축된 LLM의 일반화 능력을 평가하려면 더 신중하고 철저한 평가가 필요하다.
2. 기존 서베이는 KD 기법을 실제 시나리오에 적용한 구체적 사례 없이 단순 요약에 그친다.

이 논문은 이를 해결하기 위해 KD 알고리즘을 **method, evaluation, application** 세 가지 관점에서 다루며, method는 다시 **white-box KD**(teacher 내부 정보에 접근 가능)와 **black-box KD**(teacher의 출력만 접근 가능)로 구분한다.
## Overview of KD — 최적화 목적함수
### Logits-Based KD
Logits-based KD는 teacher 모델의 logit을 지식 전달의 매개로 사용하는 패러다임이다. 일반적인 KD loss는 다음과 같다.
$$
\mathcal{L}_{logits} = KL(p^t \| p^s) = \sum_{j=1}^{C} p^t_j \log\left(\frac{p^t_j}{p^s_j}\right)
$$
$$
p^s_i = \frac{\exp(z^s_i/\tau)}{\sum_{j=1}^{C}\exp(z^s_j/\tau)}, \quad p^t_i = \frac{\exp(z^t_i/\tau)}{\sum_{j=1}^{C}\exp(z^t_j/\tau)}
$$
여기서 $z^s, z^t \in \mathbb{R}^C$는 각각 student와 teacher의 logit이고, $\tau$는 logit의 매끄러움(smoothness)을 조절하는 temperature, $C$는 클래스 개수이다.
KL divergence 대신 **Reverse KL(RKL)**, **Jensen-Shannon(JS) divergence** 등으로 대체할 수도 있다.
### Hint-Based KD
Logits-based KD에서 student가 습득할 수 있는 지식이 제한적이라는 문제의식에서, teacher의 중간 layer 출력(intermediate feature)까지 맞추는 hint-based KD가 도입되었다. 일반적인 형태는 다음과 같다.
$$
\mathcal{L}_{hint} = \mathcal{H}(F^s, F^t) = \|F^t - \phi(F^s)\|^2
$$
$F^s, F^t \in \mathbb{R}^{H\times W\times C}$는 각각 student와 teacher의 중간 feature이고, $\phi$는 student feature의 차원을 teacher와 맞춰주는 함수이며, $\mathcal{H}$는 거리 함수(예시로는 MSE)이다.
### ICL, CoT, Instruction Following (Black-box KD의 세 축)
Black-box KD는 teacher의 출력만 관찰 가능한 API 기반 상황을 다루며, 다음 세 가지 방식으로 지식을 student에 전달한다.
- **ICL**: task 설명과 $k$개의 예제 $f(x_1,y_1),...,f(x_k,y_k)$를 시연(demonstration)으로 제공해 $\hat{y}_{k+1}$을 예측하게 한다.
- **CoT**: ICL의 input-output 쌍에 rationale $r_k$를 추가하여, student가 정답뿐 아니라 그 근거까지 모사하도록 한다.
$$
\text{LLM}(I, f(x_1,r_1,y_1),...,f(x_k,r_k,y_k), f(x_{k+1}, \_, \_)) \rightarrow \hat{r}_{k+1}, \hat{y}_{k+1}
$$
- **Instruction Following**: 구조화된 멀티태스크 데이터셋으로 fine-tuning하여, 명시적 예제 없이도 새로운 지시문 기반 태스크를 수행하게 만든다.
## White-Box KD
### Logits-Based KD
**DistilBiLSTM**은 BERT를 BiLSTM으로 증류한 초기 시도로, student와 teacher logit 간 MSE를 최소화하며 ELMo와 동등한 성능을 100배 적은 파라미터, 15배 빠른 속도로 달성했다.
**DistillBERT**는 teacher의 파라미터로 얕은 student를 초기화하고, language modeling + distillation + cosine distance를 결합한 triple loss로 학습하며, 파라미터를 40% 줄이면서 BERT 성능의 97%를 유지했다.
**MixKD**는 예제 쌍의 선형 보간(linear interpolation)으로 데이터를 증강해 KD 효율을 높인다.
**ReAugKD**는 inference 단계에서 student 임베딩과 유사한 teacher soft label을 지식베이스에서 검색해 활용하고, training 단계에서는 teacher-student 임베딩 분포 차이를 최소화하는 relational KD loss를 사용한다. baseline 대비 3% 미만의 latency 오버헤드로 우수한 성능을 달성했다.
**PD(Pre-Training Distillation)**는 표준 3단계 학습 시퀀스로 구성된 범용 압축 알고리즘으로, 임의의 아키텍처에 적용 가능하며 평균적으로 teacher 성능을 능가하기도 했다.

모델 규모가 커지며 GLUE/BERT 중심 평가로는 한계가 생기는데, 이에 **MINILLM**은 free-running generation 상황에서의 forward KLD 최소화가 갖는 한계를 지적하고 **reverse KLD**로 대체하여 student가 teacher 분포의 저확률 영역을 과대추정하는 것을 방지한다. 학습 안정화를 위해 (1) 분산 감소를 위한 single-step decomposition, (2) reward hacking 완화를 위한 teacher mixed sampling, (3) 길이 편향 제거를 위한 length normalization을 도입했다.
**GKD**는 고정된 출력 시퀀스 집합에만 의존하지 않고 student가 스스로 시퀀스를 생성하며 teacher의 피드백을 받도록 하고, reverse KL·generalized JS 등 대체 divergence를 유연하게 사용할 수 있어 distillation과 RL fine-tuning의 결합을 가능케 한다.
**$f$-DISTILL**은 sequence-level KD를 일반화된 $f$-divergence 최소화로 정식화하며, 기존 SeqKD·ENGINE이 각각 KL·reverse KL distillation의 근사임을 보이고, step-wise decomposition으로 word-level loss 계산을 용이하게 한다.
**MiniMA**는 student가 teacher 크기의 약 40%일 때 최적의 증류 효과가 나타남을 발견하고, structured pruning과 logit-based KD를 결합했다.
### Hint-Based KD
**PKD(Patient KD)**는 teacher의 마지막 $k$개 layer(PKD-Last) 또는 $k$-layer마다(PKD-Skip) 학습하는 두 전략을 제안한다.
**MetaDistil**은 teacher를 고정한 채 meta-learning 프레임워크 안에서 student 성능에 대한 피드백을 distill한다.
**AD-KD**는 gradient 기반 attribution으로 입력 토큰의 중요도를 계산하고, top-K 필터링으로 덜 중요한 차원을 제거한 뒤 teacher의 여러 잠재적 예측에 대한 multi-view attribution을 distill한다.
**XtremeDistil**은 41개 언어의 다국어 NER에 적용되어 파라미터를 35배, 배치 추론 지연을 51배 줄이면서 95% 성능을 유지했다.
**TinyBERT**는 embedding layer, hidden state, attention matrix, transformation layer 등 다양한 layer에서 지식을 추출한다.
**MiniLM**은 teacher 마지막 layer의 self-attention만을 깊이 모방(deep self-attention distillation)하는 task-agnostic 방식으로, teacher-student 간 layer 수 제약 없이 SQuAD2·GLUE에서 파라미터 50%로 99% 이상의 정확도를 유지했다.
**TED**는 task-aware filter로 불필요한 정보를 제거해 underfitting 문제를 완화한다.
**MobileBERT**는 depth 대신 width를 조정하는 방식(bottleneck/inverted bottleneck)을, **HomoDistil**은 attention과 hidden state에 대한 반복적 pruning 방식을 취한다.
## White-Box KD의 한계와 로버스트니스 평가
Logits-based KD는 출력 분포 정렬에 집중하는 반면 hint-based KD는 중간 layer까지 정렬해 더 풍부한 정보를 전달할 수 있지만, teacher-student 간 layer mapping 설계에 아키텍처에 대한 깊은 이해가 필요하고, forward 단계의 활성값 저장으로 인해 GPU 메모리 소모가 크다는 공통적 한계가 있다.

이 논문은 AdvGLUE·ANLI(adversarial robustness, ASR 지표)와 Flipkart·DDXPlus(OOD robustness, F1 지표)로 GPT-2, OPT, LLaMA/LLaMA2에 대해 5개 distillation 알고리즘의 로버스트니스를 통일된 기준으로 평가했다. 그 결과 GPT-2에서는 MINILLM이 대체로 최고 성능을, OPT에서는 가장 단순한 KD가 최고 성능을 보였으며, LLaMA·LLaMA2에서는 SeqKD·JS가 각각 우세해 동일한 모델 크기·구조에서도 최적 알고리즘이 달라질 수 있음을 보였다.
## Black-Box KD
### ICL
**ICL Distillation**은 Meta-ICT(다양한 태스크에 대한 meta-training 후 ICL로 적응)와 Multitask-ICT(모든 target task를 학습 태스크로 취급해 직접 ICL distillation) 두 패러다임을 제안하며, 후자는 모델 크기를 93% 줄이면서 teacher 성능의 91.4%를 유지했다.
**LLM-R**은 고정된 LLM으로 후보 예제를 검색하고 랭킹을 매겨 reward model을 학습한 뒤, 이를 dual-encoder 기반 dense retriever로 distill하여 ICL 성능을 평균 7.8% 향상시켰다.
### CoT
**Distilling step-by-step**은 LLM을 단순한 label 소스가 아니라 자신의 예측을 정당화하는 자연어 추론의 제공자로 재정의해, 학습 샘플 수를 50% 이상(일부는 85% 이상) 줄이면서도 더 작은 모델(770M T5)로 540B 파라미터 LLM을 능가했다.
**Fine-tune-CoT**는 teacher로부터 random sampling으로 여러 추론 해법을 생성해 데이터를 증강하며, 0.3B 파라미터 모델이 175B teacher를 능가하는 사례를 보였다.
**MCC-KD**는 하나의 질문에 대해 여러 rationale을 생성하고 정답 분포 간 양방향 KLD를 최소화해 추론의 일관성을 강화한다.
**SOCRATIC CoT**는 문제를 하위 문제로 분해하는 decomposer와 이를 푸는 solver 한 쌍의 소형 모델로 복잡한 문제를 협력적으로 해결한다.
**SCOTT**는 teacher가 정답과 실제로 연관된 rationale만 생성하도록 **비교 디코딩(contrastive decoding)** 을 활용해 지식 전달의 충실도를 높이는 것이 핵심이다.
### Instruction Following
**Self-Instruct**는 소량의 수작업 seed task로부터 모델 스스로 더 넓은 범위의 task 설명과 input-output 인스턴스를 생성하고, 휴리스틱으로 저품질·중복 task를 걸러내는 반자동 프로세스이며, Alpaca·Vicuna·GPT4All 등에 영향을 주었다.
Peng et al.은 GPT-4로 52,000개의 instruction-following 데이터(영어·중국어)를 생성해 LLaMA-GPT4를 fine-tuning했다.
**LaMini-LM**은 258만 개의 instruction을 GPT-3.5 Turbo로 응답 생성해 다양한 encoder-decoder/decoder-only 모델을 fine-tuning했으며, 10분의 1 크기로 경쟁력 있는 성능을 냈다.
**Lion**은 imitation-discrimination-generation 3단계의 적대적 증류 프레임워크로, teacher가 "어려운" instruction을 식별해 student에게 새로운 instruction을 생성하게 함으로써 70,000개 샘플만으로 ChatGPT급 open-generation 능력을 달성했다.
**UniversalNER**는 instruction 다양성이 아닌 input 다양성 확장에 초점을 맞춰 도메인 간 일반화를 향상시켰다.
### Black-Box KD의 로버스트니스 평가 및 논의
Black-box KD는 teacher가 생성한 데이터만으로 student를 학습시키므로 메모리 효율적이지만, teacher가 closed-source인 경우가 많아 추가 데이터 생성 비용이 크고 공정한 비교가 어렵다는 한계가 있다. CoT 증류에 대한 로버스트니스 평가에서는, GPT-2·OPT 소형 모델에서는 ANLI·e-SNLI 기반 증류가 유리했으나 모델이 커질수록 그 설명력이 줄었고, 반대로 상식 데이터(CQA)·수학 데이터(SVAMP) 기반 증류는 모델 크기와 무관하게 OOD 일반화(Flipkart, DDXPlus)에 더 강건했다.
## Applications
- **Healthcare**: HuatuoGPT(ChatGPT distillation + 의사 피드백 reward model), Chatdoctor(10만 건 환자-의사 대화로 fine-tuning), PMC-LLaMA(480만 편의 생의학 논문 지식 통합)
- **Education**: DARWIN(과학 발견 자동화), WizardMath(Evol-Instruct 기반 수학 추론 강화), K2(지구과학 특화 LLM)
- **Law**: LawyerLLaMA(법률 조항 검색 모듈 결합), ChatLaw(벡터·키워드 하이브리드 검색으로 hallucination 완화)
## Challenges and Future Directions
1. **통합 평가 벤치마크**: GLUE, MMLU, BIG-Bench, HELM 등 벤치마크가 파편화되어 있어 KD를 위한 통일된 평가 기준 마련이 필요하다.
2. **고급 알고리즘**: symbolic KD, DISCO 등 특정 능력 강화에 집중한 방법들이 있으나, open-source LLM 발전에 따라 white-box distillation과 멀티모달 distillation 연구가 더 필요하다.
3. **해석 가능성(Interpretability)**: Stanton et al.의 연구에 따르면 student의 일반화 성능과 teacher와의 matching degree(fidelity)가 항상 비례하지는 않으며, fidelity가 높은 모델이 calibration은 더 우수한 경향이 있다. LLM KD에서도 CoT 증류가 어떻게 reasoning 능력을 전이시키는지에 대한 설명력이 부족해, 해석 가능성 통합이 향후 과제로 남아있다.
## Conclusion
이 서베이는 LLM을 위한 KD를 method(white-box/black-box), evaluation(robustness 포함), application(healthcare, education, law) 세 관점에서 정리했다. 소형 모델 압축에 쓰이던 기존 프레임워크에 의존하는 방법이 여전히 많아, LLM 규모에 특화된 압축 알고리즘 개발과 통합 평가 체계 구축이 앞으로의 핵심 과제로 제시된다.
## References
- https://doi.org/10.1145/3699518