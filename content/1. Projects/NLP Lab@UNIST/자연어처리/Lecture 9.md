---
title: Lecture 9
description: UNIST 나승훈 교수님의 수업 '자연어처리' 9강 PPT 내용에 대해 다룹니다.
date: 2026-07-03
tags:
  - AI
  - Lecture
---
## 1. Subword Modeling
### 1.1 기존 word-level 모델의 문제
고정된 vocabulary(수만 개 단어)를 학습 데이터로부터 구축하고, 테스트 시 새로운 단어는 전부 `UNK`로 매핑. 오탈자("taaaaasty"), 변형("laern"), 신조어("Transformerify") 모두 `UNK` 하나로 뭉개져 정보 손실.
### 1.2 극단적 해법 1 — Character 단위
**Character-based NMT [Ling et al. '16]**: 인코더에서 단어를 문자 시퀀스로부터 합성(LSTM/ConvNet), 디코더에서도 단어를 문자 단위로 하나씩 생성: 
![[Pasted image 20260703212833.png]]
$$ P(w_p \mid \mathbf{a},\mathbf{l}^f_{p-1}) = \prod_{i\in[0,y]} P(w_{j,i}\mid w_{k,0},\ldots,w_{j,j-1},\mathbf{a},\mathbf{l}^f_{p-1}) $$
- $\mathbf{a}$: aligned source word(attention 출력), $w_{j,i}$: $j$번째 단어의 $i$번째 문자
- **문제**: 모든 단어를 문자 단위로 생성 → 계산량 매우 큼
### 1.3 절충안 — Hybrid NMT [Luong & Manning '16]
![[Pasted image 20260703213043.png|500]]
**"평소엔 word-level, 필요할 때만 character-level"**:
- 아는 단어: 빠른 word-level 처리
- `<unk>`가 나온 자리만: character-level encoder/decoder로 전환해서 세부 처리
- Word-level(4층) + character-level 인코더/디코더를 **end-to-end로 8층까지 stacking**해서 함께 학습
### 1.4 표준 해법 — Byte Pair Encoding (BPE)
문자와 단어 사이의 **subword** 단위를 학습 데이터로부터 자동으로 결정.
```
procedure BPE(D, k):                     # D: 문자열 집합, k: 목표 vocab 크기
    V ← D의 모든 고유 문자 (약 4,000개, 영어 위키 기준)
    while |V| < k do:
        t_L, t_R ← D에서 가장 빈번한 인접 bigram
        t_NEW ← t_L + t_R
        V ← V + [t_NEW]                  # ← V에는 "추가"만, 제거 없음!
        D 내 모든 (t_L, t_R) → t_NEW로 치환
    return V
```
**핵심 포인트**: $|V|$는 매 iteration마다 **정확히 1씩 증가**한다 ($t_L, t_R$을 $V$에서 제거하지 않고 $t_{NEW}$만 추가하므로). "교체"는 코퍼스 $D$에서만 일어나고, 사전 $V$에는 누적된다. 따라서 while 루프는 $k - |V_{\text{초기}}|$번 후 반드시 종료.
**토큰화 예시 — "longest"**:

|Step|연산|결과|
|---|---|---|
|1|`e`+`s`→`es`|`['l','o','n','g','es','t','</w>']`|
|2|`es`+`t`→`est`|`['l','o','n','g','est','</w>']`|
|3|`est`+`</w>`→`est</w>`|`['l','o','n','g','est</w>']`|
|4|`l`+`o`→`lo`|`['lo','n','g','est</w>']`|
|5|`lo`+`w` → **적용 불가**|(현재 시퀀스에 `lo` 다음이 `n`이지 `w`가 아니므로 규칙 매칭 실패)|
|최종||`['lo','n','g','est</w>']`|
merge 규칙은 학습 코퍼스 전체에서 만들어진 것이므로, 특정 단어를 토큰화할 때 그 규칙에 해당하는 인접 쌍이 실제로 없으면 조용히 skip됨.
이후 WordPiece(BERT), SentencePiece 등 유사한 방법이 pretrained 모델의 표준 tokenizer로 자리잡음.
## 2. Word Embedding → Model Pretraining으로의 동기 전환
### 2.1 문맥성(Contextuality)의 필요성
> "You shall know a word by the company it keeps" (Firth, 1957) — distributional semantics, word2vec의 motivation "the complete meaning of a word is always **contextual**" (Firth, 1935)

"I **record** the **record**" — 같은 단어라도 문맥에 따라 의미가 전혀 다름. 그러나 word2vec류의 정적(static) 임베딩은 문맥과 무관하게 항상 동일한 벡터를 부여.
### 2.2 2017년경의 접근과 그 한계
기존: pretrained word embedding(문맥 없음)으로 시작 → LSTM/Transformer가 downstream task 학습 중에 문맥 반영법을 배움
- 문제 1: downstream task의 labeled data가 언어의 모든 문맥적 측면을 가르치기엔 턱없이 부족
- 문제 2: 네트워크 파라미터 대부분이 **무작위 초기화** 상태로 시작
### 2.3 전환: 모델 전체를 Pretrain
입력의 일부를 가리고, 그것을 복원하도록 학습(self-supervised) → 언어의 강력한 표현, 좋은 파라미터 초기화, 샘플링 가능한 확률분포를 동시에 얻음.
**"빈칸 채우기"를 통해 무엇을 배울 수 있는가** (슬라이드의 예시들이 보여주는 것):
- 사실적 지식: "Stanford University is located in ___ , California." → Stanford
- 구문적 지식: "I put ___ fork down on the table." → a/my (관사/한정사)
- 자연어 추론: "The woman walked across the street, checking for traffic over ___ shoulder." → her (공지시)
- 어휘적 의미: "I went to see the fish, turtles, seals, and ___ ." → sea lions 등 (범주적 지식)
- 감성 추론: "...The movie was ___ ." → 문맥으로부터 긍정/부정 추론
- 개체 추적: "Iroh went to make tea. Zuko left the ___ ." → kitchen (담화 추적)
- 산술/패턴: "1, 1, 2, 3, 5, 8, 13, 21, ___ ." → 34 (피보나치 패턴)
즉 언어모델링(다음 단어/빈칸 예측)이라는 **하나의 목적함수**만으로도 문법, 사실, 추론, 감성, 심지어 산술 패턴까지 암묵적으로 학습된다는 것.
### 2.4 Pretrain / Finetune 패러다임
**Step 1) Pretrain**: 대량의 unlabeled 텍스트로 language modeling 등 self-supervised 과제 학습 → 일반적인 언어 지식 습득 **Step 2) Finetune**: 소량의 labeled 데이터로 목표 downstream task에 맞춰 전체(혹은 일부) 파라미터를 추가 학습
**왜 효과적인가 (최적화 관점)**: pretraining이 $\hat\theta \approx \arg\min_\theta \mathcal{L}_{\text{pretrain}}(\theta)$를 찾아주고, finetuning은 $\hat\theta$에서 시작해 $\min_\theta\mathcal{L}_{\text{finetune}}(\theta)$를 찾음. SGD는 초기점 $\hat\theta$ 근방에 상대적으로 머무는 경향이 있으므로:
- $\hat\theta$ 근방의 국소최적해들이 일반화가 잘 될 가능성이 높고
- $\hat\theta$ 근방에서의 gradient 흐름이 (이미 잘 학습된 표현 덕분에) 더 원활할 가능성이 있음
### 2.5 왜 비지도학습(Self-supervised)인가
**데이터 규모의 압도적 차이**:

|데이터셋|토큰 수|
|---|---|
|SQuAD 2.0 (labeled QA)|< 5천만|
|DCLM-pool|240조|
|추정 전체 인터넷 텍스트|3,100조|
QA labeled 데이터와 인터넷 텍스트 사이에 **약 1000만 배**의 격차 → labeled data만으로는 도달 불가능한 규모의 학습 가능.
**다양성(diversity)도 핵심**: 단순히 양이 많은 게 아니라, 인터넷 텍스트가 다루는 주제·장르·스타일의 폭이 엄청나게 넓어서 광범위한 downstream task에 대해 (약하게라도) 사전 커버리지를 제공.
## 3. 아키텍처별 Pretraining 세 갈래

| 아키텍처                | 특징                   | 자연스러운 용도                         |
| ------------------- | -------------------- | -------------------------------- |
| **Encoder**         | 양방향 문맥, 미래 조건화 가능    | 강력한 표현 학습 — 어떻게 학습시킬 것인가가 관건     |
| **Encoder-Decoder** | 인코더의 양방향성 + 디코더의 생성력 | 가장 좋은 pretraining 목적함수가 무엇인지가 관건 |
| **Decoder**         | 언어모델 그 자체            | 생성에 자연스러움, 미래 단어 조건화 불가          |
### 3.1 Encoder Pretraining — BERT
**문제**: 인코더는 양방향 문맥을 쓰므로 일반적인 language modeling(다음 단어 예측)을 그대로 적용할 수 없음(미래를 이미 보고 있으므로 자명한 문제가 됨).
**해법 — Cloze task (빈칸 채우기)**: 입력의 일부를 `[MASK]`로 치환하고, 그 위치의 원래 단어를 예측. Masked LM(MLM)이라 부르며, masking된 단어에 대해서만 loss 계산: $p_\theta(x\mid\tilde{x})$를 학습.
**BERT의 MLM 세부 규칙** [Devlin et al., 2018]:
- (sub)word 토큰의 무작위 15%를 예측 대상으로 선정
- 그 중 80%: `[MASK]`로 치환
- 10%: 무작위 다른 토큰으로 치환
- 10%: 원래 단어 그대로 두되 **여전히 예측 대상**으로 유지
**왜 100% [MASK]로 하지 않는가?**: fine-tuning 시점에는 `[MASK]` 토큰이 전혀 등장하지 않음(train/test mismatch). 100% masking하면 모델이 "masking 안 된 단어는 신경 안 써도 된다"고 안일해질 수 있음 → 나머지 10%/10%를 섞어서, 모델이 masking되지 않은 위치에 대해서도 항상 강건한 표현을 만들도록 강제.
**Next Sentence Prediction (NSP)**: 두 텍스트 chunk가 실제로 인접한 문장인지, 무작위로 뽑힌 무관한 문장인지 이진분류. (후속 연구에서 불필요함이 지적됨 — RoBERTa 등)
**BERT 모델 상세**:

| | BERT-base | BERT-large |
|---|---|---|
| 층 수 | 12 | 24 |
| hidden dim | 768 | 1024 |
| attention head | 12 | 16 |
| 파라미터 | 1.1억 | 3.4억 |
학습 데이터: BooksCorpus(8억 단어) + English Wikipedia(25억 단어). 64 TPU 칩으로 4일 소요(pretraining은 고비용, 단일 GPU로 impractical) — 반면 **finetuning은 단일 GPU로도 실용적** → "한 번 pretrain, 여러 번 finetune."
**BERT Finetuning 방식**:
- **Single sentence classification**: 문장 전체에 레이블 1개, `[CLS]` 토큰의 출력벡터 $h_0$만 사용 (예: SST-2 감성분석, CoLA 문법성 판단)
- **Single sentence tagging**: **토큰마다** 레이블 1개, 모든 토큰 출력벡터 $h_1,\ldots,h_T$를 각각 사용 (예: POS tagging, NER)
- 벤치마크: QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE 등 광범위한 과제에서 SOTA 달성
**Pretrained Encoder의 한계**: 생성(generation) 과제에 부적합. BERT는 각 마스킹 위치를 **독립적으로** 예측하므로 자연스러운 autoregressive 생성이 안 됨 (반면 GPT류 decoder는 왼쪽 문맥만 조건으로 하지만 자연스럽게 순차 생성 가능).
### 3.2 BERT의 확장 모델들
**RoBERTa** [Liu et al. '19]: 구조 변경 없이 **더 오래, 더 많은 데이터로** 학습 + NSP 제거 + **Dynamic Masking**(BERT는 전처리 시 masking을 한 번 고정하지만, RoBERTa는 입력을 모델에 줄 때마다 masking 패턴을 매번 새로 생성). NSP 대체안으로 FULL-SENTENCES(문서 경계를 넘나들며 최대 512토큰 채움), DOC-SENTENCES(문서 경계 유지) 실험. → "구조를 안 바꿔도 compute와 data를 늘리면 pretraining 성능이 개선된다"는 시사점.
**SpanBERT** [Joshi et al. '20]: 개별 토큰이 아니라 **연속된 span 전체**를 마스킹 → 더 어렵고 유용한 pretraining 과제. **Span Boundary Objective (SBO)**: span 경계 토큰 $\mathbf{x}_{s-1}, \mathbf{x}_{e+1}$(예시에서 $\mathbf{x}_4,\mathbf{x}_9$)의 표현만으로 span 내부 각 토큰을 예측: $$ \mathcal{L}(\text{football}) = \mathcal{L}_{\text{MLM}}(\text{football}) + \mathcal{L}_{\text{SBO}}(\text{football}) $$ $$ = -\log P(\text{football}\mid\mathbf{x}_7) - \log P(\text{football}\mid\mathbf{x}_4,\mathbf{x}_9,\mathbf{p}_3) $$
- $\mathbf{p}_3$: span 내 **상대 위치 임베딩** (span 시작으로부터 3번째 토큰이라는 정보) — 경계 토큰 두 개만으로는 "지금 몇 번째 토큰을 예측 중인지" 구분할 수 없으므로 필요
- 효과: 경계 토큰이 span 전체의 의미를 압축하도록 강제 → span 단위 과제(QA, 공지시 해소)에서 강함
**ALBERT** [Lan et al. '20]: "모델을 무작정 키우면 오히려 성능이 나빠질 수 있다"는 관찰에서 출발.
- **Factorized embedding parameterization**: BERT는 $E=H$(word embedding 차원 = hidden 차원)로 묶여있는데, ALBERT는 $H\gg E$로 **분리** — vocab 크기에 비례하는 embedding 파라미터를 줄임
- **Cross-layer parameter sharing**: 모든 층이 파라미터를 공유 → 파라미터 수 대폭 절감
- **Sentence-Order Prediction (SOP)**: NSP 대신, 연속된 두 세그먼트의 **순서를 바꾼 것**을 negative로 사용 (NSP의 negative는 아예 무관한 문장쌍이라 너무 쉬운 과제였던 것을 보완)
### 3.3 Encoder-Decoder Pretraining
**기본 아이디어**: 인코더 입력으로 **prefix**(예측 대상이 아닌 부분)를 주고, 디코더가 나머지를 language modeling처럼 생성. 인코더는 양방향 문맥의 이점을, 디코더는 전체 모델을 학습시키는 signal을 제공.
**T5 (Text-to-Text Transfer Transformer)** [Raffel et al. '19]: 번역, QA, 분류 등 **모든 과제를 "텍스트 입력 → 텍스트 출력"으로 통일**.
세 가지 아키텍처를 비교 실험:
![[Pasted image 20260703214435.png|500]]
1. **Encoder-Decoder** (attention mask: 인코더는 fully-visible, 디코더는 causal)
2. **Decoder-only (Language model)**: 전부 causal
3. **Prefix LM**: 입력(prefix)엔 bidirectional attention, 생성 부분엔 unidirectional attention을 **하나의 Transformer 안**에서 적용
**Prefix LM 상세**:
- Encoder-decoder 대비 파라미터 효율적 (별도의 인코더/디코더 파라미터 없이 하나의 모델 공유)
- 분류 과제를 LM 스타일로 변환: MNLI 예시 → `"mnli premise: I hate pigeons. hypothesis: ... target: entailment"`
- `"target:"` 이전까지가 fully-visible(양방향) prefix — BERT의 `[CLS]`와 유사한 역할(입력 전체를 압축)
- `"target:"` 이후 = causal하게 레이블 텍스트 생성 (BERT처럼 별도 classifier head 없이, decoder 출력층 자체가 classifier 역할)
- task prefix("mnli")로 여러 과제를 하나의 모델이 동시에 학습 가능 (T5의 text-to-text 통일 철학의 기반)
**Unsupervised objective 비교 — Span Corruption이 최선**: $$ \text{입력: "Thank you}\ \underline{\langle X\rangle}\text{ me to your party}\ \underline{\langle Y\rangle}\text{ week."} $$ $$ \text{타깃: "}\langle X\rangle\text{ for inviting }\langle Y\rangle\text{ last }\langle Z\rangle\text{"} $$
- 입력에서 임의 길이의 span들을 **sentinel token**(고유 placeholder, $\langle X\rangle,\langle Y\rangle,\ldots$)으로 치환
- 디코더가 제거된 span들만 sentinel과 함께 순서대로 복원 (여전히 디코더 쪽에서는 language modeling 형태를 유지)
- BERT의 MLM류 여러 변형들과 비교했을 때 **가장 좋은 성능**을 보임
**T5의 결론**: (1) encoder-decoder가 decoder-only보다 우수, (2) span corruption(denoising)이 순수 language modeling보다 우수.
**T5의 놀라운 성질 — 파라미터에 저장된 지식** [Roberts et al. '20]: 외부 문맥/지식 없이, 오직 pretrained 파라미터만으로 open-domain 질문(NQ, WQ, TriviaQA)에 답하도록 finetuning 가능 → 모델 파라미터 자체가 일종의 압축된 지식 저장소로 기능. 모델 크기(220M→11B)가 커질수록 이 능력도 향상.
**BART** [Lewis et al. '19]: BERT(생성 불가)와 GPT(단방향)의 단점을 보완하는 **Denoising Encoder-Decoder**:
![[Pasted image 20260703215017.png]]
- 구조: Bidirectional Encoder + Autoregressive Decoder
- Pretraining: 원본 문서를 **임의의 방식으로 손상(noise)** 시킨 뒤, 원본의 likelihood를 복원하도록 학습
- 노이즈 종류(조합 가능): Token Masking, Token Deletion, **Text Infilling**(임의 길이 span → 하나의 mask, 가장 효과적), Sentence Permutation, Document Rotation
- BERT MLM과 달리 각 위치를 독립적으로 예측하지 않고, **전체를 autoregressive하게 순서대로 복원** → 더 어렵고 풍부한 학습 신호
**BART Fine-tuning**:
- Sequence classification: 동일 입력을 인코더/디코더 양쪽에 넣고, 최종 디코더 출력 표현을 사용
- Machine Translation: BART의 word embedding layer를 **새로 무작위 초기화한 소형 인코더**로 교체(소스 언어 전용, 별도 vocabulary 사용 가능) → 이 소형 인코더는 오직 "소스 언어 토큰을 BART가 이해하는 표현 공간으로 매핑"하는 역할만 하고, 실제 번역(문맥 이해+생성)은 기존 pretrained BART 인코더+디코더가 전담. 2단계 학습: ① embedding들만 학습 → ② 전체 end-to-end finetuning.
- 결과: 강력한 backtranslation baseline 대비, 단일언어 영어 pretraining만으로도 WMT'16 RO-EN 성능 개선
### 3.4 Decoder Pretraining — GPT
**아이디어**: decoder를 language model로 pretraining한 뒤, 그대로 생성기로 사용하거나(대화, 요약처럼 출력이 자연어 시퀀스인 과제에 적합 — pretrained language head layer를 그대로 재사용) 분류 과제용으로 재활용(이 경우 랜덤 초기화된 linear layer $W_{cls}$를 추가로 붙여 전체 네트워크를 통해 backprop).
**GPT** [Radford et al., 2018]:
- Transformer decoder, 12층, 1.17억 파라미터, hidden 768차원, FFN hidden 3072차원
- BPE(40,000 merge), BooksCorpus(7000여 권의 책 — 긴 연속 텍스트로 장거리 의존성 학습에 유리)
- Fine-tuning 입력 포맷 예 (자연어추론): `[START] The man is in the doorway [DELIM] The person is near the door [EXTRACT]` — `[EXTRACT]` 토큰 위치의 표현에 linear classifier 적용
## 4. Entity-Enhanced Pretraining
### 4.1 KnowBert [Peters et al. '19]
**동기**: "Prince sang Purple Rain"에서 Prince(가수/자동차회사/지명 중)와 Purple Rain(앨범/영화/노래 중)의 **entity 중의성**은 텍스트 문맥만으론 한계 → 외부 KB(Wikipedia 등)의 entity 정보를 BERT에 직접 주입.
**KAR(Knowledge Attention and Recontextualization) 모듈 — 7단계**:
![[Pasted image 20260703215125.png]]

|단계|연산|의미|
|---|---|---|
|① Projection|$H_i \to H_i^{\text{proj}}$|BERT 표현을 축소된 차원으로 투영|
|② Span Pooling|$H_i^{\text{proj}} \to \mathbf{S}$|mention span 내 토큰들을 pooling → span 벡터|
|③ Span Self-Attn|$\mathbf{S}\to\mathbf{S}^e$|span들 간 self-attention으로 문맥 반영|
|④ Entity Linking|$\mathbf{S}^e \to \tilde{\mathbf{S}}^e$|KB entity와의 매칭 확률로 가중평균 (아래 상세)|
|⑤ Knowledge Fusion|$\mathbf{S}^e + \tilde{\mathbf{S}}^e \to \mathbf{S}'^e$|span 표현에 KB 지식 벡터를 덧셈으로 융합|
|⑥ Word-to-Span Attn|$\to H_i'^{\text{proj}}$|모든 토큰이 지식 강화된 span들에 다시 attend(recontextualization)|
|⑦ Projection Back|$H_i'^{\text{proj}}\to H_i'$|BERT 원래 차원으로 복원|
KAR은 BERT의 **레이어 사이에 삽입**되는 플러그인 모듈로 작동 ($H_i \to [\text{KAR}] \to H_i' \to$ 다음 BERT 층).
**Step ④ Entity Linking의 상세 수식**: $$ \mathbf{E}=[\mathbf{e}_1,\ldots,\mathbf{e}_M]^T\in\mathbb{R}^{M\times E}\quad(\text{KB의 }M\text{개 entity 임베딩}) $$ $$ \mathbf{\Psi} = MLP(\mathbf{S}^e\mathbf{E}^T,\ \mathbf{P})\in\mathbb{R}^{C\times M}\quad(\text{span-entity 내적 점수 + linking prior}) $$ $$ \mathbf{\Psi}' = \mathbf{\Psi}\odot\mathcal{I}(\mathbf{\Psi}>\delta)\quad(\text{threshold 이하 후보 제거}) $$ $$ \hat{\mathbf{\Psi}}=\text{Softmax}(\mathbf{\Psi}')\quad(\text{남은 후보들에 대한 확률화}) $$ $$ \tilde{\mathbf{S}}^e = \hat{\mathbf{\Psi}}\mathbf{E}\quad(\text{entity 임베딩들의 확률 가중평균}) $$
- $\mathbf{P}$(linking prior): "Prince"라는 mention이 역사적으로 얼마나 자주 각 entity를 가리켰는지의 사전 통계
- **Soft linking을 쓰는 이유**: entity를 하나로 확정(hard)하면 오류 전파·gradient 불안정 위험 → 여러 candidate에 대한 가중평균으로 불확실성 표현 + 모든 candidate에 gradient가 흐르게 하여 end-to-end 학습 가능
**학습 알고리즘 (2단계)**:
1. **KB별 Entity Linker 사전학습**: EL supervision이 있으면, BERT와 나머지는 고정(freeze)하고 pairwise score 계산 관련 파라미터만 수렴할 때까지 학습 — entity linker를 먼저 안정화
2. **전체 통합 학습**: $W_2^{\text{proj}} = (W_1^{\text{proj}})^{-1}$로 초기화(KAR이 처음엔 항등함수처럼 작동하게 하여 기존 BERT 능력 보존) → entity 임베딩만 고정, 나머지 전부 unfreeze → 손실 최소화: $$ \mathcal{L}_{\text{KnowBert}} = \mathcal{L}_{\text{BERT}} + \sum_{i=1}^{j}\mathcal{L}_{\text{EL}_i} $$

**결과**: WiC(단어 의미 중의성), entity typing 등 entity-aware task에서 BERT 대비 향상.
### 4.2 LUKE [Yamada et al. '20]
KnowBert가 "KB 정보를 BERT 중간에 주입"하는 방식이라면, LUKE는 **entity를 처음부터 단어와 동등한 입력으로 함께 처리**.
![[Pasted image 20260703215224.png]]
- **입력 구성**: Words 파트(토큰 임베딩 + position) + **Entities 파트**(entity 임베딩 + position + **entity type embedding**)
- Multi-token entity span(예: "Los Angeles")의 position embedding은 구성 토큰들의 position embedding **평균**을 사용
- **Pretraining**: 단어 마스킹 + entity 마스킹을 **동시에** 예측 (MLM을 단어·entity 두 레벨에서 수행)
- **Entity-aware Self-Attention**: 단어-단어, 단어-entity, entity-단어, entity-entity 네 종류의 attention을 모두 계산 → 완전한 양방향 상호작용
- **Fine-tuning**: 출력된 $h_{w_i}$(단어), $h_{e_i}$(entity) 표현에 linear classifier만 추가 → Entity Typing, Relation Classification, NER, QA 등에 바로 적용
## 5. 초대형 모델과 In-Context Learning
### 5.1 GPT-2
GPT를 더 키운 버전(15억 파라미터), 더 많은 데이터로 학습 → 훨씬 설득력 있는 자연어 생성 샘플을 만들어냄.
### 5.2 GPT-3와 In-Context Learning
기존까지의 두 가지 활용법: (1) 분포에서 샘플링(생성), (2) downstream task로 fine-tuning.
**세 번째 활용법 — In-Context Learning(ICL)**: 매우 큰 언어모델은 gradient 업데이트 없이, **prompt 안에 제공된 예시만으로** 특정 과제를 수행하는 것처럼 보임.
```
입력(프롬프트):
  thanks -> merci
  hello -> bonjour
  mint -> menthe
  otter ->
출력: loutre  (수달의 프랑스어)
```
- GPT-3: 1,750억 파라미터 (당시 최대 T5가 110억이었던 것과 대비되는 규모)
- 모델을 fine-tuning하지 않고도, 문맥(context) 안의 예시 패턴을 "모방"하여 과제를 수행
### 5.3 ICL은 정말 "학습"인가?
- 흥미로운 관찰: 무작위 레이블을 준 in-context 예시로도 어느 정도 잘 수행하는 경우가 있음 → 순수한 "예시로부터의 학습"만으로는 설명이 부족
- 큰 모델(예: Gemini)일수록 **"과제를 추론하기"**(사전학습된 지식에서 과제 유형을 인식)와 **"예시로부터 배우기"** 가 혼합된 복잡한 행동을 보임 → 메커니즘이 아직 완전히 규명되지 않음
### 5.4 왜 스케일을 키우는가 — Scaling Laws
- 경험적 관찰: 모델 크기를 키우면 **perplexity가 안정적으로 개선**됨 (신뢰할 수 있는 규칙성)
- Scaling law는 모델 크기 vs. 데이터 크기의 **trade-off**를 미리 예측하는 데 사용됨 → 주어진 컴퓨팅 예산에서 최적의 (파라미터 수, 데이터 양) 조합을 결정 가능
- 실무적 시사점: 반드시 수렴할 때까지 학습할 필요 없이, **큰 모델을 적당히만 학습**시키는 것이 효율적일 수 있음
- 예시: GPT-3(1,750억, 3,000억 토큰)이 반드시 최적 배분은 아니었음 — 이후 연구에서 훨씬 작은 700억 파라미터 모델이 데이터를 더 많이 학습해서 GPT-3보다 나은 성능을 보인 사례 존재 (Chinchilla 스타일의 scaling 재조명)
### 5.5 토론: 왜 최신 LLM은 대부분 Decoder-only인가
**Encoder-Decoder의 장점**: 입력 시퀀스 인코딩에 양방향성을 부여 가능
**그런데도 Decoder-only가 대세인 이유** — Encoder-Decoder의 ICL/few-shot 관련 한계:

|                              | Decoder-only (GPT류)                                                   | Encoder-Decoder (T5류)                                                              |
| ---------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| ICL 습득 방식                    | Autoregressive pretraining 과정에서 **자연스럽게** 체득 — 데모를 이어붙이고 계속 생성하게 하면 됨 | 이 메커니즘을 염두에 두고 pretrain되지 않아 **세심한 prompt 설계**가 필요                                 |
| Autoregressive demo matching | "예시 암기 + 패턴 연장"이 태생적으로 내장                                             | 프롬프트 형식을 "그대로 따라 하는" 경향이 약함 — instruction tuning이나 few-shot finetuning으로 별도 학습해야 함 |
즉 encoder-decoder 구조 자체의 표현력 문제라기보다, **pretraining objective와 구조가 few-shot/in-context 방식의 사용 패턴과 자연스럽게 맞물리는가**의 문제이며, 이 부분에서 decoder-only 구조가 현재 우위를 점하고 있음.