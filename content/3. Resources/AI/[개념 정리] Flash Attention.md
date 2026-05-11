---
title: "[개념 정리] Flash Attention"
description: Attention 연산을 가속화하는 Flash Attention 기법에 대해 다룹니다.
date: 2026-05-03
tags:
  - AI
  - Concept
---
## FlashAttention 1
FlashAttention 1이 처음 제안된 논문명은 [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)이다.
이 논문은 제목에서 알 수 있다시피 IO, read/write 최적화 논문이며 그 중에서도 GPU의 메모리 구조를 파악하여 최적화한 논문이다.
GPU의 메모리는 작지만 빠른 on-chip과 크지만 느린 off-chip으로 구성되어 있다.
저자는 이러한 구조를 이해하지 않고 GPU 프로그래밍을 하면 최대 성능을 이끌어낼 수 없다고 말하며, 이러한 구조에 최적화된 알고리즘을 제안한다.
### 문제
- 트랜스포머 기반 LLM 모델의 경우 문자열의 길이가 매우 제한적이다.
- 문자열의 길이가 제한적이라 더 긴 문장이나 이미지 등을 학습할 수 없다.
### 원인
- 어텐션 행렬의 시간, 공간 복잡도가 $N^2$($N$ = 토큰 개수)이라서, 문자열의 길이를 늘리기 어렵다. → Long latency, OOM(Out Of Memory)
	![[Pasted image 20260503124324.png]]
### 해결
- Tiling (소프트맥스 병렬화를 통한 속도 향상)
	- SRAM의 사이즈에 맞게 어텐션 행렬을 자른 후 여러 개의 스레드 블록으로 병렬 수행
	- 각 블록 병렬 수행 후 리스케일링을 통해 정확한 소프트맥스 값 도출
	- 이러한 방식을 통해 HBM 접근 횟수를 최소화하고, 여러 개의 GPU 코어를 최대한 활용
	![[Pasted image 20260503124533.png|300]]
- 아래 그래프는 블록 사이즈를 늘리며 HBM 접근 횟수가 얼마나 줄어드는가를 실험한 그래프
	- 블록 사이즈를 늘리면 SM 활용도가 올라가고, SRAM 활용도도 같이 올라간다.
	- 하지만 SM의 개수가 제한적이고, 결국 리스케일링 커뮤니케이션(각 블록마다 따로 계산한 소프트맥스를 다시 한 번 맞춰주기 위해 서로 정보를 주고받는 과정)이 필요하므로 특정 사이즈 이상부터는 성능이 좋아지지는 않는다.
	![[Pasted image 20260503151717.png|400]]
	
- Recomputation (어텐션 행렬을 저장하지 않고, backward 때 다시 계산)
	- 전체 어텐션 행렬을 HBM에 저장하지 않고 소프트맥스를 원래 정의대로 안전하게 계산하기 위해 쓰는 정규화 인자만 저장해 둔다.
	- A100 같은 GPU에서는 $QK^\top$를 다시 한 번 더 계산하는 FLOPs 비용보다, HBM에서 $N \times N$ 크기의 행렬을 한 번 읽어오는 IO 비용이 더 비싸다.
- Kernel fusion
	- 첫 번째로 지적하는 것은, performance breakdown
	- 두 번째는 위의 알고리즘이 커널 퓨전을 가능하게 하여 최적화가 가능하다.
	![[Pasted image 20260503152422.png|300]]
### 알고리즘 (pseudo-code)
![[Pasted image 20260503221748.png]]
1. **(1~3줄) 입력 및 초기화**: Q, K, V 행렬을 입력받고, SRAM 크기에 맞춰 블록 크기($B_c$, $B_r$)를 정한 뒤, 출력 O는 0, 정규화 항 ℓ은 0, 최댓값 m은 -∞로 초기화
2. **(4~6줄) 블록 분할**: Q는 행 방향으로 $T_r$개, K·V는 열 방향으로 $T_c$개 블록으로 나누고, O·ℓ·m도 Q와 동일하게 $T_r$개 블록으로 분할
3. **(5~6줄) 외부 루프 (j = 1 to $T_c$)**: K, V 블록을 순회하면서 $K_j, V_j$를 HBM → SRAM으로 로드 (K, V를 바깥에 두어 재로드 최소화)
4. **(9~10줄) 내부 루프 시작 (i = 1 to $T_r$)**: 각 Q 블록 $Q_i$와 현재까지의 $O_i, ℓ_i, m_i$를 SRAM으로 로드
5. **11줄 - 어텐션 점수 계산**: SRAM 내에서 $S_{ij} = Q_i K_j^T$ 계산
6. **12줄 - 블록 단위 소프트맥스 통계량 계산**: 행별 최댓값 $\tilde{m}_{ij}$ → 안정화된 지수 $\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij})$ → 행별 합 $\tilde{ℓ}_{ij}$ 계산
7. **13줄 - 누적 통계량 갱신 (online softmax 핵심)**: 새로운 최댓값 $m_i^{new}$와 정규화 항 $ℓ_i^{new}$를 이전 값과 결합하여 갱신
8. **14줄 - 출력 갱신**: 이전 $O_i$를 재스케일링한 뒤 현재 블록 기여분 $\tilde{P}_{ij}V_j$를 더해 HBM에 다시 기록
9. **(15~17줄) 통계량 저장 및 루프 종료**: 갱신된 $ℓ_i, m_i$를 HBM에 저장하고 내·외부 루프 종료
10. **18줄 - 결과 반환**: 최종 출력 O 반환
## FlashAttention 2
### 문제
- FlashAttention이 아직 최적화가 덜 되었다.
	- 아직 GPU를 많이 못 쓰고 있다. (Forward pass 이론 최대 FLOPs/s의 30-50%, backward pass는 25-35%)
### 원인
- 알고리즘적, 하드웨어(GPU)적 최적화가 덜 진행됐다.
- 매우 큰 A100의 matmul/non-matmul 성능 차이
	- 312 TFLOPS/s of FP16/BE16 matmul
	- 19.5 TFLOPS/s of non-matmul FP32
### 해결
#### 1. Tweak algorithm
![[Pasted image 20260503222457.png]]
첫 번째 tweak은 non-matmul 연산을 최소화하는 것이다. 이를 위해 FlashAttention 2에서는 forward pass가 아래와 같이 바뀌었다.
![[Pasted image 20260503222302.png]]
매 반복마다 두 항 모두를 $\text{diag}(\ell^{(2)})^{-1}$로 나누는 정규화 연산을 수행하지 않고, 루프 내부에서는 “un-scaled” 버전 $\tilde{O}$를 유지하고, 마지막에 단 한번만 $\text{diag}(\ell^{(last)})^{-1}$로 나눠서 최종 출력을 얻는다.
![[Pasted image 20260503223255.png]]
두 번째 tweak는 각 블록마다 최댓값 $m^{(j)}$와 $\ell^{(j)}$를 둘 다 저장하지 않고, 두 값을 합친 logsumexp 한 가지만 저장하는 것이다.
backward pass에서 softmax를 재계산할 때 $L^{(j)}$ 하나만 있으면 충분하기 때문에(수학적으로 동치) 메모리 사용량과 HBM 입출력 횟수를 감소시킬 수 있다.
#### 2. Parallelism
![[Pasted image 20260503223654.png|500]]
#### 3. Work Partitioning Between Warps
- 간단하지만 효과적인 최적화 방법
- 기존에는 $Q$를 모든 warp가 공유하고, 각 warp가 서로 다른 $K$, $V$ 블록을 처리하기 때문에 중간 결과를 합쳐야 함
- 하지만 오른쪽은 각 warp가 서로 다른 $Q$를 담당하여 같은 $K$, $V$를 공유해서 계산하기 때문에 독립적으로 계산 가능
![[Pasted image 20260503223840.png]]
## References
- https://arxiv.org/pdf/2205.14135
- https://jinwoongkim.net/papers/FlashAttention-key-ideas/
## Backlinks
- [[[논문 리뷰] Attention Is All You Need]]