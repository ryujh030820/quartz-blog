---
title: "[개념 정리] Beam Search"
date: 2026-07-16
tags:
  - AI
  - Concept
---
## Greedy Decoding
보통 자연어 생성 같은 task에서는 자연어를 생성할 때 매 타임스텝마다 decoder를 거쳐서 나온 단어가 다음 타임스텝의 input으로 들어가게 된다.
그리고 decoder에서는 매 타임스텝마다 가장 확률이 높은 단어를 선택해서 출력하게 되는데 이를 **greedy decoding**이라 한다.
하지만 greedy decoding의 문제점은 매 타임스텝의 최대 확률 값만을 고려하는 것이 전체 타임스텝으로 보면 적절하지 않은 문장을 출력으로 가지게 될 수 있다는 점이다.
![[Pasted image 20260716143728.png|500]]
이를 해결하기 위해 현재 타임스텝만을 고려하는 방식이 아닌 전체 타임스텝에 대하여 확률을 고려하는 방법이 있지만, 그러한 방법은 계산해야 하는 경우의 수가 기하급수적으로 늘어나게 된다.
단어 집합의 수를 $V$라고 하고 타임스텝의 수를 $t$라고 하면 시간 복잡도는 경우의 수는 $V^t$이 된다.
![[Pasted image 20260716143917.png]]
## Beam Search
이를 해결하기 위한 beam search의 기본 아이디어는 매 타임스텝마다 모든 가능한 경우의 수를 고려하는 것이 아닌 $k$개의 가능성이 있는 경우의 수만을 추적해서 가능한 출력 값으로 보관하는 것이다.
![[Pasted image 20260716144058.png]]
아래는 greedy decoding과 beam search의 차이점에 대한 $k=2$일 때의 예시이다.
![[Pasted image 20260716145212.png|500]]
$k=2$이므로 매 타임스텝마다 가장 가능성이 있는 2개의 출력값을 추적한다.
greedy decoding에서는 생성 과정 중에 EOS 토큰이 출력 값으로 나오게 되면 생성을 멈추는데, beam search는 아래와 같은 조건이 충족될 때까지 계속되게 된다.
여기서 $T$와 $n$은 사용자 정의에 따라 달라질 수 있다.
![[Pasted image 20260716145458.png]]
즉 EOS 토큰이 아직 안 나왔더라도 $T$라는 최대 타임스텝에 도달했거나(미완성 상태), 미리 정한 $n$개의 완성된 hypothesis(EOS 토큰을 생성한)가 모이면 탐색이 종료된다.
그리고 최종적으로 가능성이 있는 $k$개의 출력값들 중에서 가장 score가 높은 값을 출력값으로 가지게 된다.
![[Pasted image 20260716145936.png]]
여기서 길이가 더 긴 문장들일수록 더 낮은 score를 가지게 되기 때문에 문장을 타임스텝으로 normalize해주게 된다.
## References
- https://bitrader.tistory.com/79
- https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf