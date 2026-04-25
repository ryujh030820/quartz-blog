---
title: "[개념 정리] Kosaraju's Algorithm - 코사라주 알고리즘"
description: 코사라주 알고리즘에 대해 다룹니다.
date: 2026-04-23
tags:
  - Algorithm
  - Concept
---
## SCC(Strongly Connected Component)
SCC란 방향 그래프에서 아래의 두 가지 조건을 만족하는 서브 그래프이다.
1. 한 scc안에 속한 임의의 어떤 한 노드 A와 다른 한 임의의 노드 B에 대해서 A에서 B로 갈 수 있는 경로가 존재한다.
2. 어떠한 scc에 속하지 않은 어떠한 노드도 scc에 추가로 들어왔을 때 1의 성질을 만족하면 안된다.
즉 내부에서 자유롭게 이동이 가능한 최대 크기의 노드 집합을 뜻한다. 이를 예시로 들어보면
![[Pasted image 20260423131733.png|300]]![[Pasted image 20260423131740.png|300]]
왼쪽과 같은 그래프에서 SCC별로 묶으면 오른쪽의 색칠과 같이 3개의 SCC로 만들어진다.
## 코사라주 알고리즘
코사라주 알고리즘은 다음과 같은 방식으로 진행된다.
1. 아직 방문하지 않은 점 부터 dfs를 실시한다. (이를 그래프의 모든 점을 방문할 때 까지 반복한다.)  
2. dfs하는 과정에서 방문이 완료된 순서대로 스택에 담는다.  
3. 그래프의 모든 간선을 뒤집은 새로운 그래프를 만든다.  
4. 새로운 그래프에 대해서 스택의 제일 위에서 꺼내는 순서대로 아직 방문하지 않았다면 dfs를 시작한다.
5. 4번 과정에서 어떤 한 시작점에 대해서 한 번에 방문한 모든 노드를 SCC로 묶으면 된다.
![[0093-scc-kosaraju.gif]]
## References
- https://stonejjun.tistory.com/84
- https://redspider110.github.io/2018/08/22/0093-algorithms-scc-kosaraju/