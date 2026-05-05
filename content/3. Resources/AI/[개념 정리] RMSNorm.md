---
title: "[개념 정리] RMSNorm"
description: 정규화 기법 중 RMSNorm에 대해 다룹니다.
date: 2026-05-05
tags:
  - AI
  - Concept
---
## RMSNorm
**RMSNorm**은 널리 사용되는 **LayerNorm**을 단순화한 것이다. 이 방법은 계산 비용을 줄이면서도 성능은 그대로 유지하거나 오히려 향상시킬 수 있다.
LayerNorm은 활성화 값의 평균을 빼고 표준편차로 스케일링함으로써 데이터를 중심값으로 맞춘다.
반면, RMSNorm을 평균을 빼는 단계를 생략한다. 이러한 단순화 덕분에 계산 비용을 상당 부분 줄일 수 있다.
RMSNorm은 입력 데이터의 제곱평균 값을 기준으로 스케일링하는 방식으로 작동한다.
![[Pasted image 20260505202622.png]]
여기서 $\epsilon$은 수치적 안정성을 위한 작은 상수이다.
LayerNorm과 마찬가지로, RMSNorm도 학습 가능한 스케일링 파라미터 $\gamma_i$를 포함한다. (때로는 바이어스 $b$도 포함되기도 한다.)
평균값을 계산하지 않기 때문에, RMSNorm은 연산량과 메모리 사용량을 줄일 수 있다.
GPU에서의 성능을 보면, 설정에 따라 LayerNorm보다 7-64% 더 빠른 것으로 나타났다.
## Implementation (PyTorch)
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x):
        # Calculate root mean square along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_normalized = x / rms
        
        # Apply scaling if using learnable parameters
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight
        
        return x_normalized
```
## References
- https://www.stephendiehl.com/posts/post_transformers/?utm_source=tldrai#rmsnorm