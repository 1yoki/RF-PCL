import torch

def match_cost(setA: torch.Tensor,
               setB: torch.Tensor,
               epsilon: float = 0.1,
               n_iters: int = 50) -> torch.Tensor:
    """
    input：
      - setA, setB: (B, N, D),
    return：
      - cost: Tensor of shape (B,)
    """
    B, N, D = setA.shape
    assert setB.shape[0] == B and setB.shape[1] == N, "setA, setB must have same B,N"

    # 1/N 
    a = setA.new_full((B, N), 1.0 / N)
    b = a.clone()

    C = torch.cdist(setA, setB, p=2)  # (B, N, N)

    # Sinkhorn kernel
    K = torch.exp(-C / epsilon)

    # u, v initialize
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iters):
        u = a / (K @ v.unsqueeze(-1)).squeeze(-1)       # (B,N)
        v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1)

    # P = diag(u) K diag(v)
    P = u.unsqueeze(2) * K * v.unsqueeze(1)            # (B,N,N)

    cost = torch.sum(P * C, dim=(1, 2))                # (B,)
    return cost
