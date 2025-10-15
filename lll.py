import numpy as np
from utils import gso, size_reduction

def lll(b: np.ndarray, delta: float = 0.75):
    """
    LLL簡約アルゴリズム
    input:
    b: 基底ベクトルの行列（各行が基底ベクトル）
    delta: Lovászの条件のパラメータ（通常0.75）
    output:
    LLL簡約された基底
    """
    n = len(b)
    b = b.copy()
    k = 1
    
    while k < n:
        b = size_reduction(b)
        b_star, mu = gso(b)
        
        # Lovászの条件をチェック
        # ||b*_k||² ≥ (δ - μ²_{k,k-1}) ||b*_{k-1}||²
        if np.dot(b_star[k], b_star[k]) >= (delta - mu[k][k-1]**2) * np.dot(b_star[k-1], b_star[k-1]):
            k += 1
        else:
            # ベクトルを交換
            b[k], b[k-1] = b[k-1].copy(), b[k].copy()
            k = max(k-1, 1)
    
    return b

if __name__ == "__main__":
    b = np.array([[105, 821, 404, 328, 214], [881, 667, 644, 927, -412], [181, 483, 87, 500, 73], [893, 834, 732, 441, 551]])
    result = lll(b,0.99)
    print("元の基底:")
    print(b)
    print("\nLLL簡約後の基底:")
    print(result)
    
    # 基底のノルムを確認
    print("\n各ベクトルのノルム:")
    for i, vec in enumerate(result):
        norm = np.linalg.norm(vec)
        print(f"b_{i}: {norm:.2f}")