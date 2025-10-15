import numpy as np

def gso(b: np.ndarray):
    """
    グラム・シュミット直交化を実行し、b*とμを返す
    b*は直交ベクトル（正規化しない）
    μはグラム・シュミット係数
    """
    n = len(b)
    m = len(b[0])
    b_star = np.zeros((n, m), dtype=float)
    mu = np.zeros((n, n), dtype=float)
    
    b_star[0] = b[0].astype(float)
    
    for i in range(1, n):
        b_star[i] = b[i].astype(float)
        for j in range(i):
            mu[i][j] = np.dot(b_star[i], b_star[j]) / np.dot(b_star[j], b_star[j])
            b_star[i] -= mu[i][j] * b_star[j]
    
    return b_star, mu

def size_reduction(b: np.ndarray):
    """
    サイズ簡約を実行
    """
    n = len(b)
    b_copy = b.copy()
    
    for i in range(1, n):
        for j in range(i-1, -1, -1):
            b_star, mu = gso(b_copy)
            if abs(mu[i][j]) > 0.5:
                b_copy[i] = b_copy[i] - round(mu[i][j]) * b_copy[j]
    
    return b_copy

if __name__ == "__main__":
    b = np.array([[105, 821, 404, 328], [881, 667, 644, 927], [181, 483, 87, 500], [893, 834, 732, 441]])
    print(size_reduction(b))