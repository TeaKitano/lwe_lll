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

def inverse(b: np.ndarray, s: np.ndarray,q: int):
    b = (b + q) % q
    for i in range(q):
        pow_i = {j : i ** j % q for j in range(1, q)}
        if len(set(pow_i.values())) == q-1:
            break
    log_i = {j: k for k, j in pow_i.items()}
    pow_i[0] = 1
    def mul(x, y):
        if x == 0 or y == 0:
            return 0
        return pow_i[(log_i[x] + log_i[y]) % (q-1)]
    def div(x, y):
        if x == 0:
            return 0
        return pow_i[(log_i[x] - log_i[y]+q) % (q)]
    
    b_i = np.zeros((len(b), len(b)+1), dtype=int)
    b_i[:, :len(b)] = b
    for i in range(len(b)):
        b_i[i][len(b)] = s[i]
    for i in range(len(b)):
        b_ii = b_i[i][i]
        for j in range(len(b)+1):
            b_i[i][j] = div(b_i[i][j],b_ii)
        for j in range(len(b)):
            if i != j:
                b_i[j] = (b_i[j] - list(map(lambda x: mul(x, b_i[j][i]), b_i[i])) + q) % q
    return b_i[:, len(b)]

if __name__ == "__main__":
    b = np.array([[14,15,5,2],[13,14,14,6],[6,10,13,1],[10,4,12,16]])
    print(size_reduction(b))
    print(inverse(b, np.array([8,16,12,12]), 17))