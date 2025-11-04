import numpy as np
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import hermite_normal_form

from lll import lll
from utils import inverse

def lwe(a: np.array, b: np.array, q: int):
    # BDD(CVP)に帰着
    n = len(a[0])
    m = len(a)
    aq = np.zeros((m, m+n), dtype=int)
    aq[:, :n] = a
    aq[:, n:] = np.diag([q]*m)
    aq = Matrix(aq,)
    aq_l = hermite_normal_form(aq)
    B = np.zeros((m+1, m+1))
    # kannan埋め込みでSVPに帰着
    B[:m, :m] = aq_l.T.tolist()
    B[-1, :-1] = b
    B[:-1, -1] = 0
    B[-1, -1] = 1
    # SVPから最短ベクトルを取得
    e = np.array(lll(B, 0.99)[0][:-1])
    # エラー項から秘密鍵取得、未完成
    b_d = b[:n]
    e_d = e[:n]
    a_d = a[:n, :]
    s = inverse(a_d, b_d - e_d, q)

    return e, s


if __name__ == "__main__":
    a = np.array([[1, 5, 21, 3, 14], [17, 0, 12, 12, 13],[12, 21, 15, 6, 6],
    [4, 13, 24, 7, 16], [20, 9, 22, 27, 8], [19, 8, 19, 3, 1], [18, 22, 4, 8, 18],
    [6, 28, 9, 5, 18], [10, 11, 19, 18, 21], [28, 18, 24, 27, 20]], dtype=int)
    b = np.array([28, 2, 24, 16, 11, 14, 7, 28, 27, 13], dtype=int)
    print(lwe(a, b, 29))
    # a = np.array([[14, 15, 5, 2], [13, 14, 14, 6], [6, 10, 13, 1],
    # [10, 4, 12 ,16], [9, 5, 9 , 6], [3, 6, 4, 5], [6, 7, 16, 2]], dtype=int)
    # b = np.array([8, 16, 12, 12, 9, 16, 3], dtype=int)
    # lwe(a, b, 17)
    # a = np.array([[3, 2, 4], [1, 5, -2], [7, 1, 3], [5, 10, -7], [6, 1, 2]], dtype=int)
    # b = np.array([1, 4, 5, 9, 0], dtype=int)
    # lwe(a, b, 17)