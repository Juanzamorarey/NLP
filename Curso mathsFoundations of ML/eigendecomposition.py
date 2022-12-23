import torch
import numpy as np
# Usa pyTorch para realizar el eigendecomposition de la matriz P que sigue la formula: A=VΛV−1


P = torch.tensor([[25, 2, -5], [3, -2, 1], [5, 7, 4.]])

# 1 obtenemos los eigenvectors (V) y los eigenvalues (lambda)

lambdas, V = torch.linalg.eig(P)
# print(lambdas)
# print(V)

# En pytorch hay que pasar todo a float

lambdas_Use = lambdas.float()
V_Use = V.float()
# print(lambdas_Use)
# 2 Obtenemos la inversa de V

V_inv = torch.linalg.inv(V_Use)
# print(V_inv)

# 3 Obtenemos Λ 

# ATENCION: diagonal() y diag() no son lo mismo en pytorch. El segundo no está en el paquete .linalg

Lambda_diag = torch.diag(lambdas_Use)
# print(Lambda_diag)

#  4 confirmamos la forumla

P_result = torch.linalg.matmul(V_Use, torch.linalg.matmul(Lambda_diag, V_inv))
# print(P_result)

# print(f"{P}\n es igual a: \n{P_result}")



# Ejercicio 2 Use PyTorch to decompose the symmetric matrix  S  (below) into its components  Q ,  Λ , and  QT . Confirm that  S=QΛQT .

S = torch.tensor([[25, 2, -5], [2, -2, 1], [-5, 1, 4.]])

# Para ahcerlo seguimos los mismos pasos pero usando la matriz traspuesta en vez de la inversa

lambdas_Q, Q = torch.linalg.eig(S)

lambdas_Use_Q = lambdas_Q.float()
Q_Use = Q.float()

Q_trans = Q.T
Q_trans_good = Q_trans.float()

Lambda_diag_Q = torch.diag(lambdas_Use_Q)

S_result = torch.linalg.matmul(Q_Use, torch.linalg.matmul(Lambda_diag_Q, Q_trans_good))

print(f"{S}\n es igual a: \n{S_result}")