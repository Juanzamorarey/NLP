import torch
import numpy

# Left-singular vectors of  A  = eigenvectors of  AAT .
# Right-singular vectors of  A  = eigenvectors of  ATA .
# Non-zero singular values of  A  = square roots of eigenvalues of  AAT  = square roots of eigenvalues of  ATA 
# Exercise: Using the matrix P from the preceding PyTorch exercises, demonstrate that these three SVD-eigendecomposition equations are true.

P = torch.tensor([[25, 2, -5], [3, -2, 1], [5, 7, 4.]])
# print(P)

U, d, V = torch.linalg.svd(P, full_matrices=False)
# El metodo .linalg.svd() arroja resultados diferentes

V_tras = V.T
# Careful with the deprecated .svd() method, in linalg.svd() we don't need to make the trasposed

d_fixed = torch.diag(d)
# d must_hace the same dimensions

# print(U)
# print("\n")
# print(d)
# print("\n")
# print(V)
# print("\n")
# print(d_arreglada)

# tensor([[-0.9757,  0.1823,  0.1214],
#         [-0.0975,  0.1350, -0.9860],
#         [-0.1961, -0.9739, -0.1140]])


# tensor([26.1632,  8.1875,  2.5395])


# tensor([[-0.9810,  0.0113, -0.1937],
#         [-0.1196, -0.8211,  0.5581],
#         [ 0.1528, -0.5706, -0.8069]])


# tensor([[26.1632,  0.0000,  0.0000],
#         [ 0.0000,  8.1875,  0.0000],
#         [ 0.0000,  0.0000,  2.5395]])


# 1. Check that eigenvectros of P.PT is the same as U

p_pT = torch.linalg.matmul(P,P.T)
lambdas, eign_p_pT = torch.linalg.eig(torch.linalg.matmul(P,P.T))
# print("Columns of: \n")
# # print(eign_p_pT)
# print("same as columns of: \n")
# # print(U)

# 2. Check that eigenvectros of PT.P is the same as V_tras

pT_p = torch.linalg.matmul(P.T,P)
lambdas, eign_pT_p = torch.linalg.eig(torch.linalg.matmul(P.T,P))
# print("Columns of: \n")
# print(eign_pT_p)
# print("same as columns of: \n")
# print(V_tras)

# 3. Non-zero singular values of  A  = square roots of eigenvalues of  AAT  = square roots of eigenvalues of  ATA 
# (From Nikolay in a previous comment)

P_repeated = torch.linalg.matmul(U, (torch.linalg.matmul(d_fixed,V_tras)))

# print(f"{P}\n is equal to:\n {P_repeated}")

# tensor([[25.,  2., -5.],
#         [ 3., -2.,  1.],
#         [ 5.,  7.,  4.]])
#  is equal to:
#  tensor([[24.9116, -1.6888,  5.5284],
#         [ 1.9878,  0.4925,  3.1314],
#         [ 5.9433,  6.6552, -3.2228]])
