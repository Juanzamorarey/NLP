import torch
import math

# Use the PyTorch trace method to calculate the trace of A_p.
# Use the PyTorch Frobenius norm method and the trace method to demonstrate that  ||A||F=Tr(AAT)−−−−−−−−√


A_p = torch.tensor([[-1, 2], [3, -2], [5, 7.]])

A_p_traceOperator = torch.trace(A_p)
# print(A_p)
# print(A_p_traceOperator)

# tensor([[-1.,  2.],
#         [ 3., -2.],
#         [ 5.,  7.]])
# tensor(-3.)

A_p_frobeniusNorm = torch.linalg.matrix_norm(A_p)
# By default is the frobenius norm. torch.norm() is deprecated should use torch.linalg.matrix_norm() or torch.linalg.vector_norm() instead
# print(A_p_froebiusNorm)
# tensor(9.5917)


A_p_UsingTr = math.sqrt(torch.trace(torch.linalg.matmul(A_p,A_p.T)))
# print(A_p_UsingTr)
# 9.591663046625438
