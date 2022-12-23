import torch

# Use the torch.svd() method to calculate the pseudoinverse of A_p, confirming that your result matches the output of torch.pinverse(A_p):
# Remember the formula is A+ = V.D+.UT

A_p = torch.tensor([[-1, 2], [3, -2], [5, 7.]])

U, d, V = torch.svd(A_p)
# print(U)
# print(d)
# print(V)

D = torch.diag(d)
# print(D)

D_inv = torch.linalg.inv(D)
# print(D_inv)

# Elements to be multiplied:

# print(V)
# print(D_inv.T)
# print(U.T)

A_plus = torch.matmul(V, (torch.matmul(D_inv.T, U.T)))
print(A_plus)

# tensor([[-0.0877,  0.1777,  0.0758],
#         [ 0.0766, -0.1193,  0.0869]])

A_pin_direct_method= torch.pinverse(A_p)
print(A_pin_direct_method)

# tensor([[-0.0877,  0.1777,  0.0758],
#         [ 0.0766, -0.1193,  0.0869]])