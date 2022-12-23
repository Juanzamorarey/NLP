import torch

def f(x, y):
    z = x**2 - y**2
    return z


# print(f(2,2))

x = torch.tensor(3.).requires_grad_()

y = torch.tensor(0.).requires_grad_()

z = f(x,y)

z.backward()

print(z, x.grad, y.grad)
# tensor(9., grad_fn=<SubBackward0>) tensor(6.) tensor(-0.)

x = torch.tensor(2.).requires_grad_()

y = torch.tensor(3.).requires_grad_()

z = f(x,y)

z.backward()

print(z, x.grad, y.grad)
# tensor(-5., grad_fn=<SubBackward0>) tensor(4.) tensor(-6.)

x = torch.tensor(-2.).requires_grad_()

y = torch.tensor(-3.).requires_grad_()

z = f(x,y)

z.backward()

print(z, x.grad, y.grad)
# tensor(-5., grad_fn=<SubBackward0>) tensor(-4.) tensor(6.)