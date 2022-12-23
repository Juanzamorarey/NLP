import torch
import matplotlib.pyplot as plt
# 1.Usar Pytorch para encontrar la pendiente de y = x**2 + 2x + 2 cuando x = 2 con automatic differentation

# my_x = torch.tensor(2.0)
# my_x.requires_grad_()

# y = my_x**2 + 2*my_x + 2

# y.backward()
# print(my_x.grad)

# Solution
# tensor(6.)

# 2.Inventa una serie de valores para "y" y juega con ellos usando valores para m y b simulando una relación
# lineal entre x e y y luego ajusta los valores m y b

def my_function(x):
    y = x**2 + 2*x + 2
    return y


x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
# Random values for x


y = x**2 + 2*x +2 + torch.normal(mean=torch.zeros(8),std=0.2)
# print(y)
# y = tensor([ 2.2049,  4.6593,  9.8701, 16.2913, 25.8943, 37.0377, 49.9566, 65.0630])

# fig, ax = plt.subplots()
# plt.title("Price - Number of bedrooms relation")
# plt.xlabel("Price")
# plt.ylabel("Number of bedrooms relation")
# _ = ax.scatter(x, y)
# It appears lind of a ascedent dot line

# Now we create random values for m and b let's say m = 6 Cause we know is the slope and b = -4
m = torch.tensor([6.]).requires_grad_()
b = torch.tensor([-4.]).requires_grad_()

# We create a function to get the value of y with our parameters. Remember the equation of a line is y = mx + b

def regression(x_reg, m_reg, b_reg):
    return (m_reg*x_reg + b_reg)

# We use the professor function to show our plot
def regression_plot(my_x, my_y, my_m, my_b):
    
    fig, ax = plt.subplots()

    ax.scatter(my_x, my_y)
    
    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, my_m, my_b).detach().item()
    y_max = regression(x_max, my_m, my_b).detach().item()
    
    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max])
# It shows a linea underneath our points need to be improved

pseudo_y = regression(x, m, b)
# print(pseudo_y)
# tensor([-4.,  2.,  8., 14., 20., 26., 32., 38.], grad_fn=<AddBackward0>)

# We calculate the loss function or cost function
mse = torch.nn.MSELoss()
Cost_value = mse(pseudo_y, y)
# print(Cost_value)
# tensor(158.8030, grad_fn=<MseLossBackward0>)

Cost_value.backward()
# print(m.grad)
# print(b.grad)
# tensor([-97.8611])
# tensor([-18.9750])

optimizer = torch.optim.SGD([m, b], lr=0.01)

optimizer.step()

# That's the first step for our model and the line is better but not as much as it can be so we repeat the process 1000 times

epochs = 1000
# 1 epoch is one iteration
for epoch in range(epochs):
    
    optimizer.zero_grad() # Reset gradients to zero; else they accumulate
    
    pseudo_y = regression(x, m, b) # Step 1
    C = mse(pseudo_y, y) # Step 2
    
    C.backward() # Step 3
    optimizer.step() # Step 4
    
    print('Epoch {}, cost {}, m grad {}, b grad {}'.format(epoch, '%.3g' % C.item(), '%.3g' % m.grad.item(), '%.3g' % b.grad.item()))



# Epoch 0, cost 77.2, m grad -62.7, b grad -11.8
# Epoch 1, cost 43.9, m grad -40, b grad -7.21  
# Epoch 2, cost 30.4, m grad -25.5, b grad -4.27
# Epoch 3, cost 24.9, m grad -16.3, b grad -2.4 
#                       .
#                       .
#                       .
# Epoch 10, cost 21.2, m grad -0.845, b grad 0.697
# Epoch 11, cost 21.2, m grad -0.598, b grad 0.742
#                       .
#                       .
#                       .
# Epoch 18, cost 21.1, m grad -0.18, b grad 0.792
# Epoch 19, cost 21.1, m grad -0.172, b grad 0.789
# Epoch 20, cost 21.1, m grad -0.167, b grad 0.785
# Epoch 21, cost 21.1, m grad -0.164, b grad 0.781
#                       .
#                       .
#                       .
# Epoch 998, cost 20.5, m grad -0.000574, b grad 0.00275
# Epoch 999, cost 20.5, m grad -0.000562, b grad 0.00274

print(m.item())
print(b.item())

# As we can see the minimum cost after a 1000 iterations is 20.5 and corresponds to the value of pseudo_y when m and b have this values m = 8.975499153137207 b = -4.943902015686035

regression_plot(x, y, m, b)

# 3. Lee acerca de la programación diferencial 

