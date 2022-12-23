import numpy as np


def calcula_probabilidad_binaria(n,k):
    n_minus_k = n - k
    denominador = 2**n
    for i in range(n):
        if i == 0:
            pass
        else:
            n = n*i
    if k != 0:
        for i in range(k):
            if i == 0:
                pass
            else:
                k = k*i
    else:
        k = 1
    if n_minus_k > 0:
        for i in range(n_minus_k):
            if i == 0:
                pass
            else:
                n_minus_k = n_minus_k*i
    else:
        n_minus_k = 1

    # if k == 0:
    #     k = 1
    # elif n_minus_k == 0:
    #     n_minus_k == 1
    # else:
    #     numerador = n/(k*n_minus_k)

    numerador = n/(k*n_minus_k)

    return (numerador/denominador)
    
    

caras_0 = calcula_probabilidad_binaria(5,0)
caras_1 = calcula_probabilidad_binaria(5,1)
caras_2 = calcula_probabilidad_binaria(5,2)
caras_3 = calcula_probabilidad_binaria(5,3)
caras_4 = calcula_probabilidad_binaria(5,4)
caras_5 = calcula_probabilidad_binaria(5,5)

print(f"Esto es 0 caras  {caras_1}")
print(f"Esto es 1 cara  {caras_1}")
print(f"Esto es 2 caras  {caras_2}")
print(f"Esto es 3 caras  {caras_3}")
print(f"Esto es 4 caras  {caras_4}")
print(f"Esto es 5 caras  {caras_5}")

# aras  0.15625
# Esto es 1 cara  0.15625
# Esto es 2 caras  0.3125
# Esto es 3 caras  0.3125
# Esto es 4 caras  0.15625
# Esto es 5 caras  0.03125








    