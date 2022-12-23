import torch
import numpy as np
import matplotlib

# A = torch.tensor([[-1.,4.],[2.,-2.]])
# # Recordemos que los tensores de torch simepre necesitasn un flaot de ahi el .

# lambdas, EigenVectors = torch.linalg.eig(A)

# # print(EigenVectors) #Aquí tenemos los eigenvectors
# # tensor([[ 0.8601+0.j, -0.7645+0.j],
# #         [ 0.5101+0.j,  0.6446+0.j]])

# # print(lambdas) #Aquí tenemos los eigenvalues de los eigenvectors

# # Vamos a convertirlos a floats para evitar los números imaginarios que aparecen a veces como .j

# Eigen_vectors_correcto = EigenVectors.float()
# lambdas_correcto = lambdas.float()

# # print(Eigen_vectors_correcto)
# # tensor([[ 0.8601, -0.7645],
# #         [ 0.5101,  0.6446]])

# primer_vector = Eigen_vectors_correcto[:,0] #Cogemos la priemra columna, es decir primer vector.
# primer_eigenvalue = lambdas_correcto[0] #primer eigenvalue
# # print(primer_vector)
# # tensor([0.8601, 0.5101])
# # print(primer_eigenvalue)
# # tensor(1.3723)

# # Vamos a multiplicar el primer vector por la matriz inicial A y el mismo primer vector por nuestro eigenvalue obteniendo el mismo resultado
# resultado_matriz_vector = torch.matmul(A,primer_vector)
# resultad_matriz_eigenvalue = primer_eigenvalue * primer_vector
# # print(resultado_matriz_vector)
# # print(resultad_matriz_eigenvalue)
# # tensor([1.1803, 0.7000])
# # tensor([1.1803, 0.7000])

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

# Ejercicio propio:
X = torch.tensor([[25, 2, 9.], [5, 26, -5.], [3, 7, -1.]])
X_np = np.array([[25, 2, 9], [5, 26, -5], [3, 7, -1]])
# print("Esto es tensor de pytorch:\n")
# print(X)
# print("Esto es tensor de numpy:\n")
# print(X_np)

lambdas_X, EigenVectors_X = torch.linalg.eig(X)
Eigen_vectors_X_correcto = EigenVectors_X.float()
lambdas_X_correcto = lambdas_X.float()
# print("Esto son resultados de pytorch:\n")
# print(Eigen_vectors_X_correcto)
# print(lambdas_X_correcto)
# print("Esto son resultados de numpy:\n")
lambdas_Xnp, V_X = np.linalg.eig(X_np) 
# print(lambdas_Xnp)
# print(V_X)

primer_vector_X = Eigen_vectors_X_correcto[:,0]
primer_eigenvalue_X = lambdas_X_correcto[0]

segundo_vector_X = Eigen_vectors_X_correcto[:,1]
segundo_eigenvalue_X = lambdas_X_correcto[1]

tercer_vector_X = Eigen_vectors_X_correcto[:,2]
tercer_eigenvalue_X = lambdas_X_correcto[2]

resultado_matriz_vector_uno = torch.matmul(X,primer_vector_X)
resultad_matriz_eigenvalue_uno = primer_eigenvalue_X * primer_vector_X

resultado_matriz_vector_dos = torch.matmul(X,primer_vector_X)
resultad_matriz_eigenvalue_dos = primer_eigenvalue_X * primer_vector_X

resultado_matriz_vector_tres = torch.matmul(X,primer_vector_X)
resultad_matriz_eigenvalue_tres = primer_eigenvalue_X * primer_vector_X

print("Los primeros resultados son iguales\n")
print(f"{resultado_matriz_vector_uno}\n")
print(f"{resultad_matriz_eigenvalue_uno}\n")

print("Los segundos resultados son iguales\n")
print(f"{resultado_matriz_vector_dos}\n")
print(f"{resultad_matriz_eigenvalue_dos}\n")

print("Los terceros resultados son iguales\n")
print(f"{resultado_matriz_vector_tres}\n")
print(f"{resultad_matriz_eigenvalue_tres}\n")


# Nota importante:
# El calcula de .eig() puede variar entre librerías e incluso en el propio resultado. Lo importante es que la realidad de esta función se cumpla siempre, independientemente de si
# el resultado es diferente entre una librería y otra. 
# They key is to get this equation to be true: 𝑋𝑣=𝜆𝑣. The details can vary.