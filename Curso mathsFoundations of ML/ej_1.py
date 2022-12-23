import numpy as np

#  Ejercicio 1
matriz_i3= np.array([[1,0,0],[0,1,0],[0,0,1]])
matriz_k= np.array([[2/3,1/3,2/3],[-2/3,2/3,1/3],[1/3,2/3,-2/3]])

def ejercicios(matriz):
    columna_1 = matriz[:,0]
    columna_2 = matriz[:,1]
    columna_3 = matriz[:,2]


    resultado_1_2 = np.dot(columna_1,columna_2)
    resultado_1_3 = np.dot(columna_1,columna_3)
    resultado_2_3 = np.dot(columna_2,columna_3)

    if resultado_1_2 == 0 and resultado_2_3 == 0 and resultado_1_3 == 0:
        print("todas las columnas son ortogonales entre sí")

    #  Ejercicio 2

    resultado_1 = np.linalg.norm(columna_1)
    resultado_2 = np.linalg.norm(columna_2)
    resultado_3 = np.linalg.norm(columna_3)

    if resultado_1 == 1 and resultado_2 == 1 and resultado_3 == 1:
        print("todas las columnas tienen en norm unidad")

ejercicios(matriz_k)
