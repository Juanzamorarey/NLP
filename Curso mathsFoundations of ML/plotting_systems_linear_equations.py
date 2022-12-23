# Aquí vamos a hacer el gráfico del ejercicio que hicimos del ladrón y el sheriff pero en python. 
import numpy as np
import matplotlib.pyplot as plt

tiempo = np.linspace(0,40,1000)
# el método linspace nos permite crear puntos desde un inicio (primer parametro) a un final (segundo parametro),
# el tercer método nos permite elegir cuántos puntos queremos entre los dos primeros. En este caso hemos puesto
# 1000 para tener mucho detalle.

# recordemos que la distancia recorrida por el ladrón era d = 2.5t porque recorría 2,5 km por minuto
distancia_ladron = 2.5 * tiempo

# El sheriff en cambio recorría 3 km por minuto pero sale 5 minutos después
distancia_sheriff = 3 * (tiempo-5)

# Aquí creamos el gráfico
fig, ax = plt.subplots()
# Aquí introducimos los títulos para el gráfico y los ejes
plt.title("El robo de un banco")
plt.xlabel("Tiempo en minutos")
plt.ylabel("Distancia en km")
# Aquí introducimos conde queremos que acabe x e y
ax.set_xlim([0,40])
ax.set_ylim([0,100])
# Aquí indicamos las rectas que queremos hacer. En este caso en la recta de tiempo
# Introducimos la función que daba el tiempo del ladrón y lo coloreamos de verde y la del sheriff de rojo
ax.plot(tiempo,distancia_ladron, c="green")
ax.plot(tiempo,distancia_sheriff, c="red")
# Aquí mostramos un punto a partir de dos rectas. En este caso la recta horizontal x= 30
# e y =75 vertical. Hemos introducido estos datos porque de antemano sabíamos donde coincidían las rectas.
plt.axvline(x=30, color="purple")
plt.axhline(y=75, color="purple")
# Por último ordenamos que se ejecure el gráfico.
plt.show()