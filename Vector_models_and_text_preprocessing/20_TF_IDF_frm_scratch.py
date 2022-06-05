# Aquí vamos a construir nuestro TF-IDF totalmente desde 0.

import nltk
import pandas as pd
import numpy as np
from nltk import word_tokenize

# nltk.download("punkt")


df = pd.read_csv("documents_for_course/bbc_text_cls.csv")

# print(df.head())

# Lo que vamos a intentar ver aquí es el funcionamiento intrínseco del TF-IDF. Las palabras que recojamos serán las que
# presenten los valores del TF-IDF más altos, es dcir las más representativas de cada documento. 
# Deben cumplir dos condiciones:

# Primero -> Son palabras que aparecen frecuentemente en los documentos, es decir el numero de frecuencia de término 
#(TF) es alto

#Segundo -> Deben ser palabras relativamente únicas comparadas con todo el set de datos, por lo que su frecuencia 
# en el documento (DF) es baja pero su frecuencia inversa en el documento (IDF) es alta. 

# Ahora que hemos importado nuestras librerías deberíamos poblar nuestro mapa de índices de palabras (word2index mapping)
# Recordemos que el word 2 index mapping es el resultado de decidir en qué columna se coloca qué palabra.

# Para ello podemos hacer lo siguiente:
indice = 0
word2indx = {}
# En este caso lo estructuraremos en forma de diccionario para cada palabra. Recordemos que se trata de una lista 
# de diccionarios
documentos_tokenizados = []
# Creamos una lista vacía donde almacenaremos los documentos tokenizados
for doc in df["text"]:
    palabras = word_tokenize(doc.lower())
    doc_en_numeros = []
    # Aquí almacenaremos los dicionarios
    for palabra in palabras:
        if palabra not in word2indx:
            # Si la palabra todavía no tiene un índice se lo asignamos
            word2indx[palabra] = indice
            indice+=1
            # Añadimos 1 al índice de modo que la proxima vez que se utilice tenga otro valor. Nunca diferentes palabras
            # tendrán el mismo índice.

        doc_en_numeros.append(word2indx[palabra])
    documentos_tokenizados.append(doc_en_numeros)

# Ahora que tenemos nuestro word2index mapping preparado y completo necesitamos tener preparado también
# el paso contario, esto lo ahcemos rápidamente con una lista de comprensión.
idx2word = {v:k for k, v in word2indx.items()}

# Nos faltan más variables para poder aplicar la formula del TF-IDF, primero el número de documentos:

N = len(df["text"])

# Después el vocabulario, o número de palabras total del dataset:

V = len(word2indx)

# Con esto ya podemos crear nuesta matriz de frecuencia. En este caso será una matriz densa que consistirá únicmanete
# en un array de numpy. En este caso será una matriz de 0 de tamaño NxV

tf = np.zeros((N,V))

# Ahora que tenemos nuestra matriz vamos a poblarla con nuestros datos:

for i, doc_en_numeros in enumerate(documentos_tokenizados):
    # si nos fijamos al usar enumerate podemos ver qué documento estamos mirando gracias a la variable i del bucle
    for j in doc_en_numeros:
        # dentro de este documento en concreto hacemos un 2o bucle que recorre el word2index en el documento
        tf[i,j] +=1
        # Ahora que tenemos esas posiciones sumamos un 1 a la posición correspondiente a la fila i (documento) 
        # coincidente con la columna j (palabra). Puesto que hemos indicado antes que el documento contiene esta palabra.


# A partir de aquí ya tenemos la primera parte de la formula, es deicr nuestra matriz de frecuencia (TF) ahora necesitamos
# realizar la segunda parte la matriz inversa de frecuencia del codumento. El IDF se puede calcular a partir 
# del TF. ¿Cómo? 

# recordemos que por cada palabra tendremos un valor inverso por lo que tendremos V valores de IDF (Recuerda que habíamos
# almacenado el número de palabras en V), o en otras palabras un vector del tamaño de V. Por lo tanto debemos mirar
# las columnas de nuestra matriz de frecuencia una a una puesto que cada una corresponde a una palabra del vocabulario.
# Asi que por cada columna tendremos un 0 si la palabra no apareció en el documento y un número mayor a 0 si la palabra
# apareció en el documento. Podemos crear una matriz de valores boooleanos a partir de la anterior con las mismas dimensiones
# Dado que en python True = 1 y False = 0, de este modo si queremos saber cuantos valores son verdaderos solamente
# debemos sumar todos los valores puesto que los false al ser 0 no efecturarán ningún cambio. Para que la suma se aplique
# para cada columna es importante introducir el parametro axis con el valor 0, ya que si no nos dará la suma de apariciones
# de 1 en la matriz.

document_freq = np.sum(tf > 0, axis = 0) #frecuencia de documentos de tamaño V

# Por lo tanto tenemos un array the numpy que contiene el número de documentos en los que cada palabra aparece.
# recordemos que la fórmula del IDF contiene el uso del logaritmo para aplastar los valores muy altos de cara a la 
# dimensionalidad del modelo de ML. Por lo tanto ya podemos aplicar la formula logaritmo del número de documentos
# dividido entre la frecuencia de documentos:

idf = np.log(N/document_freq)

# Ahora que tenemos el TF y el IDF ya podemos calcular el TF-IDF. Recordar que estamos multiplicando TF que es una matriz
# de tamaño (N,V) y IDF es un vector de tamaño V. Numpy arregla automáticamente estas operaciones para no tener que
# preocuparnos de convertir el vector en matriz o viceversa. 

tf_idf = tf * idf

# Ya tenemos por tanto nuestro tf_idf, vamos a seleccionar ahora algunos documentos de nuestro dataset y pedirle que
# nos muestre las 5 palabras más representativas de el documento en terminos de valores para el TF-IDF:

i = np.random.choice(N) 
# Selecciona un número entre 0 y N de forma aleatoria
fila = df.iloc[i]
# Selecciona el documento con el indice elegido aleatoriamente
print("Etiqueta: ", fila["labels"])
# Impirme a qué tipo de noticia pertenece el texto
print("Texto: ", fila["text"].split("\n",1)[0])
# Imprime la primera linea para ver de qué trata (normalmente debería salir el titular)
print("Los 5 términos más representativos: ")

valoraciones = tf_idf[i]
# Del tf_idf selecciona el valor del índice
indices = (-valoraciones).argsort()
# Coloca los índices de manera inversa (-valoraciones) puesto que por defecto mostraría los datos en orden ascendente, 
# nosotros queremos los datos en forma descendente y no nos importa el valor en sí, si no el orden en el que aparecen
# los valores, por lo que necsitamos llamar a argsort para darnos los índices en el orden deseado.

# Finalmente para ese índice concreto imprime las primera 5 correspondencias de indice a palabra.
# Es para este paso que antes hemos creado el idx2word
for j in indices[:5]:
    print(idx2word[j])

# Etiqueta:  business
# Texto:  Barclays shares up on merger talk
# Los 5 términos más representativos:
# barclays
# fargo
# wells
# banking
# pre-tax

# Etiqueta:  entertainment
# Texto:  Musical treatment for Capra film
# Los 5 términos más representativos:
# musical
# thoday
# spend
# doren
# capra

# Etiqueta:  business
# Texto:  Bank payout to Pinochet victims
# Los 5 términos más representativos:
# pinochet
# gen
# victims
# riggs
# bank

# Como podemos ver en los ejemplos las palabras más representativas parecen estar funcionando correctmente. 