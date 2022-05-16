import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.tree import plot_tree as plt


# path = "documents_for_course/tmdb_5000_movies.csv"

# def analisis_peliculas(path):
#     lista_strings_pelis = []
#     documento = pd.read_csv(path,index_col="id")
#     documento_no_nulos = documento.dropna()
#     for index, row in documento_no_nulos.iterrows():
#         # print(row)
#         # print("Este es el genero" + row["genres"] + "Estas son las palabras clave" + row["keywords"] )
#         generos_pelicula = ""
#         palabras_clave = ""
#         generos = row["genres"]
#         keywords = row["keywords"]
#         titulo = row["original_title"]
#         diccionario_generos = json.loads(generos)
#         for i in diccionario_generos:
#             generos_pelicula = generos_pelicula + " " + i["name"]
#         # print(fila_1)
#         diccionario_titulos = json.loads(keywords)
#         for i in diccionario_titulos:
#             palabras_clave = palabras_clave + " " + i["name"]
        
#         # texto_pelicula = generos_pelicula + palabras_clave
#         pelicula_final = [f"{titulo}",f"{palabras_clave}",f"{generos_pelicula}"]
#         lista_strings_pelis.append(pelicula_final)

#     return lista_strings_pelis

# datos = analisis_peliculas(path)

# # print(datos[0])
# tfidf = TfidfVectorizer(analyzer="word")
# lista_strings = []
# for i in datos:
#     string_final = ""
#     for x in i:
#         string_final = f"{string_final} {x}" 
#     lista_strings.append(string_final)

# recomendador_tfidf = tfidf.fit_transform(lista_strings)
# # El TF-IDF se alimenta de todos los datos unidos en un solo string pero la posterior conversion
# # a vector requiere de una lsita


# titulos_con_peliculas = []

# # contador = 0
# # for i in textos:
# #     titulos_con_peliculas.append(f"{etiquetas[contador]}:{i}")
# #     contador +=1

# # diccionario = {}


# def añadir_vectores(pelicula):
#     for pelicula_elegida in datos:
#         if pelicula in pelicula_elegida[0]:
#             vector = tfidf.transform(pelicula_elegida)
#     lista_vectores_similares = {}
#     for peliculas_similares in datos:
#         if pelicula not in peliculas_similares[0]:
#             vector_a_comparar = tfidf.transform(peliculas_similares)
#             similitud = cosine_similarity(vector, vector_a_comparar)
#             lista_vectores_similares[peliculas_similares][0] = similitud.get(max)
    

documento = pd.read_csv("documents_for_course/tmdb_5000_movies.csv")

# print(documento.head())

# x = documento.iloc[0]
# print(x)
# Miramos la primera linea

# print(x["genres"])
# Es un alista de jsons con 2 valores
# print(x["keywords"])
# Es un alista de jsons con 2 valores



# Unimos todos los géneros y palabras clave en un solo tring que es lo que requiere el TF-IDF
def generos_y_keywords_string(fila):
    diccionario_generos = json.loads(fila["genres"])
    diccionario_pclave = json.loads(fila["keywords"])
    generos_pelicula = ""
    palabras_clave = ""
    generos_pclave =""
    for i in diccionario_generos:
        # Para unir género como Science fiction en un solo string y no aumentar 
        # la dimensionalidad lo adjuntamos de esta manera e igual en las keywords
        generos_pelicula = generos_pelicula + " " + "".join(i["name"].split())
    for i in diccionario_pclave:
        palabras_clave = palabras_clave + " " + "".join(i["name"].split())
    
    generos_pclave = f"{generos_pelicula} {palabras_clave}"

    return generos_pclave

# Añaidmos el reusltado de unir estas palabras al df
documento["string"] = documento.apply(generos_y_keywords_string, axis=1)
# print(documento.head())
# Lo tenemos todo listo, vamos a vrear una isntancia del TFIDF y asignarle
# un número máximo de dimensiones (2000) de esto modo se tomarán únicamente las 
# 2000 palabras más relevantes
# x = documento.iloc[0]
# print(x)
tfidf = TfidfVectorizer(max_features=2000)

X = tfidf.fit_transform(documento["string"])

print(X.shape)
# (4803, 2000)
# Como vemos tiene las 2000 columnas que hemos puesto como limite con las 2000 palabras
# más frecuentes. Ahora vamos a crear algún tipo de mapa que nos sirva para
# localizar a qué película corresponde cada índice. Para ello creamos un objeto Series 
# de pandas.

# DEFINICION objeto Series de Pandas

# La serie Pandas es una matriz etiquetada 
# unidimensional capaz de contener datos de 
# cualquier type(entero, cadena, flotante, 
# objetos de python, etc.). Las etiquetas de 
# los ejes se denominan colectivamente índice.
# Las etiquetas no necesitan ser únicas, pero 
# deben ser de tipo hash.

pelicula_a_indice = pd.Series(documento.index, index=documento["title"])
# print(pelicula_a_indice.head())

# title
# Avatar                                      0
# Pirates of the Caribbean: At World's End    1
# Spectre                                     2
# The Dark Knight Rises                       3
# John Carter                                 4

# Ahora tenemos una manera de saber qué índice aplica a qué película de modo que podemos
# ver qué vector es para qué película.

# Por ejemplo si queremos ver qué índice corresponde a Scream 3

# print(pelicula_a_indice["Scream 3"])
# 1164

# Vamos a guardar esto en una variable

# pelicula_buscada = pelicula_a_indice["Scream 3"]

# busqueda = X[pelicula_buscada]

# Ahora mismo búsqueda contiene el vector correspondiente a Scream_3
# print(busqueda.shape)
# (1, 2000)
# Esto tiene sentido puesto que es una película con las 2000 columnas que hemos determinado

# podemos pasarlo a un array para ver mejor lso datos
# busqueda.toarray()

# Ahora vamos a ejecutar el cosine_similarity entre nuestra película y todo el resto de
# películas presentes en el df. Al no poner un dato de comparación extraerá las similitudes
# mayores dentro de los datos proporcionados.

# scores = cosine_similarity(busqueda, X)
# print(scores)
# [[0. 0. 0. ... 0. 0. 0.]]


# La mayoría de valores son 0 esto tiene sentido porque muchas palabras presentes en 
# este tipo de película no estarán presentes en otros tipos. 

# Ahora dado que nuestro array en 1-N vamos a hacer un flattern para convertirlo en 1-D array.

# scores = scores.flatten()

# Para visualizarlo podemos usar

# plt.plot(scores)

# Al visualizarlo vemos que tenemos un 1 y una serie de vectores que no llegan al 0.4 en
# similitud.

# Ahora necesitamos colocar estos valores en ordean descendente (pueto que queremos los 5
# primeros. 

# print((-scores).argsort())
# [1164 3902 4628 ... 1714 1720 4802]

# Como vemos esto tiene sentido porque ahora vemos los índices colocados en orden descendiente
# comenzando por el más similar a nuestra película Scream 3. Dado que no hemos eliminado nuestra
# propia película del conteo general vemos que está pareciendo la 1a (recordemos que 
# 1164 era el índice de Scream 3) puesto que son vectores idénticos. 

# plt.plot(scores[(-scores).argsort()])
# Aquí vemos que lso resultados tienen mucho más sentido

# Finalmente solo tenemos que rescatar, a aprtir de los índices las posiciones de 1:6
# ya que 0 será la propia pelñícula. Y ya habremos completado satisfactoriamente la recomendación

# peliculas_recomendadas = (-scores).argsort()[1:6]

# fin = documento["title"].iloc[peliculas_recomendadas]

# print(fin)

# 3902    Friday the 13th Part VI: Jason Lives
# 4628                          Graduation Day
# 4053        Friday the 13th: A New Beginning
# 4048                             The Calling
# 1084                         The Glimmer Man

# Vamos ahora a crear la función

def recomendar(pelicula):
    pelicula_recomendada = pelicula_a_indice[pelicula]
    # El API de pandas es un poco inconsistente por loq ue vamos a comrpobar que
    # se trata de un obejto Series aquí antes de seguir.
    if type(pelicula_recomendada) == pd.Series:
        # localizamos el ídnice
        pelicula_recomendada = pelicula_recomendada.iloc[0]

    # creamos el vector de la pelicula interesada
    busqueda = X[pelicula_recomendada]
    # creamos el cosine_similarity con el resto de películas
    scores = cosine_similarity(busqueda, X)

    # nuestro array en 1-N lo convertimos en 1-D array.
    scores = scores.flatten()

    peliculas_recomendadas = (-scores).argsort()[1:6]

    return documento["title"].iloc[peliculas_recomendadas]

print(recomendar("Scream 3"))

# 3902    Friday the 13th Part VI: Jason Lives
# 4628                          Graduation Day
# 4053        Friday the 13th: A New Beginning
# 4048                             The Calling
# 1084                         The Glimmer Man

print(recomendar("Mortal Kombat"))

# 1611              Mortal Kombat: Annihilation
# 1670                       DOA: Dead or Alive
# 3856              In the Name of the King III
# 1001    Street Fighter: The Legend of Chun-Li
# 2237                        Alone in the Dark

print(recomendar("Hulk"))

# 4424                               Fear Clinic
# 215     Fantastic 4: Rise of the Silver Surfer
# 30                                Spider-Man 2
# 174                        The Incredible Hulk
# 864                                   Blade II