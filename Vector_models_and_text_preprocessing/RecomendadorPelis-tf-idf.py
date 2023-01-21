# Build a recommendation system with TF-IDF
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



peliculas = pd.read_csv("archive/tmdb_5000_movies.csv")

columnas_a_utilizar = peliculas[['genres','original_title','keywords']]
generos = peliculas.genres.unique()
titulos = peliculas['original_title']
keywords = peliculas['keywords']
# print(columnas_a_utilizar['genres'])
# print(columnas_a_utilizar['original_title'][0])
# print(columnas_a_utilizar['keywords'][0])
# print(generos)
# print(titulos)
# print(keywords)

def nuevo_diccionario_pelis(peliculas):
    diccionario_pelis = {}
    lista_generos = []
    lista_keywords = []
    lista_titulos = []
    
    
    for i in peliculas['genres']:
        genero_final = ""
        i = re.sub(r"{|}|\"","",i)
        matches_no_limpios = re.findall(r"name:\s\w*,|name:\s\w*\s\w*",i)
        for i in matches_no_limpios:
            genero = re.sub(r"^name:\s|,","",i)
            genero_final = genero_final + " " + genero
        genero_final = re.sub(r"^\s|\s{2,}","",genero_final)
        lista_generos.append(genero_final)

    for i in peliculas['keywords']:
        keywords_final = ""
        i = re.sub(r"{|}|\"","",i)
        matches_no_limpios = re.findall(r"name:\s\w*,|name:\s\w*\s\w*",i)
        for i in matches_no_limpios:
            keyword= re.sub(r"^name:\s|,","",i)
            keywords_final = keywords_final + " " + keyword
        keywords_final = re.sub(r"^\s|\s{2,}","",keywords_final)
        lista_keywords.append(keywords_final)

    for i in peliculas['original_title']:
        lista_titulos.append(i)

    for i in range(len(titulos)):
        genero = lista_generos[i]
        keywords = lista_keywords[i]
        movie_data = genero + " " + keywords
        titulo = lista_titulos[i]
        diccionario_pelis[titulo] = movie_data

    # print(lista_titulos)

    return diccionario_pelis

corpus = nuevo_diccionario_pelis(columnas_a_utilizar)
# corpus es un diccionario que contiene el titulo de la peli como key y los datos de la peli como value

datos_pelis = [datos for datos in corpus.values()]
# datos_pelis contiene todos los datos de las pelis
titulos = [titulo for titulo in corpus.keys()]
# Instanciamos el tfIdfVectorizer
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
# Lo pasamos a los datos
tfIdf = tfIdfVectorizer.fit_transform(datos_pelis)
# Creamos un DF con cada dato para visualizar
# df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print (df.head(25))
# print(len(datos_pelis))
# print(tfIdf.shape)
# print(type(tfIdf))
# print(tfIdf[0])

def ejecutar_recomendacion(titulos):
    titulo_pelicula = input("Introduce el titulo de la película y te daré una recomendación: ")
    if titulo_pelicula in titulos:
        indice_pelicula = titulos.index(titulo_pelicula)
    else:
        print("esta película no está en nuestro catálogo, inténtalo con otra película")
    vector_pelicula = tfIdf[indice_pelicula] 
    similitud = cosine_similarity(vector_pelicula, tfIdf)
    similitud = similitud.T
    df_similitud = pd.DataFrame(similitud)
    df_similitud_colocado = df_similitud.sort_values(by = 0, ascending=False)
    lista_peliculas_recomendadas = df_similitud_colocado.index[1:6].values
    print("Te recomendamos esta película: ")
    for i in range(len(titulos)):
        if i in lista_peliculas_recomendadas:
            print(titulos[i])


ejecutar_recomendacion(titulos)