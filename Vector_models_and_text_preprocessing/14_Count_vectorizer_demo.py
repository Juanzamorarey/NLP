# En esta parte vamos a hacer conteo de vectores con código. Se trata de una pequeña aproximación al ML. 
# Si bien con sklearn podríamos hacer esto en 3 líneas de código vamos a desgranar todo el proceso para comprenderlo mejor:

# Esto implica, realmente, 3 líneas de código y 3 pasos:
#  1- Crear un modelo
#  modelo = elmodeloquetegustemas()
#  2- Entrenar el modelo
#  modelo.fit(X_train, Y_train)
#  3- probar el modelo
#  modelo.score(X_train, Y_train)
#  modelo.score(X_test, Y_test)

# Estos pasos se pueden aplciar casi a cualquier modelo, ya sea de Data science, NLP, ML o la tia rita. 
# Empecemos importando lo necesario

# Librerías para el manejo de los datos
import numpy as np
import pandas as pd
import sklearn
# sklearn para el entrenamiento y uso de modelos
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# MultinomialNB es un clasificador
from sklearn.model_selection import train_test_split

# Nltk para el manejo del texto
import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Ahora debemos descargarnos varias BBDD de nltk para incorporar al lemmatizer
# nltk.download("wordnet")
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# Se comenta para no cargarlas cada vez

doc = "documents_for_course/bbc_text_cls.csv"

doc_df = pd.read_csv(doc)

print(doc_df.head())


# Como podemos ver a partir del head se trata de unos datos con dos columnas: texto y etiqueta
# Recordemos que se trata de una tarea supervisada puesto que estaremos dando un entrenamiento 
# en el que los datos de training contienen ya las etiquetas.
# Para poder tratar los datos de los que disponemos de forma eficaz vamos a crear dos variables que contengan
# en el primer caso el texto y en el segundo las etiqeutas

textos = doc_df["text"]
etiquetas = doc_df["labels"]

# A continuación queremos crear un histograma de nuestros datos. De esta manera podremos ver si dentro de nuestros
# datos todas las clases están bien representadas. Si el 90% de nuestros artículos de periódico corresponden a deporte
# y solo un 1% corresponde a internacional, es probable que nuestro modelo no lo haga muy bien con internacional. 
# veamos por tanto nuestro histograma. (recordemos que un histograma no es más que una representación gráfica de
# los datos con los que contamos. en este caso veremos qué porcentajes corresponden a cada etiqueta, si estuviéramos
# midiendo por ejemplo distribuciones de pesos tendríamos que establecer una serie de intervalos en lso qeu dividir 
# nuestrod datos).

etiquetas.hist(figsize=(10,5))
# En google colab esto muestro el historiograma, en VS no funciona correctamente pero extrae algo similar a esto

# 500                         
# 400       |              |       
# 300       |      |       |       |       |     
# 200       |      |       |       |       |
# 100_______|______|_______|_______|_______|__
#     bussiness politics sports  tech  entertainment

# A la vista del histograma podemos ver que nuestros datos tienen 5 etiqeutas y una cantidad representativa de cada
# tipo sobre el total del número de datos. A raíz de esto decimos que nuestros datos están balanceados.

# Procedemos ahora a hacer la división entre train y test antes de usar el count_vectorizer.

textos_train, textos_test, Ytrain, Ytest = train_test_split(textos, etiquetas, random_state=123)

# De momento como estamos en los primeros pasos usaremos el countVectorizer vacío pero recordemos que esta
# instanciación posee múltiples parametros que podemos pasar para mejorar las predicciones de nuestro modelo.

vectorizer = CountVectorizer()

# Después usando el vectorizer entrenamos nuestro modelo y aplicamos solamente el transform al set de test ¿por qué?
# porque si utilizáramos fit_transform para los datos de test estaríamos falseando el resultado. recordemos que test
# nos sirve para ver cómo de preciso será nuestro modelo sobre un set de datos que no se han vito antes, por lo tanto
# no hacemos fit porque no será de esta manera como utilizaremos los datos de test. 

Xtrain = vectorizer.fit_transform(textos_train)
Xtest = vectorizer.transform(textos_test)

# Una vez hecho esto imaginemos que queremos ver Xtrain. Xtrain contendrá, tal y como recordamos de lecciones anteriores
# los vectores de pasar nuestro texto (letras y palabras) a vectores (números utilizables por ML). Es decir Xtrain
# contiene filas y columnas con el número de apariciones de cada palabra. Si intentáramos hacer un print de Xtrain 
# veríamos que se trata de un sparse matrix ¿por qué? porque la mayoría de números presente son 0. Además veríamos 
# que la matriz tiene unas dimensiones de 1668x26287. Esto en principio debería alarmarnos ya que en Ml normalmente
#  lo deseable es tener más filas que columnas ya que si no la dimensionalidad del modelo es muy grande. En este caso
# en cambio lo tomaremos como algo deseable.
# 
# 
# En el siguiente bloque de código vamos a hacer los pasos usuales del ML

# 1o creamos la instancia del modelo:

modelo = MultinomialNB()

# 2o entrenamos el modelo

modelo.fit(Xtrain, Ytrain)

# Comprobamos el accuracy del modelo sobre los datos de training (recordemos que tienen una etiqueta previa) y de testing
print("train_scores: ", modelo.score(Xtrain,Ytrain))
print("test_scores: ", modelo.score(Xtest,Ytest))

# train_scores:  0.9922062350119905
# test_scores:  0.9712746858168761

# Como vemos los resultados ya son de por sí bastante buenos. Vamos a intentar batir ahora el resultado  partir del uso de
# parametros del count_vectorizer. Vamos a repetir todo pero eliminando las stop_words. Para esto usamos el argumetno de 
# count vectorizer stopwords

vectorizer = CountVectorizer(stop_words="english")
Xtrain = vectorizer.fit_transform(textos_train)
Xtest = vectorizer.transform(textos_test)
modelo = MultinomialNB()
modelo.fit(Xtrain, Ytrain)
print("train_scores_no_stopwords: ", modelo.score(Xtrain,Ytrain))
print("test_scores_no_stopwords: ", modelo.score(Xtest,Ytest))

# train_scores_no_stopwords:  0.9928057553956835
# test_scores_no_stopwords:  0.9766606822262118

# Como vemos hemos subido un poco el accuracy en test (aproximadamente un 0.005%)

# ¿Y si aplicáramos lematizaciones a este modelo?. Para ello vamos a crear una función y una clase que nos ayuden a
# llevarlo a cabo

def get_pos_for_wordnet(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Esta clase creará un objeto que podrá tokenizar y lemmatizar el texto por si solo
class LemmaTokenizer:
    def __init__(self):
        # En la clase creadora introducimos el WordNetLemmatizer
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        # En la función call tomamos un argumento, el documento que vamos a tokenizar. Para ello usaremos
        # el tokenizer de nltk sobre nuestro argumetno
        tokens = word_tokenize(doc)
        # Ahora necesitamos el POS para ello vamos a usar de nuevo nltk, recordemos que después tendremos que
        # convertir estas etiquetas a las que nos convienen para usar el lemmatizer
        palabras_y_etiquetas = nltk.pos_tag(tokens)
        #recordemos que palabras_y_etiquetas es una lista que contiene tuplas. Por lo tanto vamos a devoler
        # a través de una lista de comprensión una lista que contiene cada lemma del texto de los docs
        # en el input
        return [self.wnl.lemmatize(palabra, pos=get_pos_for_wordnet(etiqueta)) for palabra, etiqueta in palabras_y_etiquetas]

# Esta clase por tanto nos devuelve un objeto que podemos usar como argumento para el tokenizer del CountVector puesto que
# se trata de una serie de tokens. Probemos

#  SE COMENTA POR VELOCIDAD DEL POGRAMa

# vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
# Xtrain = vectorizer.fit_transform(textos_train)
# Xtest = vectorizer.transform(textos_test)
# modelo = MultinomialNB()
# modelo.fit(Xtrain, Ytrain)
# print("train_scores_lemmatizer: ", modelo.score(Xtrain,Ytrain))
# print("test_scores_lemmatizer: ", modelo.score(Xtest,Ytest))

# train_scores_lemmatizer:  0.9922062350119905
# test_scores_lemmatizer:  0.9676840215439856

#  SE COMENTA POR VELOCIDAD DEL POGRAMa

# Como podemos observar ahora no solamente el score de test ha bajado si no que además el procesamiento ha aumentado
# notablemente, siendo este el método que más ha tardado. No importa lo sofisticado que sea un método, debe adaptarse 
# a la realidad de los datos con los que contamos. 

# Dado que la lemmatización no nos ha dado buenos resultados, vamos a probar ahora con el stemming. 
# Vamos a crear una clase similar a la anterior pero que ahroa nos devuelva una lsita a partir de un 
# objeto del PorterStemmer() recordemos que la lógica de este método reside en cortar los morfemas
# derivativos, sufijos y flexiones para obtener el root de la palabra. 

class StemTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()
        # instanciamos un objeto del tipo PorterStemmer()
    def __call__(self, doc):
        # Al llamar a la función tokenizamos el documento y devolvemos el resultado de aplicar 
        # sobre cada elemento de la lista resultante el porterStemmer
        tokens = word_tokenize(doc)
        return [self.porter.stem(token) for token in tokens]

# repetimos proceso para comprobar el accuracy

#  SE COMENTA POR VELOCIDAD DEL POGRAMa

# vectorizer = CountVectorizer(tokenizer=StemTokenizer())
# Xtrain = vectorizer.fit_transform(textos_train)
# Xtest = vectorizer.transform(textos_test)
# modelo = MultinomialNB()
# modelo.fit(Xtrain, Ytrain)
# print("train_scores_stemmer: ", modelo.score(Xtrain,Ytrain))
# print("test_scores_stemmer: ", modelo.score(Xtest,Ytest))

# train_scores_stemmer:  0.9892086330935251
# test_scores_stemmer:  0.9694793536804309

#  SE COMENTA POR VELOCIDAD DEL POGRAMa

# Los resultados son de nuevo peores y, aunque el tiempo de procesamiento es menor que con el lemmatizer tampoco
# hemos tenido una performance muy rápida. Vamos entonces a probar con un tokenizador mucho más sencillo,
#  la función split de la clase string de python. Así dividiremos el texto a aprtir de los espacios.

def tokenizador_simple(doc):
    return doc.split()

vectorizer = CountVectorizer(tokenizer=tokenizador_simple)
Xtrain = vectorizer.fit_transform(textos_train)
Xtest = vectorizer.transform(textos_test)
modelo = MultinomialNB()
modelo.fit(Xtrain, Ytrain)
print("train_scores_tokenizador_simple: ", modelo.score(Xtrain,Ytrain))
print("test_scores_tokenizador_simple: ", modelo.score(Xtest,Ytest))


# train_scores_tokenizador_simple:  0.9952038369304557
# test_scores_tokenizador_simple:  0.9712746858168761


# Como vemos por los reusltados. A veces lo simple es lo más eficaz. 


# Algo sobre lo que merece la pena reflexionar es el conteo de las matrices. ¿Por qué en cada caso se reducirá 
# el número de filas? porque la utilización de lemmatizer o stemmer reduce el vocabulario del lenguaje haciendo
# que ocurrencias de varias palabras (run, ran, running) desaparezcan reduciendo por tanto la variabilidad
#  y dimensionalidad de los vectores.