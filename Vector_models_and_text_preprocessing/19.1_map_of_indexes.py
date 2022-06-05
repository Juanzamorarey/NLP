frase_1 = "Me gustan los perros"
frase_2 = "Me gustan los gatos"
frase_3 = "No Me gustan los perros"
frase_4 = "No Me gustan los gatos"

docs = []

docs.append(frase_1)
docs.append(frase_2)
docs.append(frase_3)
docs.append(frase_4)

map_index = {}
indice = 0

# Aquí creamos un diccionario que contiene un índice para cada palabra
for doc in docs:
    for token in doc.split():
        if token not in map_index:
            map_index[token]=indice
            indice+=1

print(map_index)
# {'Me': 0, 'gustan': 1, 'los': 2, 'perros': 3, 'gatos': 4, 'No': 5}

frase_5 = "No me gustan los cerdos"

# Aquí devovlemos los índices a partir de una frase
for i in frase_5.split():
    if i in map_index:
        print(map_index[i])

# 5
# 1
# 2

Vectores = [2,3,5,0]

# Aquí devolvemos las frase a partir de un ´dince
for i in Vectores:
    for key,value in map_index.items():
        if i == value:
            print(key)

# los
# perros
# No
# Me

# Ya tenemos una manera sencilla de devolver la palabra para cada vector y viceversa.

