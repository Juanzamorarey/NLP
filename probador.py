import numpy as np
import re
import pandas as pd

with open('C:/Users/juan_/Desktop/programación/NLP_ML_DL_UdemyCourse/documents_for_course/Edgar_allan_poe.txt',encoding="utf-8", mode = "r") as doc:
    poe_dataset_noisy = doc.readlines()

with open('C:/Users/juan_/Desktop/programación/NLP_ML_DL_UdemyCourse/documents_for_course/Robert_frost.txt',encoding="utf-8", mode = "r") as doc:
    frost_dataset_noisy = doc.readlines()

def clean_dataset(dataset):
    clean_dataset = []
    for line in dataset:
        line = line.lower()
        line = re.sub(r"([-.,\\/#!$%\\^&\\*;:{}=\\-_`~()])","",line)
        line = re.sub(r"\n","",line)
        line = line.strip()
        clean_dataset.append(line)
    return clean_dataset

def divide_set(dataset):
    dataframe_set = pd.DataFrame({'Author': pd.Series(dtype='str'),'Poem': pd.Series(dtype='str')})
    for sentence in range(len(dataset)):
        list_row = ["Poe",dataset[sentence]]
        dataframe_set.loc[len(dataframe_set)] = list_row
        

    training_data = dataframe_set.sample(frac=0.8, random_state=25)
    testing_data = dataframe_set.drop(training_data.index)

    return training_data, testing_data

poe_dataset = clean_dataset(poe_dataset_noisy)
frost_dataset = clean_dataset(frost_dataset_noisy)

poe_training_data, poe_testing_data = divide_set(poe_dataset)
frost_training_data, frost_testing_data = divide_set(frost_dataset)

def get_vocabulary_size(dataset):
    number_of_states = {}
    counter = 0
    for index, row in dataset.iterrows():
        poem_line = row['Poem']
        for word in poem_line.split(" "):
            if word not in number_of_states:
                number_of_states[word] = counter
                counter +=1
    number_of_states["unknown_word"] = len(number_of_states) + 1
    return number_of_states

states_in_poe = get_vocabulary_size(poe_training_data)
states_in_frost = get_vocabulary_size(frost_training_data)

def transform_dataset_to_index_mapping(dataset, states_map):
    dataframe_indexed_set = pd.DataFrame({'Line': pd.Series(dtype=int),'Map': pd.Series(dtype=object)})
    counter = 0
    for index, row in dataset.iterrows():
        line_of_strings = row['Poem']
        line_of_indexes = []
        for word in line_of_strings.split(" "):
            if word in states_map.keys():
                line_of_indexes.append(states_map[word])
        dataframe_indexed_set.loc[counter] = [index,line_of_indexes]
        counter +=1
    return dataframe_indexed_set

states_in_poe = get_vocabulary_size(poe_training_data)
states_in_frost = get_vocabulary_size(frost_training_data)

poe_traning_indexed_df = transform_dataset_to_index_mapping(poe_training_data, states_in_poe)

frost_traning_indexed_df = transform_dataset_to_index_mapping(frost_training_data, states_in_frost)

# def create_skeleton_transitions_matrix(states):
#     V = len(states)
#     A = np.ones((V,V))
#     pi = np.ones(V)
#     return A, pi

def create_skeleton_transitions_matrix(states):
    number_of_states = list(states.values())
    number_of_states.append('unknown')
    dictionary_dataframe = {}
    for state in number_of_states:
        dictionary_dataframe[state] = 1
    dictionary_dataframe['unknown'] = 1
    dataframe_transiciones_estados = pd.DataFrame(dictionary_dataframe, index=number_of_states)
    dataframe_initial_vector = pd.DataFrame(number_of_states)

    return dataframe_transiciones_estados, dataframe_initial_vector, len(number_of_states) 

skeleton_A_poe = create_skeleton_transitions_matrix(states_in_poe)
skeleton_A_frost = create_skeleton_transitions_matrix(states_in_poe)

# print(skeleton_A_poe[0].shape, skeleton_A_poe[1].shape)

# def update_transition_matrix(sequences, matrix_skeleton, pi_skeleton):
#     # Matrix skeleton is the matrix of states
#     # pi skeleton is the vector of initial states
#     # sequences is a list of list which has all the sequences in the dataset
#     sequences_correct = sequences.iloc[:,1]
#     for index, sequence in enumerate(sequences_correct):
#         for element in sequence:

            


# transitions_df_frost = update_transition_matrix(poe_traning_indexed_df,skeleton_A_poe[0],skeleton_A_poe[1])

def create_transition_matrix(sequences, matrix_skeleton, pi_vector):
    # Dataset debe ser la columna map del df que contiene los índices
    # Poe_training_indexed_df es el df que contiene las secuencias por indices [frase [1,2,3]]
    # Possible states es el diccionario 'palabra':indice ('but':0) poe_states.value    
    # dataframe_estados = create_skeleton_transitions_matrix(dataset)
    
    for sequence in sequences.iloc[:,1]:
        for index, element in enumerate(sequence):
            if index == 0:
                pi_vector.at[element,0] +=1
            else:
                previous_state = sequence[index-1]
                matrix_skeleton.at[previous_state,element] +=1
    # Now we have the vector as well as the initial state for the model with the add one smoothing but we need to make the division at each row
    transition_matrix = pd.DataFrame()
    for index, row in matrix_skeleton.iterrows():
        # In this case we don't need to add to the denominator anythong because the number of states is already there thanks to the add one smoothing
        denominator = np.sum(row,axis=0)
        new_row = row/denominator
        transition_matrix[index] = new_row

    return transition_matrix, pi_vector

def get_sentence_probability(sentence, model, index_mapping):
    splitted_sentence = sentence.split(" ")
    trasnformed_to_index_mapping = []
    for word in sentence:
        if word in index_mapping:
            trasnformed_to_index_mapping.append(index_mapping[word])
        else:
            trasnformed_to_index_mapping.append('unknown')
    print(trasnformed_to_index_mapping)
    print(model)

get_sentence_probability('Of the old time entombed','hola', states_in_poe)