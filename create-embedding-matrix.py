
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
from numpy import array, argmax
from pickle import dump,load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

with open('data.pickle', 'rb') as f:
   tokenized_questions, tokenized_answers, question_vocab, answer_vocab, question_w2id, question_id2w, answer_w2id, answer_id2w =  load(f)
   

def max_length(lines):
    return max(len(line) for line in lines)

def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes = vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size) 
    return y


initial_ans_vocab = 14802
start_token_index = initial_ans_vocab
end_token_index = initial_ans_vocab + 1
ans_tokenizer = Tokenizer()
ans_tokenizer.word_index = answer_w2id
ans_tokenizer.word_index['SOS'] = start_token_index
ans_tokenizer.word_index['EOS'] = end_token_index
ans_vocab_size = len(ans_tokenizer.word_index) + 1
ans_length = max_length(tokenized_answers)
print('Answer Vocabulary Size: %d' % ans_vocab_size)
print('Answer Max Length: %d' % (ans_length))

ques_tokenizer = Tokenizer()
ques_tokenizer.word_index = question_w2id
ques_vocab_size = len(ques_tokenizer.word_index) + 1
ques_length = max_length(tokenized_questions)
print('Question Vocabulary Size: %d' % ques_vocab_size)
print('Question Max Length: %d' % (ques_length))
 

# Creating embedding matrix for questions and answers
EMBEDDING_DIM_ques = 300
EMBEDDING_DIM_ans = 300
embedding_matrix_ques = np.zeros((len(ques_tokenizer.word_index) + 1, EMBEDDING_DIM_ques))
embedding_matrix_ans = np.zeros((len(ans_tokenizer.word_index) + 1, EMBEDDING_DIM_ans))
empty_matrix = np.zeros((1, EMBEDDING_DIM_ques))
f = open('pre-trained.txt','r',encoding = "utf-8")
lines = f.readlines()
print(len(lines))
print(lines[0].split())
l = int(len(lines)/10)
for k in range(0,10):
    embeddings_index = {}
    if k!=9:
        for p in range(k*l,(k+1)*l):
            values = lines[p].split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
    
    else:
        for p in range(k*l,int(len(lines))):
            values = lines[p].split()
            word = values[0]
            coefs = np.asarray(values[1:])
            embeddings_index[word] = coefs
    
    for word, i in ques_tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        
        if (embedding_vector is not None) and (np.array_equal(embedding_matrix_ques[i].reshape(1,embedding_matrix_ques[i].shape[0]),empty_matrix)):
            # words not found in embedding index will be all-zeros. - for ex:- start and end tokens
            if (len(embedding_vector)) > 300:
                for k in range(0,len(embedding_vector)-300):
                    embedding_vector = np.delete(embedding_vector, [0])
            embedding_matrix_ques[i] = embedding_vector
        
    for word, i in ans_tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and (np.array_equal(embedding_matrix_ans[i].reshape(1,embedding_matrix_ans[i].shape[0]),empty_matrix)):
            # words not found in embedding index will be all-zeros. - for ex:- start and end tokens
            if (len(embedding_vector)) > 300:
                for k in range(0,len(embedding_vector)-300):
                    embedding_vector = np.delete(embedding_vector, [0])
            embedding_matrix_ans[i] = embedding_vector


f.close()
print(len(embedding_matrix_ques))
print(embedding_matrix_ques[1])
print(embedding_matrix_ans[1])
np.savetxt('embedding_matrix_ques.txt',embedding_matrix_ques,delimiter = ',')
np.savetxt('embedding_matrix_ans.txt',embedding_matrix_ans,delimiter = ',')
