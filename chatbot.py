
import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed
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
 
# prepare training data

trainX = pad_sequences(tokenized_questions,maxlen=ques_length,padding='post')

X = np.array(tokenized_answers,copy=True)
    
for i in range(len(X)):
    X[i].insert(0,initial_ans_vocab)
        
trY = pad_sequences(X,maxlen=ans_length+1,padding='post')

for i in range(len(X)):
        X[i].append(initial_ans_vocab + 1)
    
trainY = pad_sequences(X,maxlen=ans_length+2,padding='post')
trainY = trainY[:,1:12]
l = len(X)
for i in range(0,l-1):
    k = len(X[i])
    for j in range(0,k-1):
        if j==0:
            del(X[i][j])
        if j==k-2:
            del(X[i][j])
            
# Just analysing data
print(trY[0])
print(trainY[0])
print(trainX[0])
print(trainX.shape)
print(trY.shape)
print(trainY.shape)
print(trainY.shape[0])
print(trainY.shape[1])
#print(tokenized_answers[0])
#print(X[0])
#print(tokenized_answers[1])
#print(X[1])

# Creating embedding matrix for questions and answers
EMBEDDING_DIM_ques = 300
EMBEDDING_DIM_ans = 300

embedding_matrix_ans = np.loadtxt("embedding_matrix_ans.txt",delimiter = ',')
embedding_matrix_ques = np.loadtxt("embedding_matrix_ques.txt",delimiter = ',')
####MODEL FOR TRAINING#####
latent_dim = 300

encoder_inputs = Input(shape=(ques_length,))      
encoder_embedding = Embedding(ques_vocab_size, EMBEDDING_DIM_ques)
encoder_inputs2 = encoder_embedding(encoder_inputs)
en_lstm_1 = LSTM(latent_dim, return_state=True, return_sequences = True)
en_lstm_2 = LSTM(latent_dim, return_state=True, return_sequences = True)
encoder_inputs2, state_h_1, state_c_1 = en_lstm_1(encoder_inputs2)
en_states_1 = [state_h_1, state_c_1]
encoder_inputs2, state_h_2, state_c_2 = en_lstm_2(encoder_inputs2)
en_states_2 = [state_h_2, state_c_2]
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, f_state_h, f_state_c = encoder_lstm(encoder_inputs2)
encoder_states = [f_state_h, f_state_c]
decoder_inputs = Input(shape=(ans_length+1,))
decoder_embedding = Embedding(ans_vocab_size,EMBEDDING_DIM_ans)
decoder_inputs2 = decoder_embedding(decoder_inputs)
dec_lstm_1 = LSTM(latent_dim, return_state=True, return_sequences = True)
dec_lstm_2 = LSTM(latent_dim, return_state=True, return_sequences = True)
decoder_inputs2, _1, _2 = dec_lstm_1(decoder_inputs2, initial_state=en_states_1)
decoder_inputs2, _3, _4 = dec_lstm_2(decoder_inputs2, initial_state=en_states_2)
decoder_lstm = LSTM(latent_dim,return_sequences = True,return_state=True)
decoder_outputs, f_h, f_c = decoder_lstm(decoder_inputs2,initial_state=encoder_states)
decoder_dense = Dense(ans_vocab_size,activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.summary()

###MODEL FOR TESTING#####

encoder_inputs_inf = Input(shape=(ques_length,))
encoder_inputs2_inf = encoder_embedding(encoder_inputs_inf)
encoder_inputs2_inf, state_h_1, state_c_1 = en_lstm_1(encoder_inputs2_inf)
encoder_inputs2_inf, state_h_2, state_c_2 = en_lstm_2(encoder_inputs2_inf)
encoder_outputs_inf, f_state_h_inf, f_state_c_inf = encoder_lstm(encoder_inputs2_inf)
encoder_states_inf = [state_h_1, state_c_1,state_h_2, state_c_2,f_state_h_inf, f_state_c_inf]
encoder_model = Model(encoder_inputs_inf,encoder_states_inf)
encoder_model.summary()

decoder_state_input_h_1 = Input(shape=(latent_dim,))
decoder_state_input_c_1 = Input(shape=(latent_dim,))
decoder_state_input_h_2 = Input(shape=(latent_dim,))
decoder_state_input_c_2 = Input(shape=(latent_dim,))
decoder_state_input_f_h = Input(shape=(latent_dim,))
decoder_state_input_f_c = Input(shape=(latent_dim,))
dec_states_1 = [decoder_state_input_h_1, decoder_state_input_c_1]
dec_states_2 = [decoder_state_input_h_2, decoder_state_input_c_2]
dec_states_3 = [decoder_state_input_f_h, decoder_state_input_f_c]
decoder_states_inputs = [decoder_state_input_h_1, decoder_state_input_c_1,decoder_state_input_h_2, decoder_state_input_c_2,decoder_state_input_f_h, decoder_state_input_f_c]
decoder_inputs_inf = Input(shape=(1,))
decoder_inputs2_inf = decoder_embedding(decoder_inputs_inf)
decoder_inputs2_inf, state_h_1, state_c_1 = dec_lstm_1(decoder_inputs2_inf, initial_state=dec_states_1)
decoder_inputs2_inf, state_h_2, state_c_2 = dec_lstm_2(decoder_inputs2_inf, initial_state=dec_states_2)
decoder_outputs_inf, state_f_h_inf, state_f_c_inf = decoder_lstm(decoder_inputs2_inf, initial_state=dec_states_3)
decoder_states_inf = [state_h_1, state_c_1,state_h_2, state_c_2, state_f_h_inf, state_f_c_inf]
decoder_outputs_inf = decoder_dense(decoder_outputs_inf)
decoder_model = Model(
    [decoder_inputs_inf] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states_inf)
decoder_model.summary()



# Sample code of the batch for training
i=0
batch_size = 128
X = trainX[i*batch_size : (i+1)*batch_size, : ]
Yin = trY[i*batch_size : (i+1)*batch_size, : ]
Yout = trainY[i*batch_size : (i+1)*batch_size, : ]
Yout = encode_output(Yout, ans_vocab_size)
print(X.shape)
print(X[0])
print(Yin.shape)
print(Yin[0])
print(Yout.shape)
print(Yout[0])
print(trainY[0])

batch_size = 1024
no_of_epochs = 100
training_examples = trainY.shape[0]
filepath = "model.h5"

for epochs in range(0,no_of_epochs):
    i=0
    for batches in range(0,int(training_examples/batch_size)):
        X = trainX[i*batch_size : (i+1)*batch_size, : ]
        Yin = trY[i*batch_size : (i+1)*batch_size, : ]
        Yout = trainY[i*batch_size : (i+1)*batch_size, : ]
        Yout = encode_output(Yout, ans_vocab_size)
        model.fit([X, Yin], Yout,
          batch_size=128,
          epochs=1,
          verbose = 0)
        i = i+1
    if epochs % 10 == 9 or epochs == 0:
        model.save('model{}.h5'.format(epochs + 1))
        encoder_model.save('encmodel{}.h5'.format(epochs + 1))
        decoder_model.save('decmodel{}.h5'.format(epochs + 1))

# testing 
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    target_seq = [start_token_index] #start token
    stop_condition = False
    decoded_sentence = list()
    while not stop_condition:
        output_tokens, state_h_1, state_c_1,state_h_2, state_c_2, state_f_h_inf, state_f_c_inf = decoder_model.predict(
            [target_seq] + states_value)
        
        #print(output_tokens)
        integers = [argmax(vector) for vector in output_tokens[0]]
        #print(integers)
        target_seq = integers
        sampled_char = ''
        for j in integers:
            word = word_for_id(j, ans_tokenizer)
            if word is 'EOS' or word is None:
                sampled_char = word
                break
            decoded_sentence.append(word)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == 'EOS' or sampled_char == None or len(decoded_sentence) > ans_length:
            stop_condition = True

        # Update states
        states_value = [state_h_1, state_c_1,state_h_2, state_c_2, state_f_h_inf, state_f_c_inf]

    return ' '.join(decoded_sentence)
    
print('prediction on train')
for i in range(0,10):
    temp = trainX[i].reshape((1,trainX[i].shape[0]))
    translation = decode_sequence(temp)
    raw_target, raw_src = tokenized_answers[i], tokenized_questions[i]
    src = list()
    target = list()
    for index in raw_src:
        word = question_id2w[index]
        src.append(word)
    for index in raw_target:
        word = answer_id2w[index]
        target.append(word)
    source = ' '.join(src)
    tar = ' '.join(target)
    print('src=[%s], target=[%s], predicted=[%s]' % (source, tar, translation))
