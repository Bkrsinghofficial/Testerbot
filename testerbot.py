#IMPORTING LIBRARIES
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import json 
import nltk
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM , Dense, Flatten
from keras.models import Model
import matplotlib.pyplot as plt


#importing the dataset
with open('content.json') as content:
    data1 = json.load(content)
#getting all the data to lists
tags = []
inputs = []
responses={}
for intent in data1['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

#converting to dataframe
data = pd.DataFrame({"inputs":inputs,
                        "tags":tags,
                        }) 

#printing data
data
data= data.sample(frac=1)
#removing puntuations
import string
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))
data
#tokenize the data
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
#applying padding
from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)
#encoding outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

input_shape = x_train.shape[1]
print(input_shape)

#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words :", vocabulary)
output_length = le.classes_.shape[0]
print("output length: ",output_length)
#creating a model

i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model = Model(i,x)

#compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

#training the model
train = model.fit(x_train,y_train,epochs=200)

#plotting model accuracy
plt.plot(train.history['accuracy'],label='training set accuracy')
plt.plot(train.history['loss'],label='training set loss')
plt.legend()

#chatting
import random

while True:
    text_p = []
    prediction_input = input('You:')

    #removing punctuation and converting into lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    text_p.append(prediction_input)

    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(text_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],input_shape)

    #getting output from model 
    output = model.predict(prediction_input)
    output = output.argmax()

    #finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    print("testerrbot : ",random.choice(responses[response_tag]))
    if response_tag == "goodbye":
     break