# lstm autoencoder recreate sequence
import numpy as np
import keras
from functools import reduce
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import GRU
from keras.utils import plot_model
import matplotlib.pyplot as plt
import sklearn as sk
import random

blacklist=["in","the","is","are","at","of"]

def padding(question,maxlen):
    result=question[:]
    while len(result)<maxlen:
            result=["PADDING"]+result
    return result
def read_my_file_answers(file):
    f=open(file, "r")
    linenb = 0
    result=[]
    for line in f:
        if linenb == 0:
            linenb+=1
        else:
            linenb=0
            line = line.strip()
            linelist= line.replace(",","").split()
            result.append(linelist)
    return result

def read_my_file_questions(file):
    f=open(file, "r")
    linenb = 0
    result=[]
    for line in f:
        if linenb == 1:
            linenb=0
        else:
            linenb=1
            line = line.strip()
            linelist= line.split()
            linelist=list(filter(lambda x : x not in blacklist ,linelist))
            result.append(linelist)
    return result
def read_hots(linenb,f,int_to_word,word_to_int):
    for line in f:
        if linenb == 1:
            linenb = 0
        else:
            linenb = 1
            line = line.strip()
            listStrings = line.replace(",","").split()
            for word in listStrings:
                if word not in blacklist :
                    if word not in word_to_int:
                        length = len(int_to_word)
                        int_to_word[length] = word
                        word_to_int[word] = length
    return int_to_word,word_to_int


def read_my_file(linenb):
    f1=open("qa.894.raw.train.txt", "r")
    f2=open("qa.894.raw.test.txt","r")
    int_to_word= dict()
    word_to_int = dict()
    word_to_int["PADDING"]=0
    int_to_word[0]="PADDING"
    int_to_word,word_to_int= read_hots(linenb, f1, int_to_word, word_to_int)
    int_to_word,word_to_int= read_hots(linenb, f2, int_to_word, word_to_int)
    one_hot_to_word=dict()
    word_to_one_hot=dict()
    vocnb=len(word_to_int)
    nb=0
    for key in word_to_int :
        onehotlst=[0]*vocnb
        onehotlst[nb]=1
        onehottup=tuple(onehotlst[:])
        word_to_one_hot[key]=onehottup
        one_hot_to_word[onehottup]=key
        nb+=1
    return one_hot_to_word,word_to_one_hot
def read_my_file_quest_label(linenb):
    f1=open("qa.894.raw.train.txt", "r")
    f2=open("qa.894.raw.test.txt","r")
    int_to_word= dict()
    word_to_int = dict()
    word_to_int["PADDING"]=0
    int_to_word[0]="PADDING"
    int_to_word,word_to_int= read_hots(linenb, f1, int_to_word, word_to_int)
    int_to_word,word_to_int= read_hots(linenb, f2, int_to_word, word_to_int)
    return int_to_word,word_to_int
def to_npquest(input,maxlen,word_to_hot_dict):
    resultfinal=[]
    for line in input:
        line = padding(line, maxlen)
        result= [word_to_hot_dict[word]  for word in line]
        resultfinal.append(np.array(result))
    resultfinal=np.array(resultfinal)
    return resultfinal
def to_npanswer(input,word_to_hot_dict_answ,vocnba):
    resultfinal = []
    for line in input:
        linenp=[0] * vocnba
        for word in line:
            hot= word_to_hot_dict_answ[word]
            index=np.argmax(hot)
            linenp[index]=1
        resultfinal.append(np.array(linenp))
    resultfinal=np.array(resultfinal)
    return resultfinal

hot_to_word_dict,word_to_hot_dict= read_my_file(0)
hot_to_word_dict_answ,word_to_hot_dict_answ= read_my_file(1)
train_questions= read_my_file_questions("qa.894.raw.train.txt")
train_answers=read_my_file_answers("qa.894.raw.train.txt")
test_questions=read_my_file_questions("qa.894.raw.test.txt")
test_answers=read_my_file_answers("qa.894.raw.test.txt")
n_in=len(train_questions)
n_in_2=len(test_questions)
vocnbq=len(word_to_hot_dict)
vocnba=len(word_to_hot_dict_answ)



maxlenquest= max([len(seq) for seq in train_questions+test_questions])
maxlenanswer= max([len(seq) for seq in train_answers+test_answers])
npquestions=to_npquest(train_questions,maxlenquest,word_to_hot_dict)
nptestquest=to_npquest(test_questions,maxlenquest,word_to_hot_dict)
npanswertrain=to_npanswer(train_answers,word_to_hot_dict_answ,vocnba)
npanswertest=to_npanswer(test_answers,word_to_hot_dict_answ,vocnba)

# define input sequence
# reshape input into [samples, timesteps, features]

# define model
npquestions=np.reshape(npquestions,(n_in, maxlenquest, vocnbq))
nptestquest=np.reshape(nptestquest,(n_in_2, maxlenquest, vocnbq))

class Encoder:
    @staticmethod
    def build_encoder(hiddenlayers,inputs):
        encoder1= LSTM(hiddenlayers, activation='relu') (inputs)
        encoder2=RepeatVector(maxlenquest) (encoder1)
        return encoder2

class Decoder:
    @staticmethod
    def build_decoder(encoder,hiddenlayers):
        decoder1= LSTM(hiddenlayers,return_sequences=True, activation='relu') (encoder)
        decoder2=TimeDistributed(Dense(vocnbq))(decoder1)
        decoderoutput= (Activation("softmax",name="DecoderOutput")) (decoder2)
        return decoderoutput
class Answer:
    @staticmethod
    def build_answer(encoder,hiddenlayers,outputname):
        answer1 = LSTM(hiddenlayers, activation='relu')(encoder)
        answer2= Dense(vocnba) (answer1)
        answeroutput= (Activation("softmax",name=outputname)) (answer2)
        return answeroutput
class NetworkModel:
    @staticmethod
    def build_model(hiddenlayers):
        inputs = Input(shape=(maxlenquest,vocnbq), name="input")
        encoder= Encoder.build_encoder(100,inputs)
        decoder=Decoder.build_decoder(encoder,100)
        answers=[]
        for x  in range(maxlenanswer):
            answers.append(Answer.build_answer(encoder, 200,"AnswerOutput"+str(x)))

        model = Model(inputs=inputs, outputs=[ decoder]+answers)
        return model
model = NetworkModel.build_model(100)
model.compile(optimizer='adam',  loss=['categorical_crossentropy']+['binary_crossentropy']*(maxlenanswer),
              loss_weights=[1.5]+[1.]*(maxlenanswer), metrics=['accuracy'])
# fit model
def add_answer(answerhot,answers):
    intermed = [0] * vocnba
    index = np.argmax(answerhot)
    intermed[index] = 1
    newanswer=hot_to_word_dict_answ[tuple(intermed)]
    if newanswer not in answers:
        answers.append(newanswer)
    return answers
for i in range(1):
    datatrain=[]
    dataval=[]
    for x in range(maxlenanswer):
        datatrain.append(npanswertrain)
        dataval.append(npanswertest)
    model.fit({'input': npquestions}, [npquestions]+ datatrain ,
              validation_data=(nptestquest, [nptestquest]+dataval),verbose=2)
    print("epoch",i)
    randnb=random.randint(0,len(test_questions)-10)
    elems=test_questions[randnb:10+randnb]
    answers=test_answers[randnb:10+randnb]
    for linei in range(10):
        print(elems[linei])
        print(answers[linei])
        line=padding(elems[linei], maxlenquest)
        hots=np.array([ word_to_hot_dict[word] for word in line])
        hots=np.reshape(hots,(1,maxlenquest,vocnbq))
        predictions=model.predict(hots)
        total=[]
        allanswers=[]
        question=predictions[0]
        for hot in question[0]:
            intermed = [0] * vocnbq
            index = np.argmax(hot)
            intermed[index] = 1
            total.append(hot_to_word_dict[tuple(intermed)])
        for answer in predictions[1:]:
            allanswers=add_answer(answer,allanswers)
        print(list(filter(lambda x: x!="PADDING",total)))
        print(list(filter(lambda x: x!="PADDING",allanswers)))


def answers_to_file(targetfile,answers):
        f2 = open(targetfile, "w")
        linenb = 0
        for line in answers:
            f2.write("".join(line))
            f2.write("\n")
answers=model.predict(nptestquest)[1:]
answerwords=[]
for nbquestion in range(n_in_2):
    line=[]
    for answerpos in range(maxlenanswer):
        line=add_answer(answers[answerpos][nbquestion],line)
    answerwords.append(line)
answers_to_file("predictedanswers.txt",answerwords)
# import os
# os.environ["PATH"] += os.pathsep + 'C:\\graphviz\\release\\bin'
# plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
# list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#

