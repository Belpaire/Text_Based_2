# lstm autoencoder recreate sequence
import numpy as np
import keras
from random import shuffle
import json
import tensorflow as tf
from functools import reduce
from keras.layers import concatenate
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import GRU
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import MaxPooling2D,MaxPooling1D
from keras.utils import plot_model
from keras.layers import add
import matplotlib.pyplot as plt
import sklearn as sk
import random
from keras.layers import Lambda
from keras.layers.core import Reshape
from keras import backend as K
#blacklist=["in","the","is","are","at","of"]
blacklist=[]

def read_features():
    with open('img_features.json', 'r') as f:
        feat = json.load(f)
        return feat


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
            linelist=list(filter(lambda x : x not in blacklist  ,linelist))
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
def flatten_feats(feats):
    newlist=[]
    for line in feats:
        newline=[]
        for x in line:
                flatline=x
                for z in flatline:
                    deepest=z
                    newline.append(z)
        newlist.append(newline)
    newlist=np.array(newlist)
    newlist=[np.reshape(line,(196,512))for line in newlist]
    return np.array(newlist)
def create_tuple_quest_answer(questions,answers):
    tupleList=[]
    for i in range(len(questions)):
        newtup=(questions[i],answers[i])
        tupleList.append(newtup)
    return tupleList
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
tupList=create_tuple_quest_answer(train_questions,train_answers)
maxlenquest= max([len(seq) for seq in train_questions+test_questions])
maxlenanswer= max([len(seq) for seq in train_answers+test_answers])
npquestions=to_npquest(train_questions,maxlenquest,word_to_hot_dict)
nptestquest=to_npquest(test_questions,maxlenquest,word_to_hot_dict)
npanswertrain=to_npanswer(train_answers,word_to_hot_dict_answ,vocnba)
npanswertest=to_npanswer(test_answers,word_to_hot_dict_answ,vocnba)
class Encoder:
    @staticmethod
    def build_encoder(hiddenlayers,inputs,img_feat):
        img_feat=Flatten() (img_feat)
        img_feat=Dense(5,activation="relu") (img_feat)
        img_feat=RepeatVector(maxlenquest) (img_feat)
        encoder1= GRU(hiddenlayers, activation='relu')  (inputs)
        encoder2=RepeatVector(maxlenquest) (encoder1)
        encoder2=concatenate([encoder2,img_feat])
        return encoder2

class Decoder:
    @staticmethod
    def build_decoder(encoder,hiddenlayers):
        decoder= GRU(hiddenlayers,return_sequences=True, activation='relu')  (encoder)
        decoder=TimeDistributed(Dense(vocnbq))(decoder)
        decoderoutput= (Activation("softmax",name="DecoderOutput")) (decoder)
        return decoderoutput
class Attention:
    @staticmethod
    def build_attention(u, imginput,nb,hiddenlayers):
        hi=(imginput)
        hi=Dense(512) (hi)
        hq=Dense(512) (u)
        hq=RepeatVector(196) (hq)
        added=Lambda(lambda x : x[0]+x[1],name="sumlaminputs" +str(nb))([hi,hq])
        added=Activation("tanh") (added)
        added=Dropout(0.5) (added)
        added=Dense(1)(added)
        pi=Activation("softmax") (added)
        v_attention=Lambda(lambda x: x[0]*x[1],name="addlammult"+str(nb))([pi,imginput])
        v_attention=Lambda(lambda x:K.sum(x,axis=1),name="sumlamupdate"+str(nb)) (v_attention)
        u=Lambda(lambda x : x[0]+x[1] )([v_attention,u])
        return u
class Answer:
    @staticmethod
    def build_answer(inputs,hiddenlayers):
        u=LSTM(hiddenlayers,activation="relu") (inputs)
        answer=Dense(vocnba) (u)
        answeroutput = (Activation("softmax", name="AnswerOutput"))(answer)
        return answeroutput
class NetworkModel:
    @staticmethod
    def build_model(hiddenlayers):
        inputs = Input(shape=(maxlenquest,vocnbq), name="input")
        inputs2= Input(shape=(14,14,512),name="feat")
        encoder= Encoder.build_encoder(hiddenlayers,inputs,inputs2)
        answer=Answer.build_answer(encoder,hiddenlayers)
        decoder=Decoder.build_decoder(encoder,hiddenlayers)
        model = Model(inputs=[inputs,inputs2], outputs=[answer, decoder])
        return model
model = NetworkModel.build_model(100)
model.compile(optimizer='adam',  loss={'DecoderOutput': 'categorical_crossentropy', 'AnswerOutput': 'binary_crossentropy'},
              loss_weights={'DecoderOutput': 1., 'AnswerOutput': 1.}, metrics=['accuracy'])
print(model.summary())
# define input sequence
# reshape input into [samples, timesteps, features]
feat=read_features()
# define model
npquestions=np.reshape(npquestions,(n_in, maxlenquest, vocnbq))
nptestquest=np.reshape(nptestquest,(n_in_2, maxlenquest, vocnbq))


def generator_train(batch_size,train_questions,train_answers,npquestions,npanswertrain):
    nb_internal=0
    nb_this_it=0
    input_quest=train_questions
    input_answers=train_answers
    input_npquestions=npquestions
    input_npanswertrain=npanswertrain
    while True:
        new_feat=[]
        new_hots=[]
        new_answers=[]
        while nb_this_it<batch_size:
            if nb_internal >= len(input_quest):
                nb_internal = 0
                shuffle(tupList)
                input_quest=[line[0] for line in tupList]
                input_answers = [line[1] for line in tupList]
                input_npquestions=to_npquest(input_quest,maxlenquest,word_to_hot_dict)
                input_npanswertrain=to_npanswer(input_answers,word_to_hot_dict_answ,vocnba)
                input_npquestions = np.reshape(input_npquestions, (n_in, maxlenquest, vocnbq))
            this_it=train_questions[nb_internal]
            this_answer=input_npanswertrain[nb_internal]
            this_feat=[np.array(feat[this_it[-2]])  ]
            #this_feat = flatten_feats(this_feat)
            this_hots=input_npquestions[nb_internal]
            nb_this_it+=1
            nb_internal+=1
            new_feat.append(this_feat[0])
            new_hots.append(this_hots)
            new_answers.append(this_answer)
        new_feat=np.array(new_feat)
        new_hots=np.array(new_hots)
        new_answers=np.array(new_answers)
        nb_this_it=0
        yield [new_hots, new_feat] , [new_answers,new_hots]

# fit model
import math
for i in range(150):
    batchsize=31
    epochsteps=int(math.ceil(n_in/batchsize))
    model.fit_generator(generator_train(batchsize,train_questions,train_answers,npquestions,npanswertrain),epochsteps,1)
    print("epoch",i)
    randnb=random.randint(0,n_in_2-10)
    elems=test_questions[randnb:10+randnb]
    answers=test_answers[randnb:10+randnb]
    test_feat_epoch=np.array([np.array(feat[line[-2]]) for line in elems])
    #test_feat_epoch=flatten_feats(test_feat_epoch)
    for linei in range(10):
        print(elems[linei])
        print(answers[linei])
        line=padding(elems[linei], maxlenquest)
        hots=np.array([ word_to_hot_dict[word] for word in line])
        hots=np.reshape(hots,(1,maxlenquest,vocnbq))
        (answer,question)=model.predict([hots,test_feat_epoch])
        total=[]
        allanswers=[]
        for hot in question[0]:
            intermed = [0] * vocnbq
            index = np.argmax(hot)
            intermed[index] = 1
            total.append(hot_to_word_dict[tuple(intermed)])
        intermed = [0] * vocnba
        index = np.argmax(answer)
        intermed[index] = 1
        mostlikedanswer=hot_to_word_dict_answ[tuple(intermed)]
        print(list(filter(lambda x: x!="PADDING",total)))
        print(mostlikedanswer)


def answers_to_file(targetfile,answers):
        f2 = open(targetfile, "w")
        for line in answers:
            f2.write(line)
            f2.write("\n")


def questions_to_file(targetfile, questions):
    f2 = open(targetfile, "w")
    for line in questions:
        for word in line:
            if word!="PADDING":
                f2.write(word)
                f2.write(" ")
        f2.write("\n")
def generator_test(batch_size):
    nb_batch = 0
    while True:
        if nb_batch * batch_size > len(test_questions):
            nb_batch = 0
        this_batch = test_questions[nb_batch * batch_size:(1 + nb_batch) * batch_size]
        this_answers = npanswertest[nb_batch * batch_size:(1 + nb_batch) * batch_size]
        new_feat = np.array([np.array(feat[line[-2]]) for line in this_batch])
        #new_feat = flatten_feats(new_feat)
        this_hots = nptestquest[nb_batch * batch_size:(1 + nb_batch) * batch_size]
        nb_batch += 1
        yield [this_hots, new_feat]

answerwords=[]
questionwords=[]
newgen=generator_test(31)
for i in range(len(test_questions)//31):
    newbatch=next(newgen)
    (answers,questions)=model.predict(newbatch)
    for question in questions:
        newquestion = []
        for word in question:
            intermed = [0] * vocnbq
            index = np.argmax(word)
            intermed[index] = 1
            mostlikedword = hot_to_word_dict[tuple(intermed)]
            newquestion.append(mostlikedword)
        questionwords.append(newquestion)
    for answer in answers:
        intermed = [0] * vocnba
        index = np.argmax(answer)
        intermed[index] = 1
        mostlikedanswer = hot_to_word_dict_answ[tuple(intermed)]
        answerwords.append(mostlikedanswer)


answers_to_file("predictedanswers.txt",answerwords)
questions_to_file("predictedquestions.txt",questionwords)

