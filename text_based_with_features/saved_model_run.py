import keras
from embedding import padding, read_my_file_questions, read_my_file_answers,read_my_file,to_npquest,to_npanswer,flatten_feats,create_tuple_quest_answer
from keras.models import load_model
from calculate_wups import *
import json
import numpy as np
def read_features(filename):
    with open(filename, 'r') as f:
        feat = json.load(f)
        return feat
def answers_to_file(targetfile,answers):
        f2 = open(targetfile, "w")
        for line in answers:
            f2.write(line)
            f2.write("\n")
def answer_to_file(linenb, file, targetfile):
    f1 = open(file, "r")
    f2 = open(targetfile, "w")
    result = []
    for line in f1:
        if linenb == 0:
            linenb += 1
        else:
            linenb = 0
            line = line.strip()
            linelist = line.replace(",", "").split()
            result.append(linelist)
    for line in result:
        f2.write(line[0])
        for word in line[1:]:
            f2.write("," + word)
        f2.write("\n")

def quest_answer_to_file(linenb, file, targetfile):
    f1 = open(file, "r")
    f2 = open(targetfile, "w")
    result = []
    for line in f1:
        if linenb == 0:
            linenb += 1
        else:
            linenb = 0
            line = line.strip()
            linelist = line.replace(",", "").split()
            result.append(linelist)
    for line in result:
        f2.write(line[0])
        for word in line[1:]:
            f2.write(" " + word)
        f2.write("\n")

def questions_to_file(targetfile, questions):
    f2 = open(targetfile, "w")
    for line in questions:
        for word in line:
            if word!="PADDING":
                f2.write(word)
                f2.write(" ")
        f2.write("\n")

hot_to_word_dict,word_to_hot_dict= read_my_file(0)
hot_to_word_dict_answ,word_to_hot_dict_answ= read_my_file(1)
train_questions= read_my_file_questions("qa.894.raw.train.txt")
train_answers=read_my_file_answers("qa.894.raw.train.txt")
n_in=len(train_questions)
vocnbq=len(word_to_hot_dict)
vocnba=len(word_to_hot_dict_answ)
tupList=create_tuple_quest_answer(train_questions,train_answers)
maxlenanswer= max([len(seq) for seq in train_answers])
embeddings_index = {}

feat=read_features("img_features.json")

def calculate_questionacc(fpred,fraw):
    f1=open(fpred)
    f2=open(fraw)
    allquestionstruth=[]
    allquestionspred=[]
    for line in f2:
        linetruth=[]
        line=line.strip()
        line=line.split()
        for word in line:
            linetruth.append(word)
        allquestionstruth.append(linetruth)
    for line in f1:
        linetruth=[]
        line = line.strip()
        line = line.split()
        for word in line:
            linetruth.append(word)
        allquestionspred.append(linetruth)
    rightguess=0
    total=0
    for questionline in range(len(allquestionstruth)):
        currline=allquestionstruth[questionline]
        total+=len(currline)
        for i in range(min(len(currline),len(allquestionspred[questionline]))):
            if currline[i]==allquestionspred[questionline][i]:
                rightguess+=1
    return rightguess/total*100
def calc_answeracc(fpred,fraw):
    f1 = open(fpred)
    f2 = open(fraw)
    allanswerstruth = []
    allanswerspred = []
    for line in f2:
        linetruth = []
        line = line.strip()
        line=line.replace(","," ")
        line = line.split()
        for word in line:
            linetruth.append(word)
        allanswerstruth.append(linetruth)
    for line in f1:
        linetruth = []
        line = line.strip()
        line = line.split()
        for word in line:
            linetruth.append(word)
        allanswerspred.append(linetruth)
    rightguess=0
    total=(len(allanswerstruth))
    for answerline in range(len(allanswerstruth)):
        if allanswerspred[answerline][0] in allanswerstruth[answerline]:
            rightguess+=1
    return rightguess/total*100



import sys
if __name__ == '__main__':
    model = load_model(sys.argv[2])
    answerwords=[]
    questionwords=[]
    test_questions= read_my_file_questions(sys.argv[1])
    test_answers = read_my_file_answers(sys.argv[1])
    nptestquest=to_npquest(test_questions, 31, word_to_hot_dict)
    npanswertest=to_npanswer(test_answers,word_to_hot_dict_answ,vocnba)
    def generator_test(batch_size):
        nb_batch = 0
        while True:
            if (nb_batch+1) * batch_size >= len(test_questions):
                this_batch = test_questions[nb_batch * batch_size:len(test_questions)]
                this_answers = npanswertest[nb_batch * batch_size:len(test_questions)]
                new_feat = np.array([np.array(feat[line[-2]]) for line in this_batch])
                new_feat = flatten_feats(new_feat)
                this_hots = nptestquest[nb_batch * batch_size:len(test_questions)]
                this_hots = np.array([[np.argmax(word) for word in line] for line in this_hots])
                yield [this_hots, new_feat]
                break
            this_batch = test_questions[nb_batch * batch_size:(1 + nb_batch) * batch_size]
            this_answers = npanswertest[nb_batch * batch_size:(1 + nb_batch) * batch_size]
            new_feat = np.array([np.array(feat[line[-2]]) for line in this_batch])
            new_feat = flatten_feats(new_feat)
            this_hots = nptestquest[nb_batch * batch_size:(1 + nb_batch) * batch_size]
            this_hots = np.array([[np.argmax(word) for word in line] for line in this_hots])
            nb_batch += 1
            yield [this_hots, new_feat]
        yield "end"
    newgen=generator_test(31)
    while True:
        newbatch=next(newgen)
        if newbatch=="end":
            break
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

    quest_answer_to_file(1,sys.argv[1],"mysavedquest.txt")
    answers_to_file("predictedanswers1.txt",answerwords)
    questions_to_file("predictedquestions1.txt",questionwords)
    answer_to_file(0,sys.argv[1],"mysavedansw.txt")
    # folders
    gt_filepath="predictedanswers1.txt"
    pred_filepath="mysavedansw.txt"

    input_gt=file2list(gt_filepath)
    input_pred=file2list(pred_filepath)

    thresh=0.9
    if thresh == -1:
        our_element_membership=dirac_measure
    else:
        our_element_membership=lambda x,y: wup_measure(x,y,thresh)

    our_set_membership=\
            lambda x,A: fuzzy_set_membership_measure(x,A,our_element_membership)

    if thresh == -1:
        print ('standard Accuracy is used')
    else:
        print ('soft WUPS at %1.2f is used' % thresh)
    score_list=[score_it(items2list(ta),items2list(pa),our_set_membership)
            for (ta,pa) in  zip(input_gt,input_pred)]
    print ('computing the final score')
    #final_score=sum(map(lambda x:float(x)/float(len(score_list)),score_list))
    final_score=float(sum(score_list))/float(len(score_list))

    # filtering to obtain the results
    #print 'full score:', score_list
    print ('exact final score:', final_score)
    print ('final score is %2.2f%%' % (final_score * 100.0))
    print(str(calculate_questionacc("predictedquestions1.txt","mysavedquest.txt"))+"% question right")
    print(str(calc_answeracc("predictedanswers1.txt","mysavedansw.txt"))+"% answer in truthanswers")



