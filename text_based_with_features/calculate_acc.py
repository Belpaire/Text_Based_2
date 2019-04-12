def calculate_questionacc():
    f1=open("predictedquestions.txt")
    f2=open("rawquestions.txt")
    allquestionstruth=[]
    allquestionspred=[]
    for line in f2:
        linetruth=[]
        for word in line:
            linetruth.append(word)
        allquestionstruth.append(linetruth)
    for line in f1:
        linetruth=[]
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
def calc_answeracc():
    f1 = open("predictedanswers.txt")
    f2 = open("rawtestsanswers.txt")
    allanswerstruth = []
    allanswerspred = []
    for line in f2:
        linetruth = []
        for word in line:
            linetruth.append(word)
        allanswerstruth.append(linetruth)
    for line in f1:
        linetruth = []
        for word in line:
            linetruth.append(word)
        allanswerspred.append(linetruth)
    rightguess=0
    total=0
    for answerline in range(len(allanswerstruth)):
        total+=len(allanswerstruth[answerline])
        if allanswerspred[answerline][0] in allanswerstruth[answerline]:
            rightguess+=1
    return rightguess/total*100

print(calc_answeracc())
print(calculate_questionacc())