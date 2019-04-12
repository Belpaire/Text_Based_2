import sys
blacklist=["in","the","is","are","at","of"]

def quest_answer_to_file(linenb,file,targetfile):
        f1 = open(file, "r")
        f2=open(targetfile, "w")
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
            for word in line:
#                if word not in blacklist and "image" not in word:
                 f2.write(word+" ")
            f2.write("\n")
if __name__ == '__main__':
    if len(sys.argv)==4:
        quest_answer_to_file(sys.argv[3],sys.argv[1]+".txt", sys.argv[2]+".txt")
    else:
        print("wrong arguments")

