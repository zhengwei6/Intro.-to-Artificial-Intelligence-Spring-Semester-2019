import pandas as pd
import numpy  as np 
import copy
import functools
class puzzleInfo():
    def __init__(self,puzzleIndex,wordGroup):
        if puzzleIndex == 0:
            self.constraintMap = [[-1,0,0,-1,-1],[0,-1,-1,-1,0],[2,-1,-1,0,2],[-1,-1,2,-1,-1],[-1,3,3,-1,-1]]
            self.variableLen   = [4,5,4,2,3]
        elif puzzleIndex == 1:
            self.constraintMap = [[-1,1,2,-1,-1,-1],[1,-1,-1,0,-1,-1],[3,-1,-1,2,0,-1],[-1,3,4,-1,-1,0],[-1,-1,6,-1,-1,2],[-1,-1,-1,4,2,-1]]
            self.variableLen   = [5,5,7,5,4,3]
        elif puzzleIndex == 2:
            self.constraintMap = [[-1,0,0,-1,-1,-1],[0,-1,-1,-1,2,0],[3,-1,-1,-1,5,3],[-1,-1,-1,-1,0,-1],[-1,2,2,1,-1,-1],[-1,4,4,-1,-1,-1]]
            self.variableLen   = [4,5,5,3,6,4]
        else:
            self.constraintMap = [[-1,0,0,-1,-1,-1],[0,-1,-1,-1,2,0],[3,-1,-1,-1,5,3],[-1,-1,-1,-1,0,-1],[-1,2,2,1,-1,-1],[-1,4,4,-1,-1,-1]]
            self.variableLen   = [4,5,5,3,6,4]
        self.wordGroup     = wordGroup
    

class puzzleNode():
    def __init__(self):
        self.assignVarIndex =  None
        self.assignWrdIndex =  None 
        self.domain         =  []

    def init_domain(self,var_len,wordGroup):
        self.alreadyAssign  =  np.empty(len(var_len))
        self.alreadyAssign.fill(-1)
        for length in var_len:
            self.domain.append(wordGroup[length].values)
        print("First node domain len:",len(self.domain))

    def print_domain(self):
        print(self.domain)

    def set_domain(self):
        # set assignVarIndex = 0 , assignWrdIndex = 500
        if self.assignVarIndex is not None :
            self.domain[self.assignVarIndex] = np.array([self.assignWrdIndex])

    def update_domain(self,constraintMap):
        varIndex = self.assignVarIndex
        wrdIndex = self.assignWrdIndex
        self.alreadyAssign[varIndex] = wrdIndex 
        assignWord = wordDataFrame.at[wrdIndex,'words']
        #print("variable index want to set: ",varIndex)
        #print("assign word: ",assignWord)
        for index,element in enumerate(self.alreadyAssign):
            if element == -1 :
                prohibitIndex = constraintMap[varIndex][index]
                if prohibitIndex == -1:
                    continue
                doProhitbitWord = assignWord[constraintMap[index][varIndex]]
                delete_list = []
                #print("effect index: ",index)
                #print("doProhitbitWord: ",doProhitbitWord)
                #print(" domain: ",self.domain[index])
                for index2,element2 in enumerate(self.domain[index]):
                    if wordDataFrame.at[element2,'words'][prohibitIndex] != doProhitbitWord:
                        delete_list.append(index2)
                #print("length of delete_list: ",len(delete_list))
                
                self.domain[index] = np.delete(self.domain[index],delete_list)
                #print("remain domain: ",self.domain[index])
                #print("--------------------------")
            else:
                continue

    def checkConsistency(self):
        for element in self.domain:
            if element.size == 0:
                return False
        solution = 1
        for index,element in enumerate(self.alreadyAssign):
            if element == -1:
                solution = 0
        
        if solution == 1:
            #print("Following is solution dictionary index:")
            ans = []
            for e in self.alreadyAssign:
                ans.append(wordDataFrame.at[int(e),'words'])
            check = 0
            for element in answer:
                check = 1
                for index2,element2 in enumerate(element):
                    if element2 != ans[index2]:
                        check = 0
                if check == 1:
                    break
            if check != 1:
                answer.append(ans)
            if len(answer) % 1000 == 0 :
                print(len(answer))
            return False
        else:
            return True
def compareDomain(Node1, Node2):
    firstLen  = len(Node1.domain[Node1.assignVarIndex])
    secondLen = len(Node2.domain[Node2.assignVarIndex])
    if firstLen < secondLen:
        return 1
    elif firstLen > secondLen:
        return -1
    else:
        return 0

def genChild(object):
    childNode = []
    for index,element in enumerate(object.alreadyAssign):
        if element == -1:
            for index2,element2 in enumerate(object.domain[index]):
                newNode = puzzleNode()
                newNode.assignVarIndex = index
                newNode.assignWrdIndex = element2
                newNode.domain = copy.copy(object.domain)
                newNode.alreadyAssign = copy.copy(object.alreadyAssign)
                childNode.append(newNode)
    #print("length of genChild: ",len(childNode))
    childNode = sorted(childNode,key=functools.cmp_to_key(compareDomain))
    return childNode
def genWordGroup(filepath):
    global wordDataFrame  
    wordDataFrame = pd.read_csv(filepath,sep = "\n",names = ['words'])
    wordDataFrame['wordLength'] = 0
    for index, row in wordDataFrame.iterrows():
        wordDataFrame.at[index,'wordLength'] = len(row['words'])
    
    re = wordDataFrame.groupby('wordLength').groups
    return re

def main():
    # init constraints map
    wordGroup     = genWordGroup('./English-words-3000.txt')
    puzzleStack   = []
    puzzleBoard   = puzzleInfo(0,wordGroup)
    firstNode     = puzzleNode()
    firstNode.init_domain(puzzleBoard.variableLen,wordGroup)
    
    puzzleStack.append(firstNode)
    
    print("Start to expand nodes")
    i = 0
    while puzzleStack:
        #print(len(puzzleStack))
        expandNode = puzzleStack[-1]
        puzzleStack.pop()
        if expandNode.assignVarIndex is not None:
            expandNode.set_domain()
            expandNode.update_domain(puzzleBoard.constraintMap)
        
        if expandNode.checkConsistency() == True:
            child = genChild(expandNode)
        else:
            continue
        for element in child:
            puzzleStack.append(element)
    print("End of the expansion")
if __name__ == '__main__':
    global answer
    answer = []
    main()
    df = pd.DataFrame(answer,columns=['0','1','2','3','4'])
    df.to_csv('./ans.csv',sep = ',')
