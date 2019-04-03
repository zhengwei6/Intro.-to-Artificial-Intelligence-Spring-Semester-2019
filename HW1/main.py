import pandas as pd
import numpy  as np 
import copy
import functools
import random
import csv

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
            self.constraintMap = [[-1,0 ,-1,-1,0 ,-1,-1,-1,-1,-1,-1,-1],
                                  [0 ,-1,-1,-1,-1,-1,0 ,0 ,-1,-1,-1,-1],
                                  [-1,-1,-1,1 ,1 ,0 ,-1,-1,-1,-1,-1,-1],
                                  [-1,-1,2 ,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                  [4 ,-1,0 ,-1,-1,-1,4 ,4 ,3 ,-1,-1,-1],
                                  [-1,-1,3 ,-1,-1,-1,7 ,-1,-1,-1,-1,-1],
                                  [-1,3 ,-1,-1,3 ,2 ,-1,-1,-1,0 ,-1,-1],
                                  [-1,5 ,-1,-1,5 ,-1,-1,-1,-1,2 ,0 ,-1],
                                  [-1,-1,-1,-1,7 ,-1,-1,-1,-1,4 ,-1,-1],
                                  [-1,-1,-1,-1,-1,-1,2 ,2 ,1 ,-1,-1,-1],
                                  [-1,-1,-1,-1,-1,-1,-1,6 ,-1,-1,-1,0 ],
                                  [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1 ,-1]]
            self.variableLen   = [5,7,4,2,8,4,8,7,4,5,2,2]
    

class puzzleNode():
    def __init__(self):
        self.assignVarIndex =  None
        self.assignWrdIndex =  None 
        self.domain         =  []

    def init_domain(self,var_len,wordGroup):
        self.alreadyAssign  =  np.empty(len(var_len))
        self.alreadyAssign.fill(-1)
        print("First domain example:")
        for length in var_len:
            self.domain.append(wordGroup[length].values)
            print(length,wordDataFrame.at[wordGroup[length].values[0],'words'])

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
        for index,element in enumerate(self.alreadyAssign):
            if element == -1 :
                prohibitIndex = constraintMap[varIndex][index]
                if prohibitIndex == -1:
                    continue
                doProhitbitWord = assignWord[constraintMap[index][varIndex]]
                delete_list = []
                for index2,element2 in enumerate(self.domain[index]):
                    if wordDataFrame.at[element2,'words'][prohibitIndex] != doProhitbitWord:
                        delete_list.append(index2)
                
                self.domain[index] = np.delete(self.domain[index],delete_list)
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
            return False
        else:
            return True
def compareDomain(Node1, Node2):
    firstLen  = len(Node1.domain[Node1.assignVarIndex])
    secondLen = len(Node2.domain[Node2.assignVarIndex])
    if firstLen < secondLen :
        return 1
    elif firstLen > secondLen:
        return -1
    elif mode > 0:
        firstconstrainNum  = 0
        secondconstrainNum = 0
        for index,element in enumerate(Node1.alreadyAssign):
            if element == -1 and puzzleBoard.constraintMap[Node1.assignVarIndex][index] != -1 :
                firstconstrainNum = firstconstrainNum + 1
        for index2,element2 in enumerate(Node2.alreadyAssign):
            if element2 == -1 and puzzleBoard.constraintMap[Node2.assignVarIndex][index2] != -1 :
                secondconstrainNum = secondconstrainNum + 1
        
        if firstconstrainNum > secondconstrainNum:
            return 1
        elif firstconstrainNum < secondconstrainNum:
            return -1
        elif mode > 1:
            domain1 = copy.copy(Node1.domain)
            domain2 = copy.copy(Node2.domain)
            count1 = 0
            count2 = 0
            varIndex = Node1.assignVarIndex
            wrdIndex = Node1.assignWrdIndex
            assignWord = wordDataFrame.at[wrdIndex,'words']

            varIndex2 = Node2.assignVarIndex
            wrdIndex2 = Node2.assignWrdIndex
            assignWord2 = wordDataFrame.at[wrdIndex2,'words']

            delete_list = []
            for index,element in enumerate(Node1.alreadyAssign):
                if element == -1 and index != Node1.assignVarIndex:
                    prohibitIndex = puzzleBoard.constraintMap[varIndex][index]
                    if prohibitIndex == -1:
                        continue
                    doProhitbitWord = assignWord[puzzleBoard.constraintMap[index][varIndex]]
                    for index2,element2 in enumerate(domain1[index]):
                        if wordDataFrame.at[element2,'words'][prohibitIndex] != doProhitbitWord:
                            delete_list.append(index2)
                    domain1[index] = np.delete(domain1[index],delete_list)
                count1 = count1 + len(domain1[index])
            delete_list = []
            for index,element in enumerate(Node2.alreadyAssign):
                if element == -1 and index != Node2.assignVarIndex:
                    prohibitIndex = puzzleBoard.constraintMap[varIndex2][index]
                    if prohibitIndex == -1:
                        continue
                    doProhitbitWord = assignWord2[puzzleBoard.constraintMap[index][varIndex2]]
                    for index2,element2 in enumerate(domain2[index]):
                        if wordDataFrame.at[element2,'words'][prohibitIndex] != doProhitbitWord:
                            delete_list.append(index2)
                    domain1[index] = np.delete(domain2[index],delete_list)
                count2 = count2 + len(domain2[index])
            if count1 < count2:
                return -1
            elif count1 > count2:
                return 1  
    return 0

def genChild(parent):
    childNode = []
    for index,element in enumerate(parent.alreadyAssign):
        if element == -1:
            for index2,element2 in enumerate(parent.domain[index]):
                newNode = puzzleNode()
                newNode.assignVarIndex = index
                newNode.assignWrdIndex = element2
                newNode.domain = copy.copy(parent.domain)
                newNode.alreadyAssign = copy.copy(parent.alreadyAssign)
                childNode.append(newNode)
    #random.shuffle(childNode)
    if mode != 3 :
        childNode = sorted(childNode,key=functools.cmp_to_key(compareDomain))
        #print(childNode[-1].assignVarIndex)
    return childNode

def genWordGroup(wordDataFrame):      
    wordDataFrame['wordLength'] = 0
    for index, row in wordDataFrame.iterrows():
        wordDataFrame.at[index,'wordLength'] = len(row['words'])
    
    re = wordDataFrame.groupby('wordLength').groups
    return re

# starTest is for testing many kind of mode with heuristic
def starTest(wordGroup):
    expandNodeNum = 0
    puzzleStack = []
    if mode == 0:
        print("Heuristic: MRV")
        firstNode     = puzzleNode()
        firstNode.init_domain(puzzleBoard.variableLen,wordGroup)
        puzzleStack.append(firstNode)
        print("Start to expand nodes")
        i = 0
        while puzzleStack:
            expandNode = puzzleStack[-1]
            puzzleStack.pop()
            expandNodeNum = expandNodeNum + 1
            if expandNode.assignVarIndex is not None:
                expandNode.set_domain()
                expandNode.update_domain(puzzleBoard.constraintMap)
            
            
            if expandNode.checkConsistency() == True:
                child = genChild(expandNode)
            else:
                if len(answer) == anserLimit:
                    print("answer equal the limit: ",anserLimit)
                    break
                continue
            for index,element in enumerate(child):
                puzzleStack.append(element)
        print("End of the expansion")
        print("Number of expand node: ",expandNodeNum)
        print("-----------------------")
    elif mode == 1:
        print("Heuristic: MRV、Degree")
        firstNode     = puzzleNode()
        firstNode.init_domain(puzzleBoard.variableLen,wordGroup)
        puzzleStack.append(firstNode)
        print("Start to expand nodes")
        i = 0
        while puzzleStack:
            expandNode = puzzleStack[-1]
            puzzleStack.pop()
            expandNodeNum = expandNodeNum + 1
            if expandNode.assignVarIndex is not None:
                expandNode.set_domain()
                expandNode.update_domain(puzzleBoard.constraintMap)
                expandNodeNum = expandNodeNum + 1
    
            if expandNode.checkConsistency() == True:
                #print(456)
                child = genChild(expandNode)
            else:
                #print(123)
                if len(answer) == anserLimit:
                    break
                continue
            for index,element in enumerate(child):
                puzzleStack.append(element)
        print("End of the expansion")
        print("Number of expand node: ",expandNodeNum)
        print("-----------------------")
    elif mode == 2:
        print("Heuristic: MRV、Degree、LCV")
        firstNode     = puzzleNode()
        firstNode.init_domain(puzzleBoard.variableLen,wordGroup)
        puzzleStack.append(firstNode)
        print("Start to expand nodes")
        i = 0
        while puzzleStack:
            expandNode = puzzleStack[-1]
            puzzleStack.pop()
            expandNodeNum = expandNodeNum + 1
            if expandNode.assignVarIndex is not None:
                expandNode.set_domain()
                expandNode.update_domain(puzzleBoard.constraintMap)
                expandNodeNum = expandNodeNum + 1
    
            if expandNode.checkConsistency() == True:
                child = genChild(expandNode)
            else:
                print(expandNode.alreadyAssign)
                if len(answer) == anserLimit:
                    break
                continue
            
            for index,element in enumerate(child):
                puzzleStack.append(element)
        print("End of the expansion")
        print("Number of expand node: ",expandNodeNum)
        print("-----------------------")
    else:
        print("Heuristic: ")
        firstNode     = puzzleNode()
        firstNode.init_domain(puzzleBoard.variableLen,wordGroup)
        puzzleStack.append(firstNode)
        print("Start to expand nodes")
        i = 0
        while puzzleStack:
            expandNode = puzzleStack[-1]
            puzzleStack.pop()
            expandNodeNum = expandNodeNum + 1
            if expandNodeNum % 10000 == 0:
                print(expandNodeNum)
            if expandNode.assignVarIndex is not None:
                expandNode.set_domain()
                expandNode.update_domain(puzzleBoard.constraintMap)
                expandNodeNum = expandNodeNum + 1
    
            if expandNode.checkConsistency() == True:
                child = genChild(expandNode)
            else:
                if len(answer) == anserLimit:
                    break
                continue
            
            for index,element in enumerate(child):
                puzzleStack.append(element)
        print("End of the expansion")
        print("Number of expand node: ",expandNodeNum)
        print("-----------------------")
    return expandNodeNum
    

def main():
    global mode
    global puzzleBoard 
    global anserLimit
    global answer
    global wordDataFrame
    # init constraints map
    # setup mode、 answer 、 answerLimit 、 puzzleBoard
    
    testingSet = []
    for step in range(0,10):
        answer = []
        anserLimit = 1
        wordDataFrame = pd.read_csv('./English-words-3000.txt',sep = "\n",names = ['words'])
        wordDataFrame = wordDataFrame.sample(frac=1).reset_index(drop=True) 
        wordGroup     = genWordGroup(wordDataFrame) 
        puzzleBoard = puzzleInfo(3,wordGroup)
        tmpList = []
        for i in range(2,3):
            answer = []
            mode   = i
            print("mode: ",mode)
            path   = './ans' + str(i) + '.csv'
            expandNodeNum = starTest(wordGroup)
            tmpList.append(expandNodeNum)
            df = pd.DataFrame(answer,columns=['0','1','2','3','4','5','6','7','8','9','10','11'])
            df.to_csv(path,sep = ',')
        testingSet.append(tmpList)
    df = pd.DataFrame(testingSet,columns=['0'])
    df.to_csv('output.csv',sep=',')
    print(testingSet)  
if __name__ == '__main__':
    main()
