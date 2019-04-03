import pandas as pd
import numpy  as np 
import copy
import functools

class puzzleNode():
    def __init__(self):
        self.assignVarIndex =  None
        self.assignWrdIndex =  None 
        self.domain         =  [[1],[2],[4,5,6]]
def compareDomain(Node1, Node2):
    firstLen  = len(Node1.domain[Node1.assignVarIndex])
    secondLen = len(Node2.domain[Node2.assignVarIndex])
    print(firstLen)
    print(secondLen)
    if firstLen < secondLen:
        print(1)
        print("----")
        return 1
    elif firstLen > secondLen:
        print(-1)
        print("----")
        return -1
    else:
        print(0)
        print("----")
        return 0

def MRV(childNode):
    childNode = sorted(childNode,key=functools.cmp_to_key(compareDomain))
    return childNode
def main():
    first  = puzzleNode()
    second = puzzleNode()
    third  = puzzleNode()
    first.assignVarIndex  = 1
    second.assignVarIndex = 1
    third.assignVarIndex  = 2
    childNode = []
    childNode.append(first)
    childNode.append(second)
    childNode.append(third)
    print(childNode)
    childNode = MRV(childNode)
    print(childNode)
if __name__ == '__main__':
    main()
    