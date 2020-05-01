from lda import TopicModeling
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
##import winsound


import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import re
import pyLDAvis
import pyLDAvis.gensim

class GroupedSentence:
    def __init__(self, numWords, coherenceList, bestTopicNum, grouping):
        self.numWords = numWords
        self.coherenceList = coherenceList
        self.bestTopicNum = bestTopicNum
        self.bestGrouping = grouping

def testSingleHyperParameter(numParagraph, alpha, eta):
    lda_modeling = TopicModeling()

    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    
    startTopicNum = 2
    endTopicNum = 6

    output = []

    groupSentObjList = []
    tempCoherence = []
    wordCount = 0
    bestTopicNum = 2
    maxCoherence = -9999
    bestGrouping = ""

    for paragraph in paragraphs[:numParagraph]:
        for j in range(startTopicNum, endTopicNum):
            lda_modeling.setTopicNum(j)
            groupedSentence = lda_modeling.groupSentence(paragraph, alpha, eta)
            c, wordCount = lda_modeling.coherenceScore()
            tempCoherence.append(c)
            if c > maxCoherence:
                bestTopicNum = j
                bestGrouping = groupedSentence
                maxCoherence = c

        gs = GroupedSentence(wordCount, tempCoherence, bestTopicNum, bestGrouping)
        print(bestGrouping)
        groupSentObjList.append(gs)
        maxCoherence = -9999
    

    output.append("Eta=" + str(eta).replace('.', '_') + "\n")
    output += [str(x.bestGrouping) + "\n\n" for x in groupSentObjList]
    output.append("--------------------------------------------------------------\n")

    groupSentObjList.sort(key=lambda x: x.numWords)

    wordRange = []
    averageCoherence = []
    wordBound = 10
    tempGSObj = []

    index = 0

    while index < len(groupSentObjList):
        gs = groupSentObjList[index]
        if gs.numWords < wordBound and gs.numWords >= wordBound - 10:
            tempGSObj.append(gs)
            index += 1
        elif len(tempGSObj) != 0:
            tempAvgCoherence = []
            for i in range(endTopicNum - startTopicNum):
                avg = np.mean([x.coherenceList[i] for x in tempGSObj])
                tempAvgCoherence.append(avg)
            averageCoherence.append(tempAvgCoherence)
            wordRange.append(str(wordBound - 10) + " - " + str(wordBound - 1))
            wordBound += 10
            tempGSObj = []
        else:
            wordBound += 10

    if len(tempGSObj) != 0:
        tempAvgCoherence = []
        for i in range(endTopicNum - startTopicNum):
            avg = np.mean([x.coherenceList[i] for x in tempGSObj])
            tempAvgCoherence.append(avg)
            
        averageCoherence.append(tempAvgCoherence)
        wordRange.append(str(wordBound - 10) + " - " + str(wordBound - 1))

    ##print(wordRange)
    ##for i in range(len(averageCoherence)):
        ##print(averageCoherence[i])

    ##print(alpha, ': ', eta)

    plt.title("Average Coherence Score vs Number of Topics: Alpha=" + str(alpha) + " Eta=" + str(eta))

    for i in range(len(wordRange)):
        plt.plot([2, 3, 4, 5], averageCoherence[i], label=wordRange[i])
    
    plt.legend(title="Word Count")
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.savefig('HyperparameterTestingGraphs/' + 'coherenceGraph-Alpha' + str(alpha).replace('.', '_') + '.png')
    plt.clf()
    
    plt.scatter([x.numWords for x in groupSentObjList], [x.bestTopicNum for x in groupSentObjList])
    plt.title("Best Topic Number vs Word Count: Alpha=" + str(alpha) + " Eta=" + str(eta))
    plt.xlabel('Word Count')
    plt.ylabel('Best Topic Number', va='center')
    plt.savefig('BestTopicNum/' + 'bestTopic-Alpha' + str(alpha).replace('.', '_') + '.png')

    fileName = 'resultBestGrouping-' + 'Alpha' + str(alpha).replace('.', '_') + '.txt'
    path = os.path.dirname(__file__)
    folder = "BestGroupingTexts/" + fileName

    with open(os.path.join(path, folder), 'w') as result:
        result.writelines(output)

# -------------------------------------------------------------------------------------------------------

def testHyperParameters(numParagraph, alpha, etaValues):
    lda_modeling = TopicModeling()

    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    
    startTopicNum = 2
    endTopicNum = 6

    fig, a = plt.subplots(nrows=2, ncols=len(etaValues)//2, sharex=True, sharey=True, figsize=(24, 5))
    fig.suptitle("Average Coherence Score vs Number of Topics: Alpha=" + str(alpha))

    fig2, a2 = plt.subplots(nrows=2, ncols=len(etaValues)//2, sharex=True, sharey=True, figsize=(24, 5))
    fig2.suptitle("Best Topic Number vs Word Count: Alpha=" + str(alpha))

    output = []

    for eta in etaValues:
        groupSentObjList = []
        tempCoherence = []
        wordCount = 0
        bestTopicNum = 2
        maxCoherence = -9999
        bestGrouping = ""
        print(len(paragraphs))
        for paragraph in paragraphs[:numParagraph]:
            for j in range(startTopicNum, endTopicNum):
                lda_modeling.setTopicNum(j)
                groupedSentence = lda_modeling.groupSentence(paragraph, alpha, eta)
                c, wordCount = lda_modeling.coherenceScore()
                tempCoherence.append(c)
                if c > maxCoherence:
                    bestTopicNum = j
                    bestGrouping = groupedSentence
                    maxCoherence = c

            gs = GroupedSentence(wordCount, tempCoherence, bestTopicNum, bestGrouping)
            ##print(bestGrouping)
            groupSentObjList.append(gs)
            maxCoherence = -9999

        output.append("Eta=" + str(eta).replace('.', '_') + "\n")
        output += [str(x.bestGrouping) + "\n\n" for x in groupSentObjList]
        output.append("--------------------------------------------------------------\n")

        groupSentObjList.sort(key=lambda x: x.numWords)

        wordRange = []
        averageCoherence = []
        wordBound = 10
        tempGSObj = []

        index = 0

        while index < len(groupSentObjList):
            gs = groupSentObjList[index]
            if gs.numWords < wordBound and gs.numWords >= wordBound - 10:
                tempGSObj.append(gs)
                index += 1
            elif len(tempGSObj) != 0:
                tempAvgCoherence = []
                for i in range(endTopicNum - startTopicNum):
                    avg = np.mean([x.coherenceList[i] for x in tempGSObj])
                    tempAvgCoherence.append(avg)
                averageCoherence.append(tempAvgCoherence)
                wordRange.append(str(wordBound - 10) + " - " + str(wordBound - 1))
                wordBound += 10
                tempGSObj = []
            else:
                wordBound += 10

        if len(tempGSObj) != 0:
            tempAvgCoherence = []
            for i in range(endTopicNum - startTopicNum):
                avg = np.mean([x.coherenceList[i] for x in tempGSObj])
                tempAvgCoherence.append(avg)
            averageCoherence.append(tempAvgCoherence)
            wordRange.append(str(wordBound - 10) + " - " + str(wordBound - 1))

        ##print(wordRange)
        ##print(averageCoherence)

        etaIndex = etaValues.index(eta)
        row = etaIndex//(len(etaValues)//2)
        col = etaIndex % (len(etaValues)//2)
        ##print(alpha, ': ', etaIndex)

        a[row][col].set_title('Eta=' + str(eta))
        a2[row][col].set_title('Eta=' + str(eta))

        for i in range(len(wordRange)):
            a[row][col].plot([2, 3, 4, 5], averageCoherence[i], label=wordRange[i])
            a[row][col].legend(title="Word Count")
        
        a2[row][col].scatter([x.numWords for x in groupSentObjList], [x.bestTopicNum for x in groupSentObjList])
        
    fig.text(0.5, 0.04, 'Number of Topics', ha='center')
    fig.text(0.04, 0.5, 'Coherence Score', va='center', rotation='vertical')
    fig.subplots_adjust(left=0.07, bottom=0.13, top=0.89)
    fig.savefig('HyperparameterTestingGraphs/' + 'coherenceGraph-Alpha' + str(alpha).replace('.', '_') + '.png')
    
    fig2.text(0.5, 0.04, 'Word Count', ha='center')
    fig2.text(0.04, 0.5, 'Best Topic Number', va='center', rotation='vertical')
    fig2.subplots_adjust(left=0.07, bottom=0.13, top=0.89)
    fig2.savefig('BestTopicNum/' + 'bestTopic-Alpha' + str(alpha).replace('.', '_') + '.png')

    fileName = 'resultBestGrouping-' + 'Alpha' + str(alpha).replace('.', '_') + '.txt'
    path = os.path.dirname(__file__)
    folder = "BestGroupingTexts/" + fileName

    with open(os.path.join(path, folder), 'w') as result:
        result.writelines(output)


def main():
    alphaValues = ['symmetric', 'asymmetric', 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    etaValues=[None, 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    # for alpha in alphaValues[1:2]:
    #     testHyperParameters(8, alpha, etaValues)

    testSingleHyperParameter(5, alphaValues[0], etaValues[0])


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    ##print(end - start)
    ##winsound.MessageBeep(-1)
    # fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 5), sharex=True, sharey=True)
    # x = [2, 3, 4, 5]
    # eat7label = ['40 - 49', '70 - 79', '80 - 89', '110 - 119', '120 - 129']
    # eta7y = [[-0.13671589711902998, -0.13671589711902998, -0.13671589711902998, -0.13671589711902998], [-0.13671589711902998, -0.13671589711902998, -0.13671589711902998, -0.13671589711902998], [-0.13671589711902998, -0.13671589711902998, -0.13671589711902998, -0.13671589711902998], [-0.13671589711902998, -0.13671589711902998, -0.13671589711902998, -0.13671589711902998], [-0.13671589711902998, -0.13671589711902998, -0.13671589711902998, -0.13671589711902998]]

    # index = 0

    # for i in ax:
    #     for j in i:
    #         j.plot(x, eta7y[index], label=eat7label[index])
    #         j.set_title('hi')
    #         index+=1
    #     index = 0

    # fig.suptitle("Help")
    # fig.text(0.5, 0.04, 'Number of Topics', ha='center')
    # fig.text(0.04, 0.5, 'Coherence Score', va='center', rotation='vertical')
    # fig.subplots_adjust(left=0.07, bottom=0.13, top=0.89)
    # plt.show()