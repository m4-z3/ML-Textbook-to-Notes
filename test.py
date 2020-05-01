from lda import TopicModeling
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time
import winsound
import csv

class GroupedSentence:
    def __init__(self, paragraphIndex, numWords, coherenceList, bestTopicNum, grouping):
        self.paragraphIndex = paragraphIndex
        self.numWords = numWords
        self.coherenceList = coherenceList
        self.bestTopicNum = bestTopicNum
        self.bestGrouping = grouping

def logData(eta, paragraphNum, numWords, bestTopicNum, CoherenceScores):
    with open('BestTopicNum_Coherence-' + str(eta).replace('.', '_') + '.csv', 'a', newline='') as csvfile:
        logger = csv.writer(csvfile)
        logger.writerow([eta, paragraphNum, numWords, bestTopicNum] + CoherenceScores)


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
        print(tempCoherence)
        groupSentObjList.append(gs)
        maxCoherence = -9999
        tempCoherence = []

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

    print(wordRange)
    print(averageCoherence)

    print(alpha, ': ', eta)

    plt.title("Average Coherence Score vs Number of Topics")

    for i in range(len(wordRange)):
        plt.plot([2, 3, 4, 5], averageCoherence[i], label=wordRange[i])
    
    plt.legend(title="Word Count")
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.savefig('HyperparameterTestingGraphs/' + 'coherenceGraph-Alpha' + str(alpha).replace('.', '_') + + '.png')
    plt.clf()
    
    plt.scatter([x.numWords for x in groupSentObjList], [x.bestTopicNum for x in groupSentObjList])
    plt.title("Best Topic Number vs Word Count")
    plt.xlabel('Word Count')
    plt.ylabel('Best Topic Number', va='center')
    plt.savefig('BestTopicNum/' + 'bestTopic-Alpha' + str(alpha).replace('.', '_') + '.png')

    fileName = 'resultBestGrouping-' + 'Alpha' + str(alpha).replace('.', '_') + '.txt'
    path = os.path.dirname(__file__)
    folder = "BestGroupingTexts/" + fileName

    with open(os.path.join(path, folder), 'w') as result:
        result.writelines(output)

# -------------------------------------------------------------------------------------------------------

def testHyperParameters(alpha, etaValues, startParagraph=0, endParagraph=0):
    lda_modeling = TopicModeling()

    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    
    startTopicNum = 2
    endTopicNum = 6

    rowSize = 2
    colSize = 4

    fig, a = plt.subplots(nrows=rowSize, ncols=colSize, sharex=True, sharey=True, figsize=(24, 5))
    fig.suptitle("Average Coherence Score vs Number of Topics: Alpha=" + str(alpha))

    fig2, a2 = plt.subplots(nrows=rowSize, ncols=colSize, sharex=True, sharey=True, figsize=(24, 5))
    fig2.suptitle("Best Topic Number vs Word Count: Alpha=" + str(alpha))

    output = []

    for eta in etaValues:
        groupSentObjList = []
        tempCoherence = []
        wordCount = 0
        bestTopicNum = 2
        maxCoherence = -9999
        bestGrouping = ""

        paraIndex = startParagraph if startParagraph != 0 else 1
        start = startParagraph - 1 if startParagraph != 0 else 0
        end = endParagraph if endParagraph != 0 else len(paragraphs)
        for paragraph in paragraphs[start:end]:
            for j in range(startTopicNum, endTopicNum):
                lda_modeling.setTopicNum(j)
                groupedSentence = lda_modeling.groupSentence(paragraph, alpha, eta)
                c, wordCount = lda_modeling.coherenceScore()
                tempCoherence.append(c)
                if c > maxCoherence:
                    bestTopicNum = j
                    bestGrouping = groupedSentence
                    maxCoherence = c

            gs = GroupedSentence(paraIndex, wordCount, tempCoherence, bestTopicNum, bestGrouping)
            print(bestGrouping)
            print(tempCoherence)
            logData(eta, paraIndex, wordCount, bestTopicNum, tempCoherence)
            groupSentObjList.append(gs)
            maxCoherence = -9999
            tempCoherence = []
            paraIndex += 1


        output.append("Eta=" + str(eta).replace('.', '_') + "\n")
        output += [str(x.paragraphIndex) + ": " + str(x.bestGrouping) + "\n\n" for x in groupSentObjList]
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

        print(wordRange)
        print(averageCoherence)

        etaIndex = etaValues.index(eta)
        row = etaIndex//(len(etaValues)//2)
        col = etaIndex % (len(etaValues)//2)
        print(alpha, ': ', etaIndex)

        a[row][col].set_title('Eta=' + str(eta))
        a2[row][col].set_title('Eta=' + str(eta))

        for i in range(len(wordRange)):
            a[row][col].plot([2, 3, 4, 5], averageCoherence[i], label=wordRange[i])
        
        a2[row][col].scatter([x.numWords for x in groupSentObjList], [x.bestTopicNum for x in groupSentObjList])
        
    handles, labels = a[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title="Word Count")
    fig.text(0.5, 0.04, 'Number of Topics', ha='center')
    fig.text(0.04, 0.5, 'Coherence Score', va='center', rotation='vertical')
    start, end = a[0][0].get_ylim()
    plt.setp(a, xticks=[2, 3, 4, 5], yticks=np.arange(np.floor(start * 10) / 10, np.ceil(end * 10) / 10, 0.05))
    fig.subplots_adjust(left=0.07, bottom=0.13, top=0.89, wspace=0.04)
    fig.savefig('HyperparameterTestingGraphs-Asymmetric/' + 'coherenceGraph-Alpha' + str(alpha).replace('.', '_') + '.png')
    
    fig2.text(0.5, 0.04, 'Word Count', ha='center')
    fig2.text(0.04, 0.5, 'Best Topic Number', va='center', rotation='vertical')
    fig2.subplots_adjust(left=0.07, bottom=0.13, top=0.89, wspace=0.04)
    fig2.savefig('BestTopicNum-Asymmetric/' + 'bestTopic-Alpha' + str(alpha).replace('.', '_') + '.png')

    fileName = 'resultBestGrouping-' + 'Alpha' + str(alpha).replace('.', '_') + '.txt'
    path = os.path.dirname(__file__)
    folder = "BestGroupingTexts-Asymmetric/" + fileName

    with open(os.path.join(path, folder), 'w') as result:
        result.writelines(output)


def createMultiGraphs(alpha, etaValues):
    rowSize = 4
    colSize = 2

    startTopicNum = 2
    endTopicNum = 6

    fig, a = plt.subplots(nrows=rowSize, ncols=colSize, sharex=True, sharey=True, figsize=(8, 7))
    fig.suptitle("Average Coherence Score vs Number of Topics: Alpha=" + str(alpha))

    fig2, a2 = plt.subplots(nrows=rowSize, ncols=colSize, sharex=True, sharey=True, figsize=(8, 7))
    fig2.suptitle("Best Topic Number vs Word Count: Alpha=" + str(alpha))

    for eta in etaValues:
        groupSentObjList = []

        with open('BestTopicNum_Coherence-' + str(eta).replace('.', '_') + '.csv', 'r') as csvFile:
            readCSV = csv.reader(csvFile, delimiter=',')
            for row in readCSV:
                coherenceList = [float(x) for x in row[4:]]
                gs = GroupedSentence(int(row[1]), int(row[2]), coherenceList, int(row[3]), [])
                groupSentObjList.append(gs)

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

        etaIndex = etaValues.index(eta)
        row = etaIndex//(colSize)
        col = etaIndex % (colSize)
        print(alpha, ': ', etaIndex)

        a[row][col].set_title('Eta=' + str(eta))
        a2[row][col].set_title('Eta=' + str(eta))

        color=iter(plt.cm.rainbow(np.linspace(0,1,len(wordRange))))

        for i in range(len(wordRange)):
            c = next(color)
            a[row][col].plot([2, 3, 4, 5], averageCoherence[i], label=wordRange[i], c=c)
        
        a2[row][col].scatter([x.numWords for x in groupSentObjList], [x.bestTopicNum for x in groupSentObjList])

        a2[row][col].text(140, 4.7, 'Avg=' + str(round(np.mean([x.bestTopicNum for x in groupSentObjList]), 3)))


    handles, labels = a[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title="Word Count")
    fig.text(0.5, 0.04, 'Number of Topics', ha='center')
    fig.text(0.04, 0.5, 'Coherence Score', va='center', rotation='vertical')
    start, end = a[0][0].get_ylim()
    plt.setp(a, xticks=[2, 3, 4, 5], yticks=np.arange(np.floor(start * 10) / 10, np.ceil(end * 10) / 10, 0.05))
    fig.subplots_adjust(left=0.14, right=0.83, bottom=0.13, top=0.89, wspace=0.04, hspace=0.28)
    fig.savefig('Phase2-104Paragraph/HyperparameterTestingGraphs-Asymmetric/' + 'coherenceGraph-Alpha' + str(alpha).replace('.', '_') + '.png')
    
    fig2.text(0.5, 0.04, 'Word Count', ha='center')
    fig2.text(0.04, 0.5, 'Best Topic Number', va='center', rotation='vertical')
    fig2.subplots_adjust(left=0.10, bottom=0.10, top=0.89, wspace=0.04, hspace=0.25)
    fig2.savefig('Phase2-104Paragraph/BestTopicNum-Asymmetric/' + 'bestTopic-Alpha' + str(alpha).replace('.', '_') + '.png')


def main():
    alphaValues = ['symmetric', 'asymmetric', 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    etaValues=[None, 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

    # testHyperParameters('asymmetric', etaValues, startParagraph=91)

    # testSingleHyperParameter(5, alphaValues[1], etaValues[4])

    createMultiGraphs('asymmetric', etaValues)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
    winsound.MessageBeep(-1)
    # fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(24, 5), sharex=True, sharey=True)
    # x = [2, 3, 4, 5]
    # eat7label = ['40 - 49', '70 - 79', '80 - 89', '110 - 119', '120 - 129']
    # eta7y = [[-0.11, -0.12, -0.13, -0.14], [-0.11, -0.13, -0.14, -0.17], [-0.12, -0.17, -0.16, -0.12], [-0.15, -0.18, -0.11, -0.13], [-0.13, -0.13, -0.12, -0.18]]

    # index = 0

    # for i in ax:
    #     for j in i:
    #         j.plot(x, eta7y[index], label=eat7label[index])
    #         j.set_title('hi')
    #         index+=1
    #     index = 0

    # handles, labels = ax[0][0].get_legend_handles_labels()
    # fig.suptitle("Help")
    # start, end = ax[0][0].get_ylim()
    # print(start, end)
    # plt.setp(ax, xticks=[2, 3, 4, 5], yticks=np.arange(-0.2, -0.1, 0.05))
    # fig.legend(handles, labels, loc="upper right")
    # fig.text(0.5, 0.04, 'Number of Topics', ha='center')
    # fig.text(0.04, 0.5, 'Coherence Score', va='center', rotation='vertical')
    # fig.subplots_adjust(left=0.1, bottom=0.13, top=0.89, wspace=0.04)
    # plt.show()