import csv
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
import matplotlib.pyplot as plt
import numpy as np
import scipy, pylab
from sklearn.cluster import KMeans
from sklearn import svm
from matplotlib import style
style.use("ggplot")
from lda import TopicModeling


def classify(data): 
    with open('data.csv', newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        header = next(readCSV)
        rows = [[row[0], int(row[1]), int(row[2])] for row in readCSV]
        ary = []
        len_sent = []
        for row in rows:
            doc = nlp(row[0])
            len_sent.append(len(row[0]))
            ary.append(len(doc.ents))

    x = list(zip(len_sent[0:20], ary[0:20]))
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    clf = svm.SVC(kernel= 'linear', C= 1.0)
    clf.fit(x,y)


    entities = [0 for x in range(len(data))]
    total_length = [0 for x in range(len(data))]
    for i in range(0, len(data)):
        doc = nlp(data[i][0])
        entities[i] = ((len(doc.ents)))
        total_length[i] = (len(data[i][0]))

    predictions = list(zip(total_length, entities))

    classify = clf.predict(predictions)
    ## append mode I believe 
    file1 = open("Notes.txt","a") 
    file1.write("-" + data[0][0])
    file1.write("\n")
    for i in range(1, len(classify)): 
        if(classify[i] == 1):
            file1.write("\t" + "\t" + "-" + data[i][0])
        
        elif(classify[i] == 0):
            file1.write("-" + data[i][0])
        file1.write("\n")

    file1.write("\n") 



def sentClassify(numParagraph, alpha, eta):
    lda_modeling = TopicModeling()
    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    lda_modeling.setTopicNum(2)
    for paragraph in paragraphs[:len(paragraphs)]:
        groupedSentence = lda_modeling.groupSentence(paragraph, alpha, eta)       
        classify(groupedSentence)
        ##print(bestGrouping)
        
def main():
    alphaValues = ['symmetric', 'asymmetric', 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    etaValues=[None, 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    sentClassify(5, alphaValues[1], 0.0001)

main()










    


    

    

    

