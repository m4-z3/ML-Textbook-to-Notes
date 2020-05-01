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
from joblib import dump, load

clf = load('model.joblib') 

def classify(data): 

    # print("length of data: ")
    # print(len(data))
    # print("\n")
    # for i in range(len(data)):
    #     print(len(data[i]))
    #     print("\n")

    file1 = open("Notes.txt","a")
    for i in range(len(data)):
        entities = []
        total_length = []
        predict_array = []
        for j in range(len(data[i])): 
            doc = nlp(data[i][j])
            entities.append((len(doc.ents)))
            total_length.append((len(data[i][j])))
            predictions = list(zip(total_length, entities))
            predict_array = clf.predict(predictions)
        
        file1.write("-" + data[i][0])
        file1.write("\n")
        
        for y in range(1,len(predict_array)): 
            if(predict_array[y] == 1 and y != 0): 
                file1.write("\t" + "\t" + "-" + data[i][y])
                file1.write("\n")
            elif(predict_array[y] == 0 and y != 1):
                file1.write("\t" + "-" + data[i][y])
                file1.write("\n")
    file1.write("\n")

        # for x in range(len(data[i])):
        #     print(len(data[i][x]))
        #     for y in range(len(data[i][x])): 
                
            #     if(predict_array[0][y] == 1 and y != 0): 
            #         file1.write("\t" + "\t" + "-" + data[i][y])
            #         file1.write("\n")
            #     elif(predict_array[0][y] == 0 and y != 1):
            #         file1.write("-" + data[i][y])
            #         file1.write("\n")
            # file1.write("\n")



    # col = []
    # for i in range(0, len(data)): 
    #     col.append(len(data[i]))

    # classify_data = []
    # for i in range(0, len(data)):
    #     print("Changing the paragraph")
    #     entities = []
    #     total_length = []
    #     predict_ary = []
    #     for j in range(0, len(data[i])):
    #         doc = nlp(data[i][j])
    #         entities.append((len(doc.ents)))
    #         total_length.append((len(data[i][j])))
    #         predictions = list(zip(total_length, entities))
    #         predict_ary.append(np.array((clf.predict(predictions))))
    #     #print(predict_ary)
    #     #file1 = open("Notes.txt","a")
    #     print(len(predict_ary[i]))
    #     print(len(data[i][j]))
        # for x in range(len(data[i])):
        #     for y in range()
        


    #simple_array[1:3]
    
    ## append mode I believe 
    #  
    # #
    # for i in range(0, len(classify)): 
    #     for j in range(0, len(data[i])):
    #         if(classify[i] == 1 and j != 0):
    #             file1.write("\t" + "\t" + "-" + data[i][j])
    #             file1.write("\n")
    #         elif(classify[i] == 0):
    #             file1.write("-" + data[i][j])
    #             file1.write("\n")
    #     file1.write("\n")
        



def sentClassify():
    lda_modeling = TopicModeling()
    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    
    for paragraph in paragraphs[0:60]:
        groupedSentence = lda_modeling.groupSentence(paragraph)   
        #print("CHANGES HERE" , "\n ", "\n")    
        classify(groupedSentence)
        ##print(bestGrouping)
        
def main():
    alphaValues = ['symmetric', 'asymmetric', 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    etaValues=[None, 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    sentClassify()

main()