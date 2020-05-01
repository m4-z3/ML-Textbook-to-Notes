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

    #print(type_sent)
    # plt.scatter(len_sent[0:10], ary[0:10], c='b')
    # plt.scatter(len_sent[10:20],ary[10:20] , c='r')
    # plt.show()

    x = list(zip(len_sent[0:20], ary[0:20]))

    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    clf = svm.SVC(kernel= 'linear', C= 1.0)
    clf.fit(x,y)


    ## to visualize 
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(0,400)
    yy = a * xx - clf.intercept_[0] / w[1]
    h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

    # plt.scatter(len_sent[0:10], ary[0:10], c='b')
    # plt.scatter(len_sent[10:20],ary[10:20] , c='r')
    # plt.legend()
    # plt.show()


    # docs = nlp(data[0][0])
    # entities = len(docs.ents)
    # total_length = len(data[0][0])
    # print(entities)
    # print(total_length)
    # final = list(zip(entities, total_length))
    # print(clf.predict(final))



    
        

         







    


    

    

    

