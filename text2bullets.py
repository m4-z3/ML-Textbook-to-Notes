from lda import TopicModeling
from sent_class import classify

def main():
    lda_modeling = TopicModeling()
    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    
    for paragraph in paragraphs[0:60]:
        groupedSentence = lda_modeling.groupSentence(paragraph) 
        classify(groupedSentence)

if __name__ == "__main__":
    main()