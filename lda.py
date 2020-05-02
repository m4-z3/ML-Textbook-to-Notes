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
import string

class TopicModeling:    

    def __init__(self):
        """Start up code that should be run once in order to set up for separating sentences by topic"""
        # making sure user has required resources
        try:
            nltk.data.find('corpora/wordnet')
        except:
            nltk.download('wordnet')

        try: 
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # list of stop words to remove curtesy of https://gist.github.com/sebleier/554280
        self._stop_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
        # set pattern to tokenize sentence to words
        self._word_tokenizer = tokenize.RegexpTokenizer(r"[a-zA-Z]{3,}(?:-[a-zA-Z]+)+|[a-zA-Z]{3,}(?:'t)?")
        # creates lemmatizer
        self._lemmatizer = WordNetLemmatizer()
        self._topicNum = 2

    def setTopicNum(self, topics):
        # mainly used for testing
        self._topicNum = topics

    def groupSentence(self, paragraph, alpha='asymmetric', eta=0.000001):
        """Group sentences by similar topic"""
        # prevent separating at periods which don't denote the end of a sentence
        wordWithPeriod = ['Fig.', 'Ms.', 'Mrs.', 'Mr.', 'Dr.', 'St.']
        for w in wordWithPeriod:
            paragraph = paragraph.replace(w, w.replace('.', ','))
        # separates paragraph into individual sentences
        sentenceList = tokenize.sent_tokenize(paragraph)
        # making sure no empty strings or strings with only special characters were included
        sentenceList = [sentence for sentence in sentenceList if sentence != "" and not all(t in string.punctuation for t in sentence)] 

        # preprocessing for lda
        self.processedSentenceList = self.preprocessing(sentenceList)

        # create dictionary (which maps all words to its unique id)
        self.dictionary = corpora.Dictionary(self.processedSentenceList)

        # create corpus (which creates bag of words)
        self.corpus = [self.dictionary.doc2bow(sentence) for sentence in self.processedSentenceList]
        # NOTE: will be changing hyper parameters
        # creating the lda model
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self._topicNum, random_state=100, chunksize=3, update_every=1, passes=60, alpha=alpha, eta=eta)

        # print(self.lda_model.print_topics())

        groupedSentenceList = []
        previousTopicNum = -1
        
        # grouping sequential sentences with the same topic together
        for i in range(len(self.corpus)):
            # returns tuple that contains the topic index and highest score value
            indexScoreTup = sorted(self.lda_model[self.corpus[i]], key= lambda tup : tup[1], reverse=True)[0]
            # isolates the topic index
            topicIndex = indexScoreTup[0]

            # group sequential sentences together based on topic
            if previousTopicNum != topicIndex:
                groupedSentenceList.append([sentenceList[i]])
                previousTopicNum = topicIndex
            else:
                groupedSentenceList[-1].append(sentenceList[i])

        return groupedSentenceList

    def preprocessing(self, sentenceList):
        """Preprocessing the input in order to provide clean data to lda"""
        processedSentenceList = []
        # documents number of words in 
        self.num_tokens = 0
        for sentence in sentenceList:
            sentence = sentence.lower()
            # extract all words from sentence
            # this excludes numeric values and punctuation unless it's a hyphen or apostrophe for 't 
            processing = self._word_tokenizer.tokenize(sentence)
            self.num_tokens += len(processing)
            # remove stop words
            processing = [token for token in processing if not token in self._stop_words]
            # lemmatize words
            processing = [self._lemmatizer.lemmatize(token) for token in processing]

            processedSentenceList.append(processing)

        return processedSentenceList

    def coherenceScore(self):
        """Gets the coherence score of the current model (as well as word count if needed)"""
        coherence = CoherenceModel(model=self.lda_model, corpus=self.corpus, dictionary=self.dictionary, texts=self.processedSentenceList, coherence='c_npmi')
        coherenceScore = coherence.get_coherence()
        return (coherenceScore, self.num_tokens)

    def visualize(self):
        """Visualizes the current lda model"""
        visualizeLDA = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.dictionary)
        pyLDAvis.show(visualizeLDA)

    def getNumTokens(self):
        return self.num_tokens