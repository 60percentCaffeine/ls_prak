class Sentence:
    def __init__(self, words=[], complex_words=[]):
        self.words = words # list of strings
        self.complex_words = complex_words # list of indeces of Sentence.words
    
    def __str__(self):
        return "Sentence(" + str(self.words) + "," + str([self.words[x] for x in self.complex_words]) + ")"

class Document:
    def __init__(self, sentences=[], dictionary=[], dictionary_size=0):
        self.sentences = sentences
        self.dictionary = dictionary
        self.dictionary_size = dictionary_size

def print_sentences(sentences):
    for x in sentences:
        print(str(x))

#####################
### preprocessing ###
#####################

import nltk
nltk.download('punkt')

from pymystem3 import Mystem
mystem = Mystem()

def preproc(text):
    # tokenize the text into sentences first
    sentences = nltk.sent_tokenize(text, language="russian")

    # tokenize & lemmatize words in every sentence
    sentences_lemmatized = list(map(lambda x: Sentence(mystem.lemmatize(x)), sentences))

    return Document(sentences_lemmatized)

###################################
### complex word identification ###
###################################

# preprocessing leaves punctuation in, so we need a function to ignore it
alphabet_str = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
alphabet = dict((x, True) for x in alphabet_str)

def is_punkt(x):
    for c in x.lower():
        if not (c in alphabet):
            return True
    
    return False

from wordfreq import zipf_frequency, freq_to_zipf

# returns a zipf freq dictionary for a document
def make_document_dictionary(self):
    dictionary = {}
    size = 0

    for sentence in self.sentences:
        for word in sentence.words:
            if not is_punkt(word):
                dictionary[word] = (dictionary[word] + 1) if (word in dictionary) else 1
                size += 1
    
    for word in dictionary:
        dictionary[word] = freq_to_zipf(dictionary[word] / size)

    self.dictionary = dictionary
    self.dictionary_size = size

Document.make_dictionary = make_document_dictionary

CWI_SCORE_THRESHOLD = 3.5

# calculate the CWI score of the word in a document
# TODO: calculate the weight for the document's dictionary based on its size?
def word_cwi_score(self, word):
    return 0.5 * self.dictionary[word] + 0.5 * zipf_frequency(word, lang="ru")

Document.word_cwi_score = word_cwi_score

def cwi(self):
    if self.dictionary_size == 0:
        raise ValueError("The document dictionary has not been initialized yet.")
    
    for sentence in self.sentences:
        sentence.complex_words = []
        for k, word in enumerate(sentence.words):
            if (not is_punkt(word)) and self.word_cwi_score(word) < CWI_SCORE_THRESHOLD:
                sentence.complex_words.append(k)

Document.cwi = cwi