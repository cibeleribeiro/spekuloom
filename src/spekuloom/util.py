#import nltk
#from nltk import *
#from nltk.book import *
#from spekuloom.word_class import tagset

class Mont:
    def __init__(self):
        #self.text = text2
        #self.word_classes()
        self.mont_symbol()

    def word_classes(self):
        print(tagset)

    def mont_symbol(self):

        #classes=dict(noun,article,adjective, verb, adverb, preposition, conjuction,interjection, pronoun)
        classes=dict(NN = "\033[1;31m"+u"\u25B2",
                     DT = "\033[1;36m"+u"\u25B4",
                     JJ = "\033[1;34m"+u"\u25ED",
                     VB = "\033[1;31m"+u"\u25CF",
                     MD = "\033[1;31m"+u"\u25CF",
                     RB = "\033[1;31m"+u"\u25CB",
                     IN = "\033[1;32m"+u"\u25E1",
                     CC = "\033[1;35m"+u"\u25AC",
                     UH = "\033[1;33m"+u"\u25C8",
                     PR = "\033[1;35m"+u"\u25BB")
        for a, b in classes.items():print(b)




Mont()

