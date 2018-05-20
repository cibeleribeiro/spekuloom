import nltk
from nltk import *
from nltk.book import text2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Clauses:
    def __init__(self):
        self.text = ' '.join(text2[400:900])
        self.token = self.punct = self.clauses = ''
        self.punctuation = list(",.:;!?")+["CC", "IN", "--"]
        print(self.punctuation)
        # self.split_clauses()
        self.plot_clause('a')

    def split_clauses(self):
        token = word_tokenize(self.text)
        self.token = nltk.pos_tag(token)
        punctuation=",.:;!?CCIN--"
        # self.punct = [(index, punct) for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        self.punct = pun= [index for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        hist = [(punct, sum(1 for x, p in self.token if punct == p))
                for punct in self.punctuation]
        self.clauses = [self.token[start: stop] for start, stop in zip([0]+pun, pun[1:])]
        print(self.clauses)

    def plot_clause(self, a_clause):
        fig = plt.figure()
        fig.suptitle('bold figure subtitle', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('axes title')

        ax.set_xlabel('xlabel')
        ax.set_ylabel('ylabel')

        N = 100
        r0 = 0.6
        x = 0.9 * np.random.rand(N)
        y = 0.9 * np.random.rand(N)
        area = (20 * np.random.rand(N)) ** 2  # 0 to 10 point radii
        c = np.sqrt(area)
        r = np.sqrt(x * x + y * y)
        area1 = np.ma.masked_where(r < r0, area)
        area2 = np.ma.masked_where(r >= r0, area)
        plt.scatter(x, y, s=area1, marker='^', c=c)
        plt.scatter(x, y, s=area2, marker='o', c=c)
        # Show the boundary between the regions:
        # theta = np.arange(0, np.pi / 2, 0.01)
        # plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

        plt.show()


Clauses()

"""
Noun 	black tri l
Article cyan tri s
Adjective 	blue tri m
Pronoun 	magenta tri l
Verb 	red cir l
Adverb 	yellow cir s
Conjunction 	magenta squ l
Preposition 	in, on, under, to, at, before
Interjection 	hey, ah, wow, ouch, hello, shh
"""