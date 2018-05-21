import operator

import nltk
from nltk import *
from nltk.book import text2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from util import Mont


class Clauses:
    def __init__(self):
        self.text = text = ' '.join(text2[400:1500])
        self.token = self.punct = self.clauses = ''
        self.punctuation = list(",.:;!?")+["CC", "--"]
        punctuation = list(",.:;!?-")
        # print(self.punctuation)
        for pt in punctuation:
            text = text.replace(" {} ".format(pt), "{} ".format(pt))
        text = text.replace("--", u"\u2014")
        text = text.replace("- ", "-")
        text = text.replace(" ' ", "'")
        text = text.replace("Mr.", "Mrp")
        self.text = text
        self.split_clauses()
        # self.plot_clause('a')
        self.classes = Mont().mont_symbol()
        # self.marker()
        self.survey_patterns()

    def survey_patterns(self):
        def window(clause):
            return zip(clause, clause[1:], clause[2:])
        patterns = [(a, b, c) for a_clause in self.clauses for (_, a), (_, b), (_, c) in window(a_clause)]
        self.patterns = {}
        for pat in patterns:
            self.patterns[pat] = self.patterns.setdefault(pat, 0)+1
        print(sorted(self.patterns.items(), key=operator.itemgetter(1), reverse=True))
        # print(self.patterns)


    def marker(self):
        print(self.text)
        _ = [self.mark_clauses(clause) for clause in self.clauses[:200]]

    def mark_clauses(self, clause):
        NO = self.classes["ZZ"]
        mak_claus = [(self.classes[tag[:2]] if tag[:2] in self.classes else NO) + wd for wd, tag in clause]
        # mak_claus = [tag + wd for wd, tag in clause]
        print(' '.join(mak_claus))

    def split_clauses(self):
        token = word_tokenize(self.text)
        self.token = token = nltk.pos_tag(token)
        punctuation=",.:;!?CCIN--"
        # self.punct = [(index, punct) for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        self.punct = pun= [index for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        hist = [(punct, sum(1 for x, p in self.token if punct == p))
                for punct in self.punctuation]
        self.clauses = [token[start + (0 if token[start][1] == "CC" else 1): stop]+[self.token[stop]]
                        for start, stop in list(zip(pun, pun[1:])) if stop > (start+1)]
        # print(self.clauses)

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