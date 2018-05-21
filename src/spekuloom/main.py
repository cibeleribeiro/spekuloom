import operator

import nltk
from nltk import *
from nltk.book import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from util import Mont


class Clauses:
    def __init__(self):
        self.token = self.punct = self.clauses = ''
        self.punctuation = list(",.:;!?")+["CC", "--"]
        self.punctuate = list(",.:;!?-")
        _text = self.text_setup(text5)
        self.split_clauses(_text)
        # self.plot_clause('a')
        self.classes = Mont().mont_symbol()
        # self.marker()
        self.survey_corpora()
        # self.survey_patterns()

    def text_setup(self, text):
        text = ' '.join(text[400:2500])
        for pt in self.punctuate:
            text = text.replace(" {} ".format(pt), "{} ".format(pt))
        text = text.replace("--", u"\u2014")
        text = text.replace("- ", "-")
        text = text.replace(" ' ", "'")
        text = text.replace("Mr.", "Mrp")
        return text

    def survey_corpora(self, wind=3):
        corpora = []
        dcorpora = {}
        _texts =[text1, text2, text3, text4]
        for ind, txt in enumerate(_texts):
            _text = self.text_setup(txt)
            self.split_clauses(_text)
            dcorpora[ind] = {k: v for k, v in self.survey_patterns(wind)}

            corpora.extend(self.survey_patterns(wind))

        self.patterns = {}
        for pat in corpora:
            self.patterns[pat] = self.patterns.setdefault(pat, 0)+1
        sorted_pattern = [(shape, count)
                          for shape, count in sorted(
                self.patterns.items(), key=operator.itemgetter(1), reverse=True)][:60]
        for shape, count in sorted_pattern:
            print(shape[0], end='')
            print(count)
        for shape, count in dcorpora[3].items():
            print(shape[0], end='')
            print(count)
        marks = 'rs gs bs ms r^ g^ b^ m^ c^'.split()
        _xpat = list(k for k, _ in sorted_pattern)
        patt = [(_xpat, [dcorpora[ind].setdefault(_x, 0) for _x, _ in _xpat],
                 marks[ind]) for ind in range(len(_texts))]
        labels = ["".join(list(l)[x] for x in range(7, wind*14, 14)) for l, _ in _xpat]
        print(labels)
        self.plot_patterns(patt, labels)

    def survey_patterns(self, wind=5):
        def to_shapes(shapes):
            return "".join(self.classes.get(shape[:2], self.classes['ZZ']) for shape in shapes.split(":"))

        def window(clause):
            return [":".join(x) for x in zip(*[[_tag for _, _tag in clause[ind:]]for ind in range(wind)])]
            # return zip(clause, clause[1:], clause[2:])
        patterns = [pat for a_clause in self.clauses for pat in window(a_clause)]
        self.patterns = {}
        for pat in patterns:
            self.patterns[pat] = self.patterns.setdefault(pat, 0)+1
        sorted_pattern = [(to_shapes(shape), count)
                          for shape, count in sorted(self.patterns.items(), key=operator.itemgetter(1), reverse=True)]
        # for shape, count in sorted_pattern[:60]:
        #     print(shape, end='')
        #     print(count)
        # print(self.patterns)
        return sorted_pattern

    def marker(self):
        print(self.text)
        _ = [self.mark_clauses(clause) for clause in self.clauses[:200]]

    def mark_clauses(self, clause):
        NO = self.classes["ZZ"]
        mak_claus = [(self.classes[tag[:2]] if tag[:2] in self.classes else NO) + wd for wd, tag in clause]
        # mak_claus = [tag + wd for wd, tag in clause]
        print(' '.join(mak_claus))

    def split_clauses(self, text):
        token = word_tokenize(text)
        self.token = token = nltk.pos_tag(token)
        punctuation=",.:;!?CCIN--"
        # self.punct = [(index, punct) for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        self.punct = pun= [index for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        hist = [(punct, sum(1 for x, p in self.token if punct == p))
                for punct in self.punctuation]
        self.clauses = [token[start + (0 if token[start][1] == "CC" else 1): stop]+[self.token[stop]]
                        for start, stop in list(zip(pun, pun[1:])) if stop > (start+1)]
        return self.clauses

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

    def plot_patterns(self, patt, labels):
        fig = plt.figure()
        fig.suptitle('bold figure subtitle', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('axes title')
        ax.set_xticklabels(labels)
        plt.xticks(list(range(0, len(labels), 2)))
        # handles = ax.get_legend()
        handles, labels = ax.get_legend_handles_labels()
        labels = ['title{}'.format(range(len(patt)))]
        ax.legend(handles, labels)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax.set_xlabel('xlabel')
        ax.set_ylabel('ylabel')

        for _x, _y, _c in patt:
            plt.plot(range(len(_y)), _y, _c)
        plt.legend(["moby", "sense", "genesis", "address"], loc='upper right')

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