import operator
from statistics import median

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import *
from nltk.book import *
from scipy.interpolate import spline
import util

#
'''
WINDOW = 3
COUNT_MIN = -2
COUNT_MAX = 50
PATTERNS = 400
CNT_CUT = 15
FACTOR = 10
==============
WINDOW = 2
COUNT_MIN = -2
COUNT_MAX = 80
PATTERNS = 400
CNT_CUT = 25
FACTOR = 10
==============
WINDOW = 4
COUNT_MIN = -2
COUNT_MAX = 100
PATTERNS = 400
CNT_CUT = 6
FACTOR = 10
==============
WINDOW = 5
COUNT_MIN = -2
COUNT_MAX = 100
PATTERNS = 400
CNT_CUT = 5
FACTOR = 10
'''
WINDOW = 4
COUNT_MIN = -2
COUNT_MAX = 100
PATTERNS = 400
CNT_CUT = 10
FACTOR = 10


class Clauses:
    def __init__(self):
        self.token = self.punct = self.clauses = self.patterns = ''
        self.punctuation = list(",.:;!?") + ["CC", "--"]
        self.punctuate = list(",.:;!?-")
        # _text = self.text_setup(text5)
        # self.split_clauses(_text)
        # self.plot_clause('a')
        self.classes = util.Mont().mont_symbol()
        self.classes.update({pt: "\033[1;33m{}\033[1;0m".format(pt) for pt in self.punctuate})
        # self.marker()
        # self.survey_corpora()
        # self.survey_patterns()

    def mark_text(self, text):
        _text = self.text_setup(text)
        _clauses = self.split_clauses(_text)
        self.marker(_clauses)

    def text_setup(self, text):
        text = ' '.join(text[500:8500])
        for pt in self.punctuate:
            text = text.replace(" {} ".format(pt), "{} ".format(pt))
        text = text.replace("--", u"\u2014")
        text = text.replace("- ", "-")
        text = text.replace(" ' ", "'")
        text = text.replace("Mr.", "Mrp")
        return text

    def survey_corpora(self, wind=WINDOW):
        corpora = []
        dcorpora = {}
        lcorpora = []
        cp = PATTERNS
        _texts = [text1, text2, text3, text6, text7, text8, text9]
        _ntexts = range(len(_texts))
        for ind, txt in enumerate(_texts):
            _text = self.text_setup(txt)
            self.split_clauses(_text)
            dcorpora[ind] = {k: v for k, v in self.survey_patterns(wind)}

            corpora.extend(self.survey_patterns(wind))
            lcorpora.append(self.survey_patterns(wind))

        self.patterns = {}
        for pat in corpora:
            self.patterns[pat] = self.patterns.setdefault(pat, 0) + 1
        average_pattern = [
            (shape, count) for shape, count in sorted(
                self.patterns.items(), key=operator.itemgetter(1), reverse=True)]
        for shape, count in average_pattern:
            print(shape[0], end='')
            print(count)
        for shape, count in dcorpora[3].items():
            print(shape[0], end='')
            print(count)
        sorted_pattern = []
        avp = {k: v for k, v in average_pattern}
        _ = [sorted_pattern.extend(_pat) for _pat in zip(lcorpora)]
        for sp, c in sorted_pattern[0][:cp]:
            print("sp : {} {}\n".format(sp, c))
        marks = 'rs gs bs ms c^ yv k. w* c^'.split()
        _xpat = list(k for k, _ in sorted_pattern[0][:cp])
        _xpat = [_x for _x in _xpat if any([dcorpora[ind].setdefault(_x, 0) > CNT_CUT for ind in _ntexts])]
        corp_avg = {ind: median(dcorpora[ind].setdefault(_x, 0) for _x in _xpat) for ind in _ntexts}
        # corp_avg = {ind: sum(dcorpora[ind].setdefault(_x, 0) for _x in _xpat)/len(_xpat) for ind in _ntexts}
        patt = [(_xpat, [FACTOR * dcorpora[ind].setdefault(_x, 0) / corp_avg[ind] for _x in _xpat],
                 marks[ind]) for ind in _ntexts]
        labels = ["".join(list(l)[x] for x in range(7, wind * 14, 14)) for l in _xpat]
        print(labels)
        print(list(avp.values()))
        self.plot_patterns(patt, labels)

    def survey_patterns(self, wind=5):
        def to_shapes(shapes):
            return "".join(self.classes.get(shape[:2], self.classes['ZZ']) for shape in shapes.split(":"))

        def window(clause):
            return [":".join(x) for x in zip(*[[_tag for _, _tag in clause[ind:]] for ind in range(wind)])]
            # return zip(clause, clause[1:], clause[2:])

        patterns = [pat for a_clause in self.clauses for pat in window(a_clause)]
        self.patterns = {}
        for pat in patterns:
            self.patterns[pat] = self.patterns.setdefault(pat, 0) + 1
        sorted_pattern = [(to_shapes(shape), count)
                          for shape, count in sorted(self.patterns.items(), key=operator.itemgetter(1), reverse=True)]
        # for shape, count in sorted_pattern[:60]:
        #     print(shape, end='')
        #     print(count)
        # print(self.patterns)
        return sorted_pattern

    def marker(self, clauses=None, clip=200):
        # print(self.text)
        clauses = clauses if clauses else self.clauses
        _ = [self.mark_clauses(clause) for clause in clauses[:clip]]

    def mark_clauses(self, clause):
        no = self.classes["ZZ"]
        mak_claus = [(self.classes[tag[:2]] if tag[:2] in self.classes else no) + wd for wd, tag in clause]
        # mak_claus = [tag + wd for wd, tag in clause]
        print(' '.join(mak_claus))

    def split_clauses(self, text):
        token = word_tokenize(text)
        self.token = token = nltk.pos_tag(token)
        self.punct = pun = [index for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        # hist = [(punct, sum(1 for x, p in self.token if punct == p))
        #         for punct in self.punctuation]
        self.clauses = [token[start + (0 if token[start][1] == "CC" else 1): stop] + [self.token[stop]]
                        for start, stop in list(zip(pun, pun[1:])) if stop > (start + 1)]
        return self.clauses

    def plot_patterns(self, patt, labels):
        _ = self.punct
        fig = plt.figure()
        fig.suptitle('Corpora Pattern Count', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.92, left=0.05, right=0.98, bottom=0.08)
        # ax.set_title('axes title')
        ax.set_xticklabels(labels)
        plt.xticks(list(range(0, len(labels))))
        # handles = ax.get_legend()
        handles, labels = ax.get_legend_handles_labels()
        labels = ['title{}'.format(range(len(patt)))]
        ax.legend(handles, labels)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        axes = plt.gca()
        axes.set_ylim([COUNT_MIN, COUNT_MAX])
        ax.set_xlabel('patterns')
        ax.set_ylabel('count')
        plt.xticks(rotation=90)
        for _x, _y, _c in patt:
            _x = list(range(len(_y)))
            x_sm = np.array(_x)
            x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
            # y_smooth = spline(_x, npy.log([y+0.001 for y in _y]), x_smooth)
            y_smooth = spline(_x, [y if y > COUNT_MIN else 0 for y in _y], x_smooth)
            plt.plot(x_smooth, y_smooth)
            # plt.plot(range(len(_y)), _y, _c)
        plt.legend(["moby", "sense", "genesis", "monty", "wall", "person", "thursday"], loc='upper right')

        plt.show()


if __name__ == '__main__':
    Clauses().mark_text(text5)

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
