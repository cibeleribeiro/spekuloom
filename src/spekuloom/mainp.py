import operator
from statistics import median

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk import *
# from nltk.book import *
# from nltk.examples.pt import *
from nltk.corpus import machado
from scipy.interpolate import spline
import util
import pickle
from nltk import BrillTagger

tagger = pickle.load(open("tagger.pkl", "rb"))

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
COUNT_MAX = 20
PATTERNS = 400
CNT_CUT = 7.5
FACTOR = 10
GEN_CNT = 4
TXT_DIR = "/home/carlo/Documentos/doc/academia/projetos/Spekuloom/"
TXT_TYPES = "basico  intermediario  transitorio".split()


class Clauses:
    def __init__(self):
        classes = "contos critica cronica miscelanea poesia romance teatro".split()
        self.token = self.punct = self.clauses = self.patterns = ''
        self.punctuation = list(",.:;!?") + ["CC", "--"]
        self.punctuate = list(",.:;!?-")
        # _text = self.text_setup(text5)
        # self.split_clauses(_text)
        # self.plot_clause('a')
        self.classes = util.Mont().mont_symbol_pt()
        self.classes.update({pt: "\033[1;33m{}\033[1;0m".format(pt) for pt in self.punctuate})
        self.marker()
        self.gens = {gen: [machado.words(txid) for txid in machado.fileids() if gen in txid] for gen in classes}
        self.texts = []
        self.legends = []
        self._patt_data = []
        self._patt_labels = []
        for gen in classes:
            self.legends.extend([gen]*GEN_CNT)
            self.texts.extend([tx for tx in self.gens[gen]][:GEN_CNT])
        # self.texts = [machado.words(conto) for conto in machado.fileids() if "contos" in conto][:20]
        for txt in self.texts:
            print(txt[1000:1004])
        # for fid in machado.fileids():
        #     print(fid)
        # self.survey_corpora()
        # self.survey_patterns()

    def mark_text_ind(self, text):
        self.mark_text(self.texts[text])

    def mark_text(self, text):
        _text = self.text_setup(text)
        _clauses = self.split_clauses(_text)
        self.marker(_clauses)

    def text_setup(self, _text):
        _text = ' '.join(_text[500:4500])
        for pt in self.punctuate:
            _text = _text.replace(" {} ".format(pt), "{} ".format(pt))
        _text = _text.replace("--", u"\u2014")
        _text = _text.replace("- ", "-")
        _text = _text.replace(" ' ", "'")
        _text = _text.replace("Mr.", "Mrp")
        _text = _text.replace("", "")
        return _text

    def add_texts(self):
        legend = []
        texts = []
        for x in TXT_TYPES:
            legend.extend([x] * GEN_CNT)

        self.legends.extend(legend)
        for textype in TXT_TYPES:
            txt_dir = TXT_DIR + textype
            for _, _dir, _texts in os.walk(txt_dir):
                for _, _text in zip(range(GEN_CNT), _texts):
                    print("_dir", txt_dir, _text)
                    _text = open(os.path.join(txt_dir, _text), "r").read()
                    text_words = word_tokenize(_text)
                    texts.append(text_words)
                break
        return legend, texts

    def save_texts(self):
        SP = "\t"
        lines = list()
        lines.append(SP.join(self._patt_labels+["estilo\n"]))
        lines.append(SP.join(["c"]*len(self._patt_labels)+["d\n"]))
        lines.append(SP.join([""]*len(self._patt_labels)+["class\n"]))
        _dat_ = zip(*self._patt_data)
        _data = [SP.join(f"{dat:3<16}" for dat in line)+",{}\n".format(estilo)
                 for estilo, line in zip(self.legends, _dat_)]
        lines.extend(_data)
        with open("estilo.tab", "w") as estilo:
            estilo.writelines(lines)

    def prepare_scatter(self):
        _patt = self._patt_data
        return [[t_patt[delta:delta+GEN_CNT]for t_patt in _patt]
                for delta in range(0, len(self.legends), GEN_CNT)]

    def survey_corpora(self, wind=WINDOW):
        corpora = []
        dcorpora = {}
        lcorpora = []
        cp = PATTERNS
        # _texts = [text1, text2, text3, text6, text7, text8, text9]
        self.legends, self.texts = self.add_texts()
        _texts = self.texts[:]
        # _texts = [machado.words(conto) for conto in machado.fileids() if "contos" in conto]
        print([t[500:504] for t in _texts])
        # return
        _ntexts = range(len(_texts))
        for ind, txt in enumerate(_texts):
            _text = self.text_setup(txt)
            self.split_clauses(_text)
            patterns_surveyed = self.survey_patterns(wind)
            dcorpora[ind] = {k: v for k, v in patterns_surveyed}

            corpora.extend(patterns_surveyed)
            lcorpora.append(patterns_surveyed)

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
        marks *= 5
        _xpat = list(k for k, _ in sorted_pattern[0][:cp])
        _xpat = [_x for _x in _xpat if any([dcorpora[ind].setdefault(_x, 0) > CNT_CUT for ind in _ntexts])]
        corp_avg = {ind: median(dcorpora[ind].setdefault(_x, 0) for _x in _xpat) for ind in _ntexts if _xpat}
        # corp_avg = {ind: sum(dcorpora[ind].setdefault(_x, 0) for _x in _xpat)/len(_xpat) for ind in _ntexts}
        patt = [(_xpat, [FACTOR * dcorpora[ind].setdefault(_x, 0) / (corp_avg[ind] + 0.00001) for _x in _xpat],
                 marks[ind]) for ind in _ntexts if ind < len(marks)]
        labels = ["".join(list(l)[x] for x in range(7, wind * 14, 14)) for l in _xpat]
        print(labels)
        print(list(avp.values()))
        filter, _means = self.check_deviation(patt)
        _xpat = [pt for i, pt in enumerate(_xpat) if i in filter]
        _xpmeans = zip(_xpat, _means)
        for x, m in _xpmeans:
            print(x, end=" ")
            print(m, end=" ")
        _xpmeans = zip(_xpat, _means)
        _texts = self.texts[:]
        _patt_data = [[FACTOR * dcorpora[ind].setdefault(_x, 0) / (_m*corp_avg[ind] + 0.00001) for _x, _m in _xpmeans]
                      for ind in _ntexts]
        self._patt_data = [[FACTOR * dcorpora[ind].setdefault(_x, 0) / (_means[ipat]*corp_avg[ind] + 0.00001)
                            for ipat, _x in enumerate(_xpat)] for ind in _ntexts if ind < len(marks)]
        patt = [(_xpat, [FACTOR * dcorpora[ind].setdefault(_x, 0) / (_means[ipat]*corp_avg[ind] + 0.00001)
                         for ipat, _x in enumerate(_xpat)],
                 marks[ind]) for ind in _ntexts if ind < len(marks)]
        self._patt_labels = labels = [pt for i, pt in enumerate(labels) if i in filter]
        self.save_texts()
        self.plot(self.prepare_scatter(), x=7, y=10)
        print("len(self._patt_data)", len(self._patt_data), len(self._patt_labels))
        # self.plot_patterns(patt, labels)

    def check_deviation(self, patt):
        from statistics import stdev, variance, mean, median_high
        _patt = list(zip(*patt))[1]
        results = [10*stdev(pat)/mean(pat) for pat in zip(*_patt)]
        varia = [variance(pat)/mean(pat) for pat in zip(*_patt)]
        mres = mean(results)
        fresults = [10*stdev(pat)/mean(pat) if 9 < 10*stdev(pat)/mean(pat) else 0 for pat in zip(*_patt)]
        _filter = [i for i, d in enumerate(fresults) if d > 9]
        _means = [mean(pat) / (100/max(pat)) for i, pat in enumerate(zip(*_patt)) if i in _filter]
        print([(i, d) for i, d in enumerate(fresults) if d > 9])
        print(_means)
        return _filter, _means
        plt.plot(results)
        plt.plot(fresults)
        plt.plot(varia)
        plt.show()


    def survey_patterns(self, wind=5):
        def to_shapes(shapes):
            return "".join(self.classes.get(shape[:3], self.classes['ZZ']) for shape in shapes.split(":"))

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
        # mak_claus = [(self.classes[tag[:2]] if tag[:2] in self.classes else no) + wd for wd, tag in clause]
        mak_claus = [(self.classes[tag[:3]] if tag[:3] in self.classes else no) + wd for wd, tag in clause]
        # mak_claus = [tag + wd for wd, tag in clause]
        print(' '.join(mak_claus))

    def split_clauses(self, text):
        token = word_tokenize(text, language='portuguese')
        self.token = token = tagger.tag(token)
        # self.token = token = nltk.pos_tag(token)
        self.punct = pun = [index for index, (txt, punct) in enumerate(self.token) if punct in self.punctuation]
        # hist = [(punct, sum(1 for x, p in self.token if punct == p))
        #         for punct in self.punctuation]
        self.clauses = [token[start + (0 if token[start][1] == "CC" else 1): stop] + [self.token[stop]]
                        for start, stop in list(zip(pun, pun[1:])) if stop > (start + 1)]
        return self.clauses

    def plot(self, txdata, x=0, y=1, colors="red blue green".split()):
        for style, style_color in enumerate(colors):
            plt.scatter(txdata[style][x], txdata[style][y], color=style_color)
        plt.show()

    def plot_patterns(self, patt, labels):
        from itertools import cycle
        lines = ["-", "-", "-", "--", "--", "--", ":", ":", ":"]
        lines = ["-"]*GEN_CNT + ["--"]*GEN_CNT +  [":"]*GEN_CNT
        # lines = ["-", "--", "-.", ":"]
        linecycler = cycle(lines)
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
            if not _x:
                continue
            x_sm = np.array(_x)
            x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)
            # y_smooth = spline(_x, npy.log([y+0.001 for y in _y]), x_smooth)
            y_smooth = spline(_x, [y if y > COUNT_MIN else 0 for y in _y], x_smooth)
            plt.plot(x_smooth, y_smooth, next(linecycler))
            # plt.plot(range(len(_y)), _y, _c)
        plt.legend(self.legends, loc='upper right')
        # plt.legend(["moby", "sense", "genesis", "monty", "wall", "person", "thursday"], loc='upper right')

        plt.show()


if __name__ == '__main__':
    # Clauses().mark_text_ind(6)
    Clauses().survey_corpora(4)

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
