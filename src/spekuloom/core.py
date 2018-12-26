#! /usr/bin/env python
# -*- coding: UTF8 -*-
# Este arquivo é parte do programa Spekuloom
# Copyright 2010-2018 Carlo Oliveira <carlo@nce.ufrj.br>,
# `Labase <http://labase.selfip.org/>`__; `GPL <http://j.mp/GNU_GPL3>`__.
#
# Spekuloom é um software livre; você pode redistribuí-lo e/ou
# modificá-lo dentro dos termos da Licença Pública Geral GNU como
# publicada pela Fundação do Software Livre (FSF); na versão 2 da
# Licença.
#
# Este programa é distribuído na esperança de que possa ser útil,
# mas SEM NENHUMA GARANTIA; sem uma garantia implícita de ADEQUAÇÃO
# a qualquer MERCADO ou APLICAÇÃO EM PARTICULAR. Veja a
# Licença Pública Geral GNU para maiores detalhes.
#
# Você deve ter recebido uma cópia da Licença Pública Geral GNU
# junto com este programa, se não, veja em <http://www.gnu.org/licenses/>

"""Core text processing.

.. moduleauthor:: Carlo Oliveira <carlo@nce.ufrj.br>

"""
import operator
import os
import pickle
from enum import Enum, auto

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar, show
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.ma import arange

from spekuloom.util import Mont


class Z:
    GEN_CNT: int = 11
    PATT_CUT: int = 200
    CLAZ: dict = Mont().mont_symbol_pt()
    KIND_CROP_PT: int = 3
    WINDOW: int = 4
    WINDOW_OVERLAP: int = 3
    COUNT_MIN: int = -2
    COUNT_MAX: int = 20
    PATTERNS: int = 400
    CNT_CUT: int = 7.5
    TXT_OFF: int = 4000
    TXT_CUT: int = 8000
    FACTOR: int = 10
    TXT_DIR: str = "/home/carlo/Documentos/dev/spekuloom/src/Spekuloom/"
    TXT_TYPES: list = "basico  intermediario  transitorio".split()


class Agregator(Enum):
    WORD_PATTERN = auto()


tagger = pickle.load(open("tagger.pkl", "rb"))


class Fragment:
    """
    Text available in corpora.
    """
    PATTERN = {patt: None for patt in Agregator}

    def __init__(self, text, kind="", parent=None, name="_anom_"):
        self.name = name
        self._parent = parent
        self._text = text
        self._kind = kind
        self._fragments = self.tokenize()
        self._pattern = self.symbolize()

    @property
    def word_pattern(self):
        return Fragment.PATTERN[Agregator.WORD_PATTERN]

    @property
    def parent(self):
        return self._parent

    @property
    def kind(self):
        return self._kind

    @property
    def fragments(self):
        return self._fragments

    def tokenize(self):
        return [Sentence(a_sentence, parent=self) for a_sentence in sent_tokenize(self._text, "portuguese")]

    def __repr__(self):
        return "\t<{}>: {}:{}--${}$-{}*µ*{}*\n".format(
            type(self).__name__, self._kind, self._pattern, len(self._fragments),
            self._text[:42], [frag.__repr__() for frag in self._fragments][:4])

    def symbolize(self):
        return "©"

    @property
    def pattern(self):
        return self._pattern


class Pattern(dict):
    def add(self, pattern: str, host: Fragment = None) -> str:
        self[pattern] = self.setdefault(pattern, []) + [host]
        return pattern


class Corpora(Fragment):
    """
        Collection of texts representing a literate language
    """

    def __init__(self, text, kind="", off=Z.TXT_OFF, cut=Z.TXT_CUT):
        # print("Text", text[:4], kind)
        Fragment.PATTERN[Agregator.WORD_PATTERN] = Pattern()
        self.off, self.cut = off,cut
        super().__init__(text, kind)

    def tokenize(self):
        gen_cnt, text_types = Z.GEN_CNT, Z.TXT_TYPES
        self._kind = []
        _texts = list()
        for x in text_types:
            self._kind.extend([x] * gen_cnt)
        _corp = [TextC(
            open(os.path.join(os.path.join(Z.TXT_DIR, textype), _text), "r").read()[self.off:self.cut],
            kind=textype, name=_text)
            for textype in text_types
            for _, _dir, _texts in os.walk(os.path.join(Z.TXT_DIR, textype))
            for _, _text in zip(range(gen_cnt), _texts)
        ]
        _corp += [TextC(
            open(os.path.join(os.path.join(Z.TXT_DIR, textype), _text), "r").read()[self.off+self.cut:self.cut+self.cut],
            kind=textype, name=_text)
            for textype in text_types
            for _, _dir, _texts in os.walk(os.path.join(Z.TXT_DIR, textype))
            for _, _text in zip(range(gen_cnt), _texts)
        ]
        return _corp


class TextC(Fragment):
    """
    Text available in corpora.
    """

    def __init__(self, text, kind="", name="_anon_"):
        # print("Text", text[:4], kind)
        super().__init__(text, kind, name=name)

    def symbolize(self):
        fragments = [pattern for sentence in self.fragments for pattern in sentence.pattern]
        window_of_n_words = [fragments[offset:] for offset in range(Z.WINDOW)]
        window_of_n_words.append(list(range(Z.WINDOW_OVERLAP)) * 1000)
        return ["".join(word for word in window) for *window, count in zip(*window_of_n_words)
                if all(x is not None for x in window) and not count]


class Sentence(Fragment):
    """
    Sentences broken by tokenizer.
    """

    def __init__(self, text, kind="", parent=None):
        super().__init__(text, kind, parent=parent)

    def tokenize(self):
        return [Word(text, kind, parent=self) for text, kind in tagger.tag(word_tokenize(self._text, "portuguese"))]

    def symbolize(self):
        window_of_n_words = [self._fragments[offset:] for offset in range(Z.WINDOW)]
        window_of_n_words.append(list(range(Z.WINDOW_OVERLAP)) * 1000)
        return [self.word_pattern.add("".join(word.pattern for word in window), host=self)
                for *window, count in zip(*window_of_n_words)
                if all(x.pattern is not None for x in window) and not count]


class Word(Fragment):
    """
    Words broken by tokenizer.
    """

    def __init__(self, text, kind="", parent=None):
        super().__init__(text, kind, parent=parent)

    def tokenize(self):
        _ = self
        return []

    def symbolize(self):
        return Z.CLAZ[self._kind[:Z.KIND_CROP_PT]] if self._kind[:Z.KIND_CROP_PT] in Z.CLAZ else None


class Inscription(Fragment):
    def __init__(self, text="portuguese", corpora=None, out_text="estilo.tab"):
        super().__init__(text)
        self.selected_patterns = []
        self.corpora = corpora or Corpora(text)
        self.pattern_count_by_text = {}
        self.out_text = out_text

    def _scatter_plot(self, txdata, x=0, y=1, colors="red blue green".split()):
        _ = self
        for style, style_color in enumerate(colors):
            plt.scatter(txdata[x][style], txdata[y][style], color=style_color)
        plt.show()

    def scatter_plot(self, txdata, x=0, y=1, colors="red blue green".split()):
        def strip_vt100(name):
            return "".join(letter for letter in name if letter not in "\x1b[0123456789m[1;")

        fig1 = plt.figure()
        # x = range(len(data)+2)
        # plt.ylim(0, 35)
        # plt.xlim(0, 128)
        plt.xlabel(strip_vt100(self.selected_patterns[x]))
        plt.ylabel(strip_vt100(self.selected_patterns[y]))
        plt.title('Contagem de Padrões nas Categorias ')
        for style, style_color in enumerate(colors):
            plt.scatter(txdata[x][style], txdata[y][style], s=80, color=style_color)
        plt.legend([plot for plot in Z.TXT_TYPES], ncol=5, bbox_to_anchor=(0, 1, 1, 3),
                   loc=3, borderaxespad=1.8, mode="expand")
        plt.grid(True)
        plt.subplots_adjust(bottom=0.08, left=.05, right=.96, top=.9, hspace=.35)
        # fig1.savefig("delta0/%s.jpg" % "_".join(u_name.split()))
        plt.show()
        # plt.show()

    def histo_plot(self, yaxis, labels):
        _ = self
        print("yaxis", list(zip(yaxis, list("rgbmyc"))))
        xaxis = arange(0.0, len(labels), 1.0)
        # xaxis = [1.0 * x for x in range(0, len(labels))]
        fig = plt.figure()
        fig.suptitle('Corpora Pattern Count', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.92, left=0.05, right=0.98, bottom=0.08)
        ax.set_xticklabels(labels)
        plt.xticks(xaxis)
        # axes = plt.gca()
        # axes.set_ylim([COUNT_MIN, COUNT_MAX])
        ax.set_xlabel('patterns')
        ax.set_ylabel('count')
        plt.xticks(rotation=90)
        ax.bar(xaxis, yaxis, width=0.6, color="blue")
        show()

    def histo_mplt(self, yaxis, labels):
        _ = self
        print("yaxis", list(zip(yaxis, list("rgbmyc"))))
        xaxis = arange(0.0, len(labels), 1.0)
        # xaxis = [1.0 * x for x in range(0, len(labels))]
        fig = plt.figure()
        fig.suptitle('Corpora Pattern Count', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.92, left=0.05, right=0.98, bottom=0.08)
        ax.set_xticklabels(labels)
        plt.xticks(xaxis)
        # axes = plt.gca()
        # axes.set_ylim([COUNT_MIN, COUNT_MAX])
        ax.set_xlabel('patterns')
        ax.set_ylabel('count')
        plt.xticks(rotation=90)
        # ax.bar(xaxis + 0.2 * xoff, yaxis, width=0.2, color="blue")
        for xoff, (ydata, ycolor) in enumerate(zip(yaxis, list("rgbmyc"))):
            ax.bar(xaxis+0.2*xoff, ydata, width=0.2, color=ycolor)
        show()

    def histo_count(self, *_):
        h_count = list(zip(*[(x, len(c.fragments)) for x, c in enumerate(self.corpora.fragments)]))
        bar(h_count[0], h_count[1])
        show()

    def show_sample(self, *_):
        for t in self.corpora.__repr__().split("\\n"):
            t = t.replace("\\t", "\t")
            t = t.replace("\\", "")
            t = t.replace("x1b[1;", "\033[1;")
            print("{}\n".format(t))

    def survey_major_ordered_absolute_pattern_count(self):
        survey = [(pattern, len(hosts)) for pattern, hosts in self.items()]
        return self.format_data_for_plotting(survey)

    def survey_ordered_pattern_dispersion_across_texts(self, threshold=3.0):
        pattern_across_texts = {pattern: [sentence.parent for sentence in hosts] for pattern, hosts in self.items()}
        self.pattern_count_by_text = {pattern: {text: texts.count(text) for text in self.corpora.fragments}
                                      for pattern, texts in pattern_across_texts.items()}
        survey = [(pattern, [max(texts.count(text) for text in set(texts))
                             - min(texts.count(text) for text in set(texts))])
                  for pattern, texts in pattern_across_texts.items()]
        survey = [(pattern, dispersion) for pattern, dispersion in survey if dispersion[0] >= threshold]
        survey.sort(key=operator.itemgetter(1), reverse=True)
        self.selected_patterns = [patt for patt, count in survey][1:]
        return self.format_data_for_plotting(survey[1:])

    def survey_given_pattern_count_across_texts(self):
        self.survey_ordered_pattern_dispersion_across_texts()
        pattern_across_texts = {pattern: [sentence.parent for sentence in self.word_pattern[pattern]]
                                for pattern in self.selected_patterns}
        pattern_dict_across_ = {pattern: [len([sentence.parent.kind
                                          for sentence in self.word_pattern[pattern] if sentence.parent.kind == kind])
                                          for kind in Z.TXT_TYPES]
                                for pattern in self.selected_patterns}
        all_marked_texts = set([text for _, texts in pattern_across_texts.items() for text in texts])
        print("all_marked_texts", [t.name[:8] for t in all_marked_texts])
        for p, t in pattern_dict_across_.items():
            print(p, t)
        survey = [[[texts_in_kind.count(text) if text.kind == text_kind else 0
                    for text in all_marked_texts]
                   for text_kind in Z.TXT_TYPES]
                  for pattern, texts_in_kind in pattern_across_texts.items()]
        # survey.sort(key=operator.itemgetter(1))
        for s in survey:
            print("given_", s)
        return survey

    def survey_ordered_pattern_count_across_texts(self, func=max):
        pattern_across_texts = {pattern: [sentence.parent for sentence in hosts] for pattern, hosts in self.items()}
        survey = [(pattern, [func(texts.count(text) for text in set(texts))])
                  for pattern, texts in pattern_across_texts.items()]
        # survey.sort(key=operator.itemgetter(1))
        return self.format_data_for_plotting(survey)

    def arrange_data_for_learning(self):
        def clean(name):
            return "".join(letter for letter in name if letter not in "\x1b[0123456789m[1;")
        _ = self.survey_ordered_pattern_dispersion_across_texts()
        data = self.selected_patterns
        pattern_across_texts = {pattern: [sentence.parent for sentence in hosts] for pattern, hosts in self.items()
                                if pattern in data}
        # texts_with_patterns
        pat_count_in_texts = {pattern: [sum(1 for text in texts if text is given)
                                        for given in self.corpora.fragments]
                              for pattern, texts in pattern_across_texts.items()}
        just_count_in_texts = [[clean(pattern)]+[sum(1 for text in texts if text is given)
                                                 for given in self.corpora.fragments]
                               if pattern else ['class']+[given.kind
                                                          for given in self.corpora.fragments]
                               for pattern, texts in list(pattern_across_texts.items())+[(0, 0)]]
        for pat, host in pat_count_in_texts.items():
            print(pat, host)
        pat_plus_pat_count = list(zip(*just_count_in_texts))
        pat_cnt = len(pat_plus_pat_count[0])-1
        table_header = [
            ['n']+list(pat_plus_pat_count[0]),
            list('c'+'c'*pat_cnt+"d"),
            list('m'+' '*pat_cnt+"c")]
        table_body = [[name]+list(row) for name, row in enumerate(pat_plus_pat_count[1:])]
        table_file = table_header + table_body
        for line in table_file:
            print(line)
        from csv import writer
        with open(self.out_text, "w") as tab_file:
            csvwriter= writer(tab_file, delimiter='\t')
            for row in table_file:
                csvwriter.writerow(row)


    @staticmethod
    def format_data_for_plotting(survey):
        survey = survey[:Z.PATT_CUT]
        labels = ["".join(letter for letter in name if letter not in "\x1b[0123456789m[1;")
                  for name, _ in survey]
        yaxis = h_count = [count for _, count in survey]
        print(h_count)
        yaxis, _ = list(zip(*h_count)) if len(h_count[0])>1 else [[h[0] for h in h_count], []]
        print(yaxis, labels)
        return yaxis, labels

    def items(self):
        return self.word_pattern.items()


insc = Inscription()


class Run:
    HISTO_COUNT, HOW_SAMPLE, ABSOLUTE, MAJOR_ACROSS, MINOR_ACROSS, DIPERS_ACROSS, SCATTER2D = [
        insc.histo_count,
        insc.show_sample,
        lambda *_: insc.histo_plot(*insc.survey_major_ordered_absolute_pattern_count()),
        lambda *_: insc.histo_plot(*insc.survey_ordered_pattern_count_across_texts()),
        lambda *_: insc.histo_plot(*insc.survey_ordered_pattern_count_across_texts(min)),
        lambda *_: insc.histo_plot(*insc.survey_ordered_pattern_dispersion_across_texts()),
        lambda x=0, y=1: insc.scatter_plot(insc.survey_given_pattern_count_across_texts(), x, y)
    ]


if __name__ == '__main__':
    insc.arrange_data_for_learning()
    # Run.SCATTER2D(1, 2)
