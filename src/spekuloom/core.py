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
import os
import pickle
from enum import Enum, auto

import matplotlib.pyplot as plt
from matplotlib.pyplot import bar, show
from nltk.tokenize import sent_tokenize, word_tokenize

from spekuloom.util import Mont


class Z:
    GEN_CNT: int = 4
    PATT_CUT: int = 200
    CLAZ: dict = Mont().mont_symbol_pt()
    KIND_CROP_PT: int = 3
    WINDOW: int = 4
    WINDOW_OVERLAP: int = 3
    COUNT_MIN: int = -2
    COUNT_MAX: int = 20
    PATTERNS: int = 400
    CNT_CUT: int = 7.5
    TXT_OFF: int = 800
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

    def __init__(self, text, kind="", parent=None):
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

    def __init__(self, text, kind=""):
        # print("Text", text[:4], kind)
        Fragment.PATTERN[Agregator.WORD_PATTERN] = Pattern()
        super().__init__(text, kind)

    def tokenize(self):
        gen_cnt, text_types = Z.GEN_CNT, Z.TXT_TYPES
        self._kind = []
        _texts = list()
        for x in text_types:
            self._kind.extend([x] * gen_cnt)
        _corp = [TextC(
            open(os.path.join(os.path.join(Z.TXT_DIR, textype), _text), "r").read()[Z.TXT_OFF:Z.TXT_CUT], kind=textype)
            for textype in text_types
            for _, _dir, _texts in os.walk(os.path.join(Z.TXT_DIR, textype))
            for _, _text in zip(range(gen_cnt), _texts)
        ]
        return _corp


class TextC(Fragment):
    """
    Text available in corpora.
    """

    def __init__(self, text, kind=""):
        # print("Text", text[:4], kind)
        super().__init__(text, kind)

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
    def __init__(self, text="portuguese"):
        super().__init__(text)
        self.corpora = Corpora(text)

    def histo_plot(self, yaxis, labels):
        _ = self
        xaxis = list(range(0, len(labels)))
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

        ax.bar(xaxis, yaxis)
        show()

    def histo_count(self):
        h_count = list(zip(*[(x, len(c.fragments)) for x, c in enumerate(self.corpora.fragments)]))
        bar(h_count[0], h_count[1])
        show()

    def show_sample(self):
        for t in self.corpora.__repr__().split("\\n"):
            t = t.replace("\\t", "\t")
            t = t.replace("\\", "")
            t = t.replace("x1b[1;", "\033[1;")
            print("{}\n".format(t))

    def survey_major_ordered_absolute_pattern_count(self):
        survey = [(pattern, len(hosts)) for pattern, hosts in self.items()]
        return self.format_data_for_plotting(survey)

    def survey_ordered_pattern_count_across_texts(self, func=max):
        pattern_across_texts = {pattern: [sentence.parent for sentence in hosts] for pattern, hosts in self.items()}
        survey = [(pattern, func(texts.count(text) for text in set(texts)))
                  for pattern, texts in pattern_across_texts.items()]
        return self.format_data_for_plotting(survey)

    @staticmethod
    def format_data_for_plotting(survey):
        h_count = [(count, "".join(letter for letter in name if letter not in "\x1b[0123456789m[1;"))
                   for name, count in survey]
        h_count = [(count, label) for x, (count, label) in enumerate(sorted(h_count, reverse=True))][:Z.PATT_CUT]
        print(h_count)
        yaxis, labels = list(zip(*h_count))
        print(yaxis, labels)
        return yaxis, labels

    def items(self):
        return self.word_pattern.items()


insc = Inscription()


class Run:
    HISTO_COUNT, HOW_SAMPLE, ABSOLUTE, MAJOR_ACROSS, MINOR_ACROSS = [
        insc.histo_count,
        insc.show_sample,
        lambda: insc.histo_plot(*insc.survey_major_ordered_absolute_pattern_count()),
        lambda: insc.histo_plot(*insc.survey_ordered_pattern_count_across_texts()),
        lambda: insc.histo_plot(*insc.survey_ordered_pattern_count_across_texts(min))
    ]


if __name__ == '__main__':
    Run.MAJOR_ACROSS()
