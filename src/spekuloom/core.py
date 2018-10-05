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

from matplotlib.pyplot import bar, show
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from spekuloom.util import Mont

PATT_CUT = 100
CLAZ = Mont().mont_symbol_pt()
KIND_CROP_PT = 3
WINDOW = 4
WINDOW_OVERLAP = WINDOW - 1
COUNT_MIN = -2
COUNT_MAX = 20
PATTERNS = 400
CNT_CUT = 7.5
TXT_OFF = 800
TXT_CUT = 8000
FACTOR = 10
GEN_CNT = 4
TXT_DIR = "/home/carlo/Documentos/dev/spekuloom/src/Spekuloom/"
TXT_TYPES = "basico  intermediario  transitorio".split()

tagger = pickle.load(open("tagger.pkl", "rb"))


class Fragment:
    """
    Text available in corpora.
    """
    WORD_PATTERN = None

    def __init__(self, text, kind="", parent=None):
        self._parent = parent
        self._text = text
        self._kind = kind
        self._fragments = self.tokenize()
        self._pattern = self.symbolize()

    @property
    def word_pattern(self):
        return Fragment.WORD_PATTERN

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
        h_count = [(x, count, label) for x, (count, label) in enumerate(sorted(h_count, reverse=True))][:PATT_CUT]
        print(h_count)
        xaxis, yaxis, labels = list(zip(*h_count))
        print(xaxis, yaxis, labels)
        return xaxis, yaxis, labels


class Corpora(Fragment):
    """
        Collection of texts representing a literate language
    """
    def __init__(self, text, kind=""):
        # print("Text", text[:4], kind)
        Fragment.WORD_PATTERN = Pattern()
        super().__init__(text, kind)

    def tokenize(self):
        self._kind = []
        _texts = list()
        for x in TXT_TYPES:
            self._kind.extend([x] * GEN_CNT)
        _corp = [TextC(
            open(os.path.join(os.path.join(TXT_DIR, textype), _text), "r").read()[TXT_OFF:TXT_CUT], kind=textype)
                 for textype in TXT_TYPES
                 for _, _dir, _texts in os.walk(os.path.join(TXT_DIR, textype))
                 for _, _text in zip(range(GEN_CNT), _texts)
                 ]
        return _corp

    def survey_major_ordered_absolute_pattern_count(self):
        return self.word_pattern.survey_major_ordered_absolute_pattern_count()

    def survey_major_ordered_pattern_count_across_texts(self):
        return self.word_pattern.survey_ordered_pattern_count_across_texts()

    def survey_minor_ordered_pattern_count_across_texts(self):
        return self.word_pattern.survey_ordered_pattern_count_across_texts(func=min)


class TextC(Fragment):
    """
    Text available in corpora.
    """
    def __init__(self, text, kind=""):
        # print("Text", text[:4], kind)
        super().__init__(text, kind)

    def symbolize(self):
        fragments = [pattern for sentence in self.fragments for pattern in sentence.pattern]
        window_of_n_words = [fragments[offset:] for offset in range(WINDOW)]
        window_of_n_words.append(list(range(WINDOW_OVERLAP))*1000)
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
        window_of_n_words = [self._fragments[offset:] for offset in range(WINDOW)]
        window_of_n_words.append(list(range(WINDOW_OVERLAP))*1000)
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
        return CLAZ[self._kind[:KIND_CROP_PT]] if self._kind[:KIND_CROP_PT] in CLAZ else None


class Inscription:
    def __init__(self):
        self.corpora = Corpora("portuguese")

    def histo_plot(self, xaxis, yaxis, labels):
        _ = self
        fig = plt.figure()
        fig.suptitle('Corpora Pattern Count', fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.92, left=0.05, right=0.98, bottom=0.08)
        ax.set_xticklabels(labels)
        plt.xticks(list(range(0, len(labels))))
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


def main():
    # histo_count()
    # show_sample()
    inscription = Inscription()
    # inscription.histo_plot(*inscription.corpora.survey_major_ordered_absolute_pattern_count())
    inscription.histo_plot(*inscription.corpora.survey_major_ordered_pattern_count_across_texts())


if __name__ == '__main__':
    main()
