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
# from nltk import *
import os
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize

WINDOW = 4
COUNT_MIN = -2
COUNT_MAX = 20
PATTERNS = 400
CNT_CUT = 7.5
FACTOR = 10
GEN_CNT = 4
TXT_DIR = "/home/carlo/Documentos/dev/spekuloom/src/Spekuloom/"
TXT_TYPES = "basico  intermediario  transitorio".split()

tagger = pickle.load(open("tagger.pkl", "rb"))


class Fragment:
    """
    Text available in corpora.
    """
    def __init__(self, text, kind=""):
        self._text = text
        self._kind = kind
        self._fragments = self.tokenize()

    def tokenize(self):
        return [Sentence(a_sentence) for a_sentence in sent_tokenize(self._text, "portuguese")]

    def __repr__(self):
        return "<{}>: {}\n\t{}".format(type(self).__name__,
                                       self._text[:12], [frag.__repr__() for frag in self._fragments][:2])


class Corpora(Fragment):
    """
        Collection of texts representing a literate language
    """
    def tokenize(self):
        self._kind = []
        _texts = list()
        for x in TXT_TYPES:
            self._kind.extend([x] * GEN_CNT)
        _corp = [TextC(open(os.path.join(os.path.join(TXT_DIR, textype), _text), "r").read(), kind=textype)
                 for textype in TXT_TYPES
                 for _, _dir, _texts in os.walk(os.path.join(TXT_DIR, textype))
                 for _, _text in zip(range(GEN_CNT), _texts)
                 ]
        return _corp


class TextC(Fragment):
    """
    Text available in corpora.
    """
    def __init__(self, text, kind=""):
        # print("Text", text[:4], kind)
        super().__init__(text, kind)


class Sentence(Fragment):
    """
    Sentences broken by tokenizer.
    """
    def __init__(self, text, kind=""):
        super().__init__(text, kind)

    def tokenize(self):
        return [Word(a_sentence) for a_sentence in word_tokenize(self._text, "portuguese")]


class Word(Fragment):
    """
    Words broken by tokenizer.
    """
    def __init__(self, text, kind=""):
        super().__init__(text, kind)

    def tokenize(self):
        _ = self
        return []


if __name__ == '__main__':
    for t in Corpora("portuguese").__repr__().split("\\n"):
        print("{}\n".format(t))
