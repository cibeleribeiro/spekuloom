import ast

tree = ast.parse('''
import random as rd
from random import randint
fruits = ['grapes', 'mango']
volatil = {1:3}
aset = set(1,2)
name = 'peter'
with open('x') as xf:
    pass

assert isinstance(name, str)
del volatil


class Fruit:
    def __init__(self):
        try:
            z = [x for x in range(1)]
        except Exception as ex:
            raise TypeError(ex)
        finally:
            pass
        return z

    @classmethod
    def go(cls, x, y=1, *args, **kwargs):
        global name
        nonlocal fruits
        z = 0 if y > 1 else -1
        while z in range(2):
            z += 1
            if x :
              continue
            elif z >=3:
              break
        if name is None:
           name = True
           fruits = {k: False for k in u"xy"}
           w = f"{name}"
           ww = (1, 2)
           print(i for i in range(4))
           zw = 1 > 2.0 < 2e1 or 3 == 5 and 3 <= 4 <= 5 or False or not True
        yield z


for fruit in fruits:
    print('{} likes {}'.format(name, fruit))
yy = lambda: 1 * 3 ** 4 + 2 - (5 / 2 // 1) % 3
''')

# print(ast.dump(tree))

# from spekuloom.util import Mont
#
# d = Mont().mont_symbol_pt()
# for k, v in d.items():
#     print(k, v)
TOKENS = dict(
    Module='▲', Assign='◉', Name='▲', Store='▣', List='▣', Str='◱', Load='◬', ClassDef='◪',
    FunctionDef='◨', arguments='▵', arg='▵', Expr='◉', ListComp='◈', comprehension='◈', Call='●',
    Num='◳', For='◠', Attribute='◭', Import='◡', alias='◡', ImportFrom='◡', Dict='▣', With='◡',
    ExceptHandler='◔', Raise='◔', Pass='◐', Return='●', Global='◭', Nonlocal='◭', IfExp='▬', Compare='◎',
    Gt='▱', UnaryOp='◉', USub='▲', While='◠', In='◎', AugAssign='◉', Add='■', If='▬', Continue='◐', GtE='▰',
    Break='◐', Is='◎', NameConstant='▲', DictComp='◈', JoinedStr='◲', FormattedValue='◲', Tuple='▣',
    GeneratorExp='◈', Yield='●', BoolOp='◻', Or='◻', Lt='▱', And='◻', Eq='◻', LtE='▰', Not='◻',
    Assert='◕', Delete='◕', Del='◕', Lambda='◨', BinOp='■', Mult='■', Pow='■', Sub='■', Div='■', FloorDiv='■', Mod='■'
)


class GenV(ast.NodeVisitor):
    def __init__(self):
        self.tokens = {}
        self.gram = ""

    def generic_visit(self, node):
        token_name = type(node).__name__
        self.tokens[token_name] = "\033[1;30m" + u"\u25B2" + "\033[1;0m"
        if token_name in TOKENS:
            self.gram += TOKENS[token_name]
        ast.NodeVisitor.generic_visit(self, node)

    def tokenize(self, text):
        self.visit(ast.parse(text))
        return list(self.gram)


'''
class NodeVisitor(ast.NodeVisitor):
    def visit_Str(self, tree_node):
        print('◱{}'.format(tree_node.s))

    def visit_Name(self, tree_node):
        print('△{}'.format(tree_node.id))

    def visit_Assign(self, tree_node):
        print('◉{}'.format(list(tree_node.targets)[0].id))
        self.visit(tree_node.value)

    def visit_For(self, tree_node):
        print('◠{}'.format(tree_node.target.id))
        [self.visit(node) for node in tree_node.body]

    def visit_Call(self, tree_node):
        print('●{}'.format(tree_node.func.id))
        self.visit(tree_node.func)

    def visit_Attribute(self, tree_node):
        print('◭{}'.format(tree_node.value.s))
        [self.visit(node) for node in tree_node.args]

    def visit_ClassDef(self, tree_node):
        print('◪{}'.format(tree_node.name))
        [self.visit(node) for node in tree_node.body]

    def visit_FunctionDef(self, tree_node):
        print('◨{}'.format(tree_node.name))
        [self.visit(node) for node in tree_node.body]



gv = GenV()
gv.visit(tree)
for tok in gv.tokens:
    print("{}='{}',".format(tok, gv.tokens[tok]), end=' ')

print("toks")

for tok in gv.gram:
    print("{}".format(tok), end='')
'''
"""
noun,▲

False True None▲

name △

numeral ◳

textual ◱
article,
adjective,◭

class def lambda◪

global nonlocal◭
verb, ●

yield return●

continue break pass◐

try raise◔

del assert import◕
adverb,○

finally except○

in is◎
preposition,◡

from with as◡

while for◠
conjuction,▬

or and not◻

if else elif▬
"""
