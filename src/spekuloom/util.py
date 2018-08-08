# import nltk
# from nltk import *
# from nltk.book import *
# from spekuloom.word_class import tagset
from time import sleep

import mechanicalsoup
from mechanicalsoup import LinkNotFoundError


class Mont:
    def __init__(self):
        # self.text = text2
        # self.word_classes()
        # self.mont_symbol()
        self.books = {}
        pass

    def get_books(self):
        browser = mechanicalsoup.StatefulBrowser()
        browser.open("https://www.gutenberg.org/wiki/Children%27s_Anthologies_(Bookshelf)")
        cats = [ln for ln in browser.links(url_regex='.*www.gutenberg.org/ebooks/.*')]
        for vol, _link in enumerate(cats[14:]):
            vol += 14
            print("browser.follow_link", _link.text, _link.attrs['href'])
            filename = _link.text
            try:
                browser.follow_link(_link)
                # pg = browser.get_current_page()
                # [print(">>>sub browser.a", a) for a in pg.find_all('a')]
                sub_link = '.*//www.gutenberg.org/ebooks/.*.txt.utf-8'
                sub_cats = [ln.attrs['href'] for ln in browser.links(url_regex=sub_link)]
                [print(">>>sub browser.link", ct) for ct in sub_cats]
                # browser.follow_link(sub_cats[0])
                # print(browser.get('https:'+sub_cats[0]).content)
                title = '/opt/iso/books/Bk{:03}_{}'.format(vol, filename)
                bk = open('/opt/iso/books/Bk{:03}_{}'.format(vol, filename), 'w')
                content = browser.get('https:' + sub_cats[0]).content
                content = bytearray(content).decode("utf-8")
                self.books[title] = content
                bk.write(content)
                # bk.write(bytearray(content.encode("UTF8"))).decode("utf-8")
                bk.close()
                sleep(100)
            except (LinkNotFoundError, IndexError):
                pass
        # sub_cats = {ct: '' if print(ct) else browser.follow_link(ct) for ct in cats}

        # print(sub_cats)

        # Fill-in the search form
        # browser.select_form('#search_form_homepage')
        # browser["q"] = "MechanicalSoup"
        # browser.submit_selected()

        # Display the results
        # for link in browser.get_current_page().find_all('a'):
        #     print(link.text, '->') # , link.attrs['href'])

    def load_books(self):
        browser = mechanicalsoup.StatefulBrowser()
        browser.open("https://www.gutenberg.org/wiki/Children%27s_Anthologies_(Bookshelf)")
        cats = [ln for ln in browser.links(url_regex='.*www.gutenberg.org/ebooks/.*')]
        for vol, _link in enumerate(cats):
            filename = _link.text
            try:
                title = '/opt/iso/books/Bk{:03}_{}'.format(vol, filename)
                bk = open(title, 'r')
                self.books[title] = bk.read()
                # bk.write(bytearray(content.encode("UTF8"))).decode("utf-8")
                bk.close()
            except FileNotFoundError:
                pass
        return self.books

    def word_classes(self):
        print('tagset')

    def mont_symbol(self):

        # classes=dict(noun,article,adjective, verb, adverb, preposition, conjuction,interjection, pronoun)
        classes = dict(NN="\033[1;30m" + u"\u25B2" + "\033[1;0m",
                       DT="\033[1;36m" + u"\u25B4" + "\033[1;0m",
                       JJ="\033[1;34m" + u"\u25ED" + "\033[1;0m",
                       VB="\033[1;31m" + u"\u25CF" + "\033[1;0m",
                       MD="\033[1;31m" + u"\u25CF" + "\033[1;0m",
                       RB="\033[1;31m" + u"\u25CB" + "\033[1;0m",
                       IN="\033[1;32m" + u"\u25E1" + "\033[1;0m",
                       CC="\033[1;35m" + u"\u25AC" + "\033[1;0m",
                       UH="\033[1;33m" + u"\u25C8" + "\033[1;0m",
                       PR="\033[1;35m" + u"\u25BB" + "\033[1;0m",
                       ZZ="\033[1;33m" + u"\u25A4" + "\033[1;0m",
                       WD="\033[1;36m" + u"\u25A5" + "\033[1;0m",
                       WP="\033[1;35m" + u"\u25A6" + "\033[1;0m",
                       WR="\033[1;31m" + u"\u25A7" + "\033[1;0m",
                       TO="\033[1;32m" + u"\u25A8" + "\033[1;0m",
                       RP="\033[1;35m" + u"\u25A9" + "\033[1;0m",
                       PD="\033[1;36m" + u"\u25F0" + "\033[1;0m",
                       PO="\033[1;30m" + u"\u25F1" + "\033[1;0m",
                       EX="\033[1;33m" + u"\u25F2" + "\033[1;0m",
                       CD="\033[1;34m" + u"\u25F3" + "\033[1;0m")
        # for a, b in classes.items():print(b)
        return classes


if __name__ == '__main__':
    b = Mont().load_books()
    [print(t, c[10000: 10100]) for t, c in b.items()]
