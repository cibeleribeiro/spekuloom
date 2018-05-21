#import nltk
#from nltk import *
#from nltk.book import *
#from spekuloom.word_class import tagset
import mechanicalsoup
from mechanicalsoup import LinkNotFoundError
from time import sleep


class Mont:
    def __init__(self):
        #self.text = text2
        #self.word_classes()
        #self.mont_symbol()

        browser = mechanicalsoup.StatefulBrowser()
        browser.open("https://www.gutenberg.org/wiki/Children%27s_Anthologies_(Bookshelf)")
        cats = [ln for ln in browser.links(url_regex='.*www.gutenberg.org/ebooks/.*')]
        for vol, _link in enumerate(cats):
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
                bk = open('/opt/iso/books/Bk{:03}_{}'.format(vol, filename), 'w')
                content = browser.get('https:'+sub_cats[0]).content
                bk.write(bytearray(content).decode("utf-8"))
                # bk.write(bytearray(content.encode("UTF8"))).decode("utf-8")
                bk.close()
                sleep(100)
            except LinkNotFoundError:
                pass
        #sub_cats = {ct: '' if print(ct) else browser.follow_link(ct) for ct in cats}

        #print(sub_cats)

        # Fill-in the search form
        # browser.select_form('#search_form_homepage')
        # browser["q"] = "MechanicalSoup"
        # browser.submit_selected()

        # Display the results
        # for link in browser.get_current_page().find_all('a'):
        #     print(link.text, '->') # , link.attrs['href'])

    def word_classes(self):
        print(tagset)

    def mont_symbol(self):

        #classes=dict(noun,article,adjective, verb, adverb, preposition, conjuction,interjection, pronoun)
        classes=dict(NN = "\033[1;30m"+u"\u25B2"+"\033[1;0m",
                     DT = "\033[1;36m"+u"\u25B4"+"\033[1;0m",
                     JJ = "\033[1;34m"+u"\u25ED"+"\033[1;0m",
                     VB = "\033[1;31m"+u"\u25CF"+"\033[1;0m",
                     MD = "\033[1;31m"+u"\u25CF"+"\033[1;0m",
                     RB = "\033[1;31m"+u"\u25CB"+"\033[1;0m",
                     IN = "\033[1;32m"+u"\u25E1"+"\033[1;0m",
                     CC = "\033[1;35m"+u"\u25AC"+"\033[1;0m",
                     UH = "\033[1;33m"+u"\u25C8"+"\033[1;0m",
                     PR = "\033[1;35m"+u"\u25BB"+"\033[1;0m",
                     ZZ = "\033[1;33m"+u"\u16E4"+"\033[1;0m",
                     WD = "\033[1;36m"+u"\u16DE"+"\033[1;0m",
                     WP = "\033[1;35m"+u"\u16De"+"\033[1;0m",
                     WR = "\033[1;31m"+u"\u16De"+"\033[1;0m",
                     TO = "\033[1;32m"+u"\u16E0"+"\033[1;0m",
                     RP = "\033[1;35m"+u"\u16B9"+"\033[1;0m",
                     PD = "\033[1;36m"+u"\u16CF"+"\033[1;0m",
                     PO = "\033[1;30m"+u"\u16CD"+"\033[1;0m",
                     EX = "\033[1;33m"+u"\u16D8"+"\033[1;0m",
                     CD = "\033[1;34m"+u"\u16E5"+"\033[1;0m")
        #for a, b in classes.items():print(b)
        return classes




Mont()

