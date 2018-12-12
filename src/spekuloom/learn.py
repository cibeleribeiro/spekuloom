from random import sample

from Orange import data, classification
from numpy import average

NUNAME = ",basico,intermediario,transitorio".split(',')
BIN = [0, 1]
NEW_CLASS = {(a, b, c): NUNAME[a]+NUNAME[b*2]+NUNAME[c*3] for a in BIN for b in BIN for c in BIN}
# print(NEW_CLASS)


class Learn:
    def __init__(self):
        self.data = data.Table("estilo")
        self.learner = classification.NaiveBayesLearner()
        self.mistakes = {}

    def probs(self):
        target_class = 2
        print("Probabilities for %s:" % self.data.domain.class_var.values[target_class])
        classifier = self.learner(self.data)
        probabilities = classifier(self.data, 1)[0]
        for p, d in zip(probabilities[:], self.data[:]):
            print(p[0], p[1], p[2], d.get_class())

    def test(self, learn, test):
        classifier = self.learner(learn)
        c_values = self.data.domain.class_var.values
        for d in test:
            c = classifier(d)
            # probabilities = [classifier(d, tc) for tc in range(3)]
            probabilities = classifier(d, 1)[0]
            post_class = c_values[int(c[0])]
            # new_class = NEW_CLASS[tuple([1 if prob_class > 0.44 else 0 for prob_class in probabilities])]
            new_class = d.get_class()
            # if post_class != d.get_class() and new_class in NUNAME:
            if post_class != d.get_class() and new_class in NUNAME:
                mistakes = self.mistakes[int(d['n'])] if int(d['n']) in self.mistakes\
                    else dict(data=d, prob=[], nclass=new_class)
                mistakes['prob'] += [probabilities]
                self.mistakes[int(d['n'])] = mistakes
                print("{}:{}, originally {}".format(d['n'], post_class, d.get_class()))

    def get_sample(self, sampler=22):
        adata = sample(self.data, sampler)
        return data.Table.from_list(domain=self.data.domain, rows=adata)

    def sampler(self):
        for _ in range(20):
            self.test(self.get_sample(), self.data)
        for m in self.mistakes.values():
            print(m['prob'])
        for name, mst in self.mistakes.items():
            class_probs = zip(*mst['prob'])
            mst['prob'] = probabilities = [average(p) for p in class_probs]
            new_class = NEW_CLASS[tuple([1 if prob_class > 0.75 else 0 for prob_class in probabilities])]
            mst['nclass'] = new_class if new_class in NUNAME[1:] else mst['nclass']
        for m in self.mistakes.values():
            print(m)
        '''
        new_domain = list(set(a['nclass'] for a in self.mistakes.values()))
        class_descriptors = [data.DiscreteVariable.make(clazz) for clazz in list(new_domain)]
        # new_domain = data.Domain(new_domain, class_descriptors)  # , source=self.data.domain)
        new_domain = data.Domain(
            class_descriptors, data.DiscreteVariable("kind", tuple(new_domain)))  # , source=self.data.domain)
        # self.data.domain = new_domain
        print(new_domain, self.data.domain.class_var.values)
        '''

        for entry, class_data in self.mistakes.items():
            new_class = class_data['nclass']
            if new_class in self.data.domain.class_var.values:
                self.data[int(entry)].set_class(new_class)
            # else:
            #     self.data.domain.append(new_class)
            #     self.data[int(entry)].set_class(new_class)
            b, i, t = class_data['prob']
            print("e:{} -> p0:{}, p1:{}, p2:{}, new:{}".format(entry, b, i, t, class_data['nclass']))

        for a_data in self.data:
            print(a_data, a_data.get_class())

    def arrange_data_for_learning(self):
        pat_cnt = len(self.data[0])-1
        table_header = [
            ['n']+list(pat.name for pat in self.data.domain.variables),
            list('c'+'c'*pat_cnt+"d"),
            list('m'+' '*pat_cnt+"c")]
        table_body = [
            [name]+list(row)[:-1] +
            [row.get_class().value if name not in self.mistakes.keys() else self.mistakes[name]['nclass']]
            for name, row in enumerate(self.data)]
        print(self.mistakes)
        table_file = table_header + table_body
        for line in table_file:
            print(line)
        from csv import writer
        with open("nestilo.tab", "w") as tab_file:
            csvwriter = writer(tab_file, delimiter='\t')
            for row in table_file:
                csvwriter.writerow(row)


if __name__ == '__main__':
    learn = Learn()
    learn.sampler()
    learn.arrange_data_for_learning()
    # Learn().probs()
    pass
