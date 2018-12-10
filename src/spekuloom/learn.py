from random import sample

from Orange import data, classification


class Learn:
    def __init__(self):
        self.data = data.Table("estilo")
        self.learner = classification.NaiveBayesLearner()

    def probs(self, learn):
        target_class = 2
        print("Probabilities for %s:" % self.data.domain.class_var.values[target_class])
        classifier = self.learner(learn)
        probabilities = classifier(self.data, 1)
        for p, d in zip(probabilities[:], self.data[:]):
            print(p[target_class], d.get_class())

    def test(self, learn, test):
        classifier = self.learner(learn)
        c_values = self.data.domain.class_var.values
        for d in test:
            c = classifier(d)
            if c_values[int(c[0])] != d.get_class():
                print("{}:{}, originally {}".format(d['n'], c_values[int(c[0])],
                                                 d.get_class()))

    def get_sample(self, sampler=22):
        adata = sample(self.data, sampler)
        return data.Table.from_list(domain=self.data.domain, rows=adata)

    def sampler(self):
        for samp in range(10):
            self.test(self.get_sample(), self.data)


if __name__ == '__main__':
    Learn().sampler()
