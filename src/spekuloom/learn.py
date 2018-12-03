from Orange import data, classification


class Learn:
    def __init__(self):
        self.data = data.Table("estilo")
        _nb = classification.NaiveBayesLearner()
        classifier = _nb(self.data)
        target_class = 2
        print("Probabilities for %s:" % self.data.domain.class_var.values[target_class])
        probabilities = classifier(self.data, 1)
        for p, d in zip(probabilities[:], self.data[:]):
            print(p[target_class], d.get_class())


if __name__ == '__main__':
    Learn()
