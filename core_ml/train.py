from collections import Counter
from itertools import chain

from sklearn_crfsuite import CRF, metrics

from core_ml.encoders import CRFEncoder
from core_ml.readers import TxtReader
from core_ml.writers import PickleWriter


class CRFTrainer:
    def __init__(self):
        self._encoder = CRFEncoder()
        self._model = CRF()
        self._pickle_writer = PickleWriter()
        self._txt_reader = TxtReader()

    def train(self, train_data):
        train_features, labels = self._encoder.encode(train_data)
        self._model.fit(train_features, labels)

        print('Distribution of labels in train data: ', Counter(list(chain(*labels))))
        
    def evaluate(self, test_data):
        test_features, true_labels = self._encoder.encode(test_data)
        print('Distribution of labels in test data: ', Counter(list(chain(*true_labels))))

        labels = list(self._model.classes_)
        predicted = self._model.predict(test_features)
        print('f1 score: ', metrics.flat_f1_score(
            true_labels, predicted, average='weighted', labels=labels
        ))

    def save(self, pth):
        self._pickle_writer.write(self._model, pth)

    def update(self, new_data, new_pth):
        old_data = self._txt_reader.read('/Users/easwica/ucu-ml-data-engineers/datasets/conll2000/train.txt')
        data = old_data + new_data

        self.train(data)
        self.save(new_pth)


if __name__ == '__main__':
    trainer = CRFTrainer()
    train_data = TxtReader().read('/Users/easwica/ucu-ml-data-engineers/datasets/conll2000/train.txt')
    test_data = TxtReader().read('/Users/easwica/ucu-ml-data-engineers/datasets/conll2000/test.txt')

    trainer.train(train_data)
    trainer.evaluate(test_data)

    trainer.save('/Users/easwica/ucu-ml-data-engineers/core_ml/model.pkl')
    trainer.update(train_data, '/Users/easwica/ucu-ml-data-engineers/core_ml/model.pkl')
