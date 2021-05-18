import pickle


class PickleWriter:
    @staticmethod
    def write(content, pth):
        with open(pth, mode='wb') as fp:
            pickle.dump(content, fp)
