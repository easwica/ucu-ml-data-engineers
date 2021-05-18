import pickle


class TxtReader:
    @staticmethod
    def read(path):
        with open(path, mode='r') as fp:
            txt = fp.read()
        return txt
