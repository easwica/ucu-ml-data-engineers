class CRFEncoder:
    @staticmethod
    def _group_sentences(sentences):
        grouped, sentence = [], []
        for word in sentences:
            sentence.append(word)
            if word[0] != '.':
                continue
            grouped.append(sentence)
            sentence = []
        return grouped

    def encode(self, data):
        sentences = [row.split() for row in data.split('\n') if row.split()]
        grouped = self._group_sentences(sentences)

        sentences_features, labels = [], []
        for sentence in grouped:
            sentence_ftrs, lbls = [], []
            for i in range(len(sentence)):
                word = sentence[i][0]
                postag = sentence[i][1]
                lbls.append(sentence[i][2])

                features = {
                    'bias': 1.0,
                    'word.lower()': word.lower(),
                    'word[-3:]': word[-3:],
                    'word[-2:]': word[-2:],
                    'word.isupper()': word.isupper(),
                    'word.istitle()': word.istitle(),
                    'word.isdigit()': word.isdigit(),
                    'postag': postag,
                    'postag[:2]': postag[:2],
                }
                if i > 0:
                    word1 = sentences[i - 1][0]
                    postag1 = sentences[i - 1][1]
                    features.update({
                        '-1:word.lower()': word1.lower(),
                        '-1:word.istitle()': word1.istitle(),
                        '-1:word.isupper()': word1.isupper(),
                        '-1:postag': postag1,
                        '-1:postag[:2]': postag1[:2],
                    })
                else:
                    features['BOS'] = True

                if i < len(sentences) - 1:
                    word1 = sentences[i + 1][0]
                    postag1 = sentences[i + 1][1]
                    features.update({
                        '+1:word.lower()': word1.lower(),
                        '+1:word.istitle()': word1.istitle(),
                        '+1:word.isupper()': word1.isupper(),
                        '+1:postag': postag1,
                        '+1:postag[:2]': postag1[:2],
                    })
                else:
                    features['EOS'] = True
                sentence_ftrs.append(features)
            sentences_features.append(sentence_ftrs)
            labels.append(lbls)
        return sentences_features, labels
