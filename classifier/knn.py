# import time
import numpy as np
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random
from statistics import mode


class Indexer:
    def __init__(self):
        self.train_set = None
        self.test_set = None
        self.label = None
        self.label_id = None

        self.train_label = None
        self.train_label_id = None
        self.test_label = None
        self.test_label_id = None
        self.classes_names = ["tennis", "football", "athletics", "cricket", "rugby"]

        self.split = 0.7
        self.filter = 0
        self.rand_seed = 11

        self.vsm = None
        self.features = {}
        self.vsm_init_size = 14000
        self.lemmatize = WordNetLemmatizer()

        self.result = None
        self.param = None
        self.accuracy = None

    def randomize_array(self):
        """randomize entries to get better training/testing set distribution and split them"""

        random.seed(self.rand_seed)
        for i in range(0, self.vsm.shape[0]):
            rand = random.randint(0, self.vsm.shape[0] - 1)
            self.vsm[[i, rand]] = self.vsm[[rand, i]]
            self.label[[i, rand]] = self.label[[rand, i]]
            self.label_id[[i, rand]] = self.label_id[[rand, i]]

    def read_file(self, docs_path, file_path):
        """read vsm and other data files, if they exist, else read from training/testing documents and create vsm model"""

        if (
            os.path.isfile(file_path + "result.npy")
            and os.path.isfile(file_path + "param.npy")
            and os.path.isfile(file_path + "label.npy")
            and os.path.isfile(file_path + "label_id.npy")
        ):
            self.result = np.load(file_path + "result.npy")
            self.param = np.load(file_path + "param.npy")
            self.label = np.load(file_path + "label.npy")
            self.label_id = np.load(file_path + "label_id.npy")

            # split train and test labels
            self.train_label, self.test_label = np.split(
                self.label, [int(len(self.label) * self.split)]
            )
            self.train_label_id, self.test_label_id = np.split(
                self.label_id, [int(len(self.label_id) * self.split)]
            )
        else:
            if (
                os.path.isfile(file_path + "train_set.npy")
                and os.path.isfile(file_path + "test_set.npy")
                and os.path.isfile(file_path + "label.npy")
                and os.path.isfile(file_path + "label_id.npy")
            ):
                self.train_set = np.load(file_path + "train_set.npy")
                self.test_set = np.load(file_path + "test_set.npy")
                self.label = np.load(file_path + "label.npy")
                self.label_id = np.load(file_path + "label_id.npy")
            else:
                self.read_from_documents(docs_path, file_path)

            # split train and test labels
            self.train_label, self.test_label = np.split(
                self.label, [int(len(self.label) * self.split)]
            )
            self.train_label_id, self.test_label_id = np.split(
                self.label_id, [int(len(self.label_id) * self.split)]
            )

            self.calculate(file_path)

    def read_from_documents(self, docs_path, file_path):
        """read all documents one by one and tokenize words to create vsm"""

        # read and lemmatize stopwords
        stop_words = [
            re.sub("[,'\n]", "", self.lemmatize.lemmatize(word))
            for word in set(stopwords.words("english"))
        ]

        dir_lists = os.listdir(docs_path)
        total_files = sum([len(os.listdir(docs_path + pth)) for pth in dir_lists])

        # initialize vector space model and labels with zeros
        self.vsm = np.zeros(shape=(total_files, self.vsm_init_size))
        self.label = np.zeros(total_files)
        self.label_id = np.zeros(total_files)

        # open every document, and index and tokenize the words in them to create vsm
        tokenize_regex = r"[^\w]"
        file_index = 0
        for dir_no, dir_list in enumerate(dir_lists):
            file_list = os.listdir(docs_path + dir_list)
            file_list = sorted(
                file_list, key=lambda x: int("".join([i for i in x if i.isdigit()]))
            )
            for doc_id, file_name in enumerate(file_list):
                self.label[file_index] = dir_no
                self.label_id[file_index] = doc_id + 1
                with open(
                    docs_path + dir_list + "/" + file_name, "r", encoding="UTF-8"
                ) as file_data:
                    for line in file_data:
                        # split and tokenize word by given charecters
                        for word in re.split(tokenize_regex, line):
                            self.tokenize(word, file_index, stop_words)
                file_index += 1
        self.store_calculation(file_path)

    def store_calculation(self, file_path):
        """store training and testing data into file"""

        # delete unused columns in the vector
        self.delete_extra_cols()

        # filter out less used features
        temp_df = np.count_nonzero(self.vsm > 0, axis=0)
        del_index = [i for i, x in enumerate(temp_df) if x < self.filter]
        self.vsm = np.delete(self.vsm, del_index, 1)

        # randomize entries to get better training/testing set distribution and split them
        self.randomize_array()
        self.vsm_train, self.vsm_test = np.split(
            self.vsm, [int(len(self.vsm) * self.split)]
        )

        # calculate document frequency and idf using df
        df = np.count_nonzero(self.vsm_train > 0, axis=0) + 1
        N = len(self.vsm_train)
        idf = np.log10(N / df)
        self.train_set = np.multiply(self.vsm_train, idf)
        self.test_set = np.multiply(self.vsm_test, idf)

        # write training and testing data to file
        np.save(file_path + "train_set.npy", self.train_set)
        np.save(file_path + "test_set.npy", self.test_set)
        np.save(file_path + "label.npy", self.label)
        np.save(file_path + "label_id.npy", self.label_id)

    def tokenize(self, word, doc_id, stop_words):
        """tokenize and insert term frequency in vsm"""

        # remove trailing commas and apostrophes and lower word
        word = re.sub("[,'\n]", "", word)
        word = word.lower()

        # apply lemmatization algo to each word
        word = self.lemmatize.lemmatize(word)

        # if word is not stopword, increment vsm value
        if word and word not in stop_words:
            if word not in self.features:
                self.features[word] = len(self.features)

            # resize vsm if feature count exceed its' current size
            if len(self.features) > len(self.vsm[0]):
                self.insert_extra_cols()
            self.vsm[doc_id][self.features[word]] += 1

    def delete_extra_cols(self):
        """delete empty and unused columns from vsm"""

        start = len(self.features)
        stop = len(self.vsm[0])
        self.vsm = np.delete(self.vsm, [i for i in range(start, stop)], axis=1)

    def insert_extra_cols(self):
        """resize vsm if features start to overflow"""

        rows = len(self.vsm)
        cols = self.vsm_init_size
        self.vsm = np.concatenate((self.vsm, np.zeros(shape=(rows, cols))), axis=1)

    def calculate(self, result_path):
        """calculate and predict the class of testing set queries"""

        total = len(self.test_set)
        correct = 0
        sim_list = np.zeros(shape=(2, len(self.train_set)))
        self.result = []
        self.param = [self.split, self.filter, self.rand_seed]

        # for every test query, compute the similarity with every training set document. from top K similar documents, select the best and most similar class
        for query_index, query in enumerate(self.test_set):
            sim_list_index = 0
            for doc_index, doc in enumerate(self.train_set):
                sim = np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc))
                # sim = np.linalg.norm(doc-query)

                sim_list[0][sim_list_index] = sim
                sim_list[1][sim_list_index] = self.train_label[doc_index]
                sim_list_index += 1

            expected = self.test_label[query_index]
            result = (sim_list[:, sim_list[0].argsort()])[1][-3:]
            computed = mode(result)
            self.result.append(computed)
            if expected == computed:
                correct += 1
        self.accuracy = correct / total * 100
        self.param.append(self.accuracy)

        np.save(result_path + "result.npy", self.result)
        np.save(result_path + "param.npy", self.param)

        return self.accuracy
