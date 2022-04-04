from implicit.nearest_neighbours import tfidf_weight
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from pathlib import Path
from numpy import bincount, log, sqrt

import scipy.sparse as sparse
import implicit
import pandas as pd
import numpy as np
import pickle
import time
import heapq
from tfidf_model import TfIdfModel


class RecommenderEvaluation:
    # Constructor:
    def __init__(self):
        True

    def recall_score(self, actual, pred):
        """
        Given two lists representing actual and predicted values
        Returns the recall of the prediction
        """
        if len(actual) == 0:
            return 0
        actual, pred = set(actual), set(pred)
        return len(actual.intersection(pred)) / len(actual)

    def new_products(self, row, df_prior_user_products):
        """
        Given a row in the test dataset
        Returns the list of new products purchased
        """
        actual = row["products"][1:-1]
        actual = set([int(p.strip()) for p in actual.strip().split(",")])
        products_target_user = df_prior_user_products.loc[df_prior_user_products['user_id'] == row["user_id"]].products
        liked = set(products_target_user.tolist()[0])
        return actual - liked

    def popular_recommend(self, row, popular_products):
        """
        Given a row in the test dataset
        Returns the recall score when popular products are recommended
        """
        actual = self.new_products(row)
        return self.recall_score(actual, popular_products)

    def tfidf_recommend(self, row, tf_idf, level=[True] * 10, total=0):
        """
        Given a row in the test dataset
        Returns the recall score when our model recommends products
        """
        actual = row["products"][1:-1]
        actual = [int(p.strip()) for p in actual.strip().split(",")]
        target_user = tf_idf[row["user_id"] - 1]
        similarities = cosine_similarity(tf_idf, target_user, False)
        cos_vec = similarities.toarray()
        productset_target_user, recommended = TfIdfModel.generateRecommendations(target_user, cos_vec, 20, 10)

        cur_recall_score = self.recall_score(actual, recommended)

        global count, progress, recall_sum
        count += 1;
        recall_sum += cur_recall_score
        if level[progress] and int(count / total * 10) - 1 == progress:
            level[progress] = False
            progress += 1
            print("{:.1f}% completed, current mean of recall = {}".format(progress * 10, recall_sum / count))

        return cur_recall_score

    def build_eval_df(self, df_user_products_test, subset=None):
        """
        Builds a dataframe of recall values of the baseline and our model for all the users
        in the test data, and saves its to disk at `filepath`
        """
        start = time.time()
        print("Building dataframe with recall values ...")

        df_eval = df_user_products_test.copy()
        if subset:
            df_eval = df_eval.sample(n=int(len(df_eval) * subset), random_state=7)
        df_eval["popular_score"] = df_eval.apply(self.popular_recommend, axis=1)
        df_eval["tfidf_score"] = df_eval.apply(self.tfidf_recommend, axis=1)

        print("Completed in {:.2f}s".format(time.time() - start))

        return df_eval
