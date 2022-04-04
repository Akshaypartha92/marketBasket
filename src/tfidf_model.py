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


class TfIdfModel:

    # Constructor:
    def __init__(self):
        True

    def tfidf_weight(self, tf):
        """
        Given a Term Frequency matrix
        Returns a TF-IDF weight matrix
        """

        tf_idf = coo_matrix(tf)

        # calculate IDF
        N = float(tf_idf.shape[0])
        idf = log(N / (1 + bincount(tf_idf.col)))

        # apply TF-IDF adjustment
        tf_idf.data = sqrt(tf_idf.data) * idf[tf_idf.col]
        return tf_idf

    def generateRecommendations(self, target_user_id, tf_idf,df_prior_user_products, df_product_frequency, cos_vec, K, N):
        """
        Given a target_user (a row), a cosine similarity vector, the number of similar users K,
              the number of products to be recommended.
        Returns product set by target user and N recommendations
        """

        # Fetch row of target user
        target_user = tf_idf[target_user_id - 1]
        # Select top K similar users
        top_K_similar_users = heapq.nlargest(K + 1, range(len(cos_vec)), cos_vec.take)

        # Initialize the result for recommendations
        recommendations = []

        # Exclude the user with same purchase history (1.00000) as the target user and implement set-minus
        products_target_user = df_prior_user_products.loc[df_prior_user_products['user_id'] == target_user_id].products

        # Products of Target User
        productset_target_user = set(products_target_user.tolist()[0])

        # Fetch the preliminary recommendations
        for similar_user_id in top_K_similar_users:

            products_similar_user = df_prior_user_products.loc[
                df_prior_user_products['user_id'] == similar_user_id + 1].products

            # Recommend the products bought by the user who firstly differs in the purchase history from A.
            candidate_recommendation = set(products_similar_user.tolist()[0]) - productset_target_user

            # If similar_user_id equals to target_user_id or the candidate_recommendation is empty,
            # skip current user
            if similar_user_id == target_user_id or not candidate_recommendation: continue

            # One candidate_recommendation found, and extend it to the result
            recommendations.extend(candidate_recommendation)

            # If length of recommendations exceed N, break
            # Needed because this will ensure the recommentations are the products bought by most similar users
            if len(recommendations) > N: break

        # Pick the top N popularity (overall sales) to recommend
        h = []
        for rec in recommendations:
            heapq.heappush(h, (df_product_frequency.loc[rec]['frequency'], rec))
            if len(h) > N:
                heapq.heappop(h)

        return productset_target_user, [item[1] for item in h]
