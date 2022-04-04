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


class DataTransformation:
    # Constructor:
    def __init__(self):
        True

    ### Helper Functions

    def sparsity(self, matrix):
        """
        Given a matrix, returns its sparsity
        """
        total_size = matrix.shape[0] * matrix.shape[1]
        actual_size = matrix.size
        sparsity = (1 - (actual_size / total_size)) * 100
        return (sparsity)

    def get_k_popular(self, k, df_merged_order_products_prior):
        """
        Base model
        Returns the `k` most popular products based on purchase count in the dataset
        """
        popular_products = list(df_merged_order_products_prior["product_id"].value_counts().head(k).index)
        return popular_products

    def create_prior_order(self, df_orders):
        """
        Generates the prior dataset including prior_user_products and product_frequency
        """
        # Read prior order csv
        df_order_products_prior = pd.read_csv("../input/order_products__prior.csv")
        current_order_user_df = df_orders.loc[(df_orders.eval_set == "prior")].reset_index()
        current_order_user_df = current_order_user_df[["order_id", "user_id"]]

        assert len(current_order_user_df["order_id"].unique()) == len(df_order_products_prior["order_id"].unique())

        # Group product_id for each order into products
        df_order_products_prior = df_order_products_prior[["order_id", "product_id"]]
        df_product_frequency = df_order_products_prior['product_id'].value_counts()
        df_order_products_prior = df_order_products_prior.groupby("order_id")["product_id"].apply(
            list).reset_index().rename(columns={"product_id": "products"})

        assert current_order_user_df.size == df_order_products_prior.size

        df_prior_user_products = pd.merge(current_order_user_df, df_order_products_prior, on="order_id")
        df_prior_user_products = df_prior_user_products[["user_id", "products"]]
        df_prior_user_products = df_prior_user_products.groupby("user_id")["products"].agg(sum).reset_index()

        return df_prior_user_products, df_product_frequency

    def create_test_data(self, test_data_path, df_orders, df_order_products_train):
        """
        Generates the test dataset and saves it to disk at the given path
        """
        start = time.time()
        print("Creating test data ...")

        # Read train csv
        df_order_user_current = df_orders.loc[(df_orders.eval_set == "train")].reset_index()
        df_order_user_current = df_order_user_current[["order_id", "user_id"]]

        # Sanity check #1: `current_order_user_df` and `df_order_products_train` should have the same number of
        # unique order ids
        assert len(df_order_user_current["order_id"].unique()) == len(df_order_products_train["order_id"].unique())

        # Convert train dataframe to a similar format
        df_order_products_test = df_order_products_train[["order_id", "product_id"]]
        df_order_products_test = df_order_products_test.groupby("order_id")["product_id"].apply(
            list).reset_index().rename(columns={"product_id": "products"})

        # Sanity check #2: `df_order_products_test` and `df_order_user_current` should have the same number of
        # records before attempting to merge them
        assert df_order_products_test.size == df_order_user_current.size

        # Merge on order id
        df_user_products_test = pd.merge(df_order_user_current, df_order_products_test, on="order_id")
        df_user_products_test = df_user_products_test[["user_id", "products"]]

        # Write to disk
        df_user_products_test.to_csv(test_data_path, index_label=False)

        print("Completed in {:.2f}s".format(time.time() - start))

    def save_data(self, dataframe, df_name):
        """
        Save the data to disk
        """
        filepath = "../input/df_{}.pkl".format(df_name)
        dataframe.to_pickle(filepath)

    def get_user_product_prior_df(self, filepath, df_orders, df_order_products_prior):
        """
        Generates a dataframe of users and their prior products purchases, and writes it to disk at the given path
        """
        start = time.time()
        print("generating prior user product dataframe ...")

        # Consider ony "prior" orders and remove all columns except `user_id` from `df_orders`
        df_order_user_prior = df_orders.loc[df_orders.eval_set == "prior"]
        df_order_user_prior = df_order_user_prior[["order_id", "user_id"]]

        # Remove all columns except order_id and user_id from df_orders and
        # merge the above on `order_id` and remove `order_id`
        df_merged = pd.merge(df_order_user_prior, df_order_products_prior[["order_id", "product_id"]], on="order_id")
        df_user_product_prior = df_merged[["user_id", "product_id"]]
        df_user_product_prior = df_user_product_prior.groupby(["user_id", "product_id"]).size().reset_index().rename(
            columns={0: "quantity"})

        # Write to disk
        df_user_product_prior.to_csv(filepath, index_label=False)

        print("Completed in {:.2f}s".format(time.time() - start))

    def generate_product_user_matrix(self, matrix_path, df_user_product_prior):
        """
        Generates a utility matrix representing purchase history of users, and writes it to disk.
        Rows and Columns represent products and users respectively.
        """
        start = time.time()
        print("Creating product user matrix ...")

        # Make the dataframe a sparse matrix
        df_user_product_prior["user_id"] = df_user_product_prior["user_id"].astype("category")
        df_user_product_prior["product_id"] = df_user_product_prior["product_id"].astype("category")
        product_user_matrix = sparse.coo_matrix((df_user_product_prior["quantity"],
                                                 (df_user_product_prior["product_id"].cat.codes.copy(),
                                                  df_user_product_prior["user_id"].cat.codes.copy())))

        sparse.save_npz(matrix_path, product_user_matrix)

        print("Completed in {:.2f}s".format(time.time() - start))
