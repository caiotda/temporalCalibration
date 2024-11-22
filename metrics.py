from scipy.stats import entropy
import numpy as np

RATING_COL = "rating"
USER_COL = "userId"
ITEM_COL = "movieId"



def KL(p, q):
    """
        ins:
            - p (list) representing a probability distribution
            - q (list): another probability distribution

        out:
            kl (float): the KL divergence between p and q
    """

    return  entropy(p,q)


def MRR(users, recs, ground_truth):
    """
        ins: 
            - ground_truth: pd.DataFrame
            - recs: pd.DataFrame
            - users: list of ids
        out:
            - MRR: float

        Given two datasets, one representing every recommendation generated
        and one representing the ground truth for every user, and 
        a listrepresenting every user id, calculate the Mean Reciprocal Rank
    """
    rr_s = 0
    for userId in users:
        recs_user = list(recs[recs["userId"] == userId].itemId.values())
        ground_truth_user = list(ground_truth[ground_truth["userId"] == userId].values())
        rr_s += rr(recs_user, ground_truth)
    n_users = len(users)

    return rr_s/n_users


def RR(rec, actual):
    """
        ins:
            - rec: (list) recommendation list for user
            - actual: (list) ground truth for user
        out:
            -rr: float

        Given a recommendation list and the ground truth list, return the 
        reciprocal rank
    """

    for rel in actual:
        if rel in rec:
            idx = rec.index(rel)
            break
    return 1/idx

def rel(itemId, ground_truth):
    """
        ins:
            - itemId (str)
            - ground_truth (list)
        out:
            - relevance of item with itemId given
            a ground_truth

        Implements binary relevance: if itemId in ground_truth, return 1. Return 0 elsewhere.
    """


    return itemId in ground_truth


def DCG(user_recs, ground_truth):
    """
        ins:
            - userRecs: list
            - ground_truth: list
        out:
            DCG for the recommendations generated for the user
    """

    dcg = 0
    for i, rec in enumerate(user_recs):
        dcg += rel(rec, ground_truth) / np.log2(i + 2)
    return dcg


def NDCG(userId, recs, ground_truth, k=10):
    """
        ins:
            - userId: (str)
            - recs: pd.DataFrame
            - ground_truth: pd.Dataframe
        out:
            NDCG for the recommendations generated for the user
    """

    user_recs = list(recs[recs[USER_COL] == userId][ITEM_COL])[:k]
    user_ground_truth = list(ground_truth[ground_truth[USER_COL] == userId][ITEM_COL])[:k]
    u_dcg = DCG(user_recs, user_ground_truth)
    i_dcg = DCG(user_ground_truth, user_ground_truth)

    return u_dcg / i_dcg


def avg_ndcg(recs, ground_truth):
  ndcg = 0
  for user in recs[USER_COL].unique():
    ndcg += NDCG(user, recs, ground_truth)
  return ndcg/len(recs[USER_COL].unique())


