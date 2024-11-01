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


def DCG(userRecs, ground_truth):
    """ 
        ins:
            - userRecs: list
            - ground_truth: list
        out:
            DCG for the recommendations generated for the user
    """ 

    dcg = 0
    for i, rec in enumerate(user_recs):
        dcg += (2 ** rel(rec, ground_truth) - 1)   / np.log(i + 1)
    return dcg


def NDCG(userId, recs, ground_truth):
     """ 
        ins:
            - userId: (str)
            - recs: pd.DataFrame
            - ground_truth: pd.Dataframe
        out:
            NDCG for the recommendations generated for the user
    """ 

    user_recs = list(recs[recs["userId"] == userId].itemId.values())
    user_ground_truth = list(ground_truth[ground_truth["userId"] == userId].itemId.values())
    u_dcg = DCG(user_recs, user_ground_truth)
    i_dcg = DDG(user_ground_truth, user_ground_truth)

    return u_dcg / i_dcg
