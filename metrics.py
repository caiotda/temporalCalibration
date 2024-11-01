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

        