import pandas as pd
import tqdm 
from surprise import Dataset, Reader


RATING_COL = "rating"
USER_COL = "userId"
ITEM_COL = "movieId"


reader = Reader(rating_scale = (0.5, 5.0))

def load(path, columns=['userId', 'movieId', 'rating']):
    return pd.read_csv(path, sep='\t', names=columns)

def save(df, path):
    df.to_csv(path, index=False, header=False, sep='\t')


def get_recommendation_raw(model, test, sample_size=None, top_k=10):

    if sample_size is None:
        sample_size = len(test)
    genres_df = test[["movieId", "genres"]].drop_duplicates()
    prediction_user_map = {}
    full_df = pd.DataFrame(columns=["userId", "movieId", "predicted_rating"])
    movies_ids = list(set(test["movieId"].unique()))

    for user in tqdm.tqdm(test[USER_COL].unique()[:sample_size]):
        teste = pd.DataFrame({ITEM_COL: movies_ids, RATING_COL: 0.0, USER_COL: user})

        testset = (
            Dataset.load_from_df(
                teste[[USER_COL, ITEM_COL, RATING_COL]],
                reader=reader,
            )
            .build_full_trainset()
            .build_testset()
        )

        pred_list = model.test(testset)

        predictions = sorted(
            [(pred.iid, pred.est)for pred in pred_list if ((pred.uid == user))],
            key=lambda x: x[1],reverse=True
        )


        user_df = pd.DataFrame(predictions[:10], columns=["movieId", "predicted_rating"])
        user_df["userId"] = user
        full_df = pd.concat([full_df, user_df])

    full_df = full_df.merge(genres_df, on="movieId")
    return full_df
