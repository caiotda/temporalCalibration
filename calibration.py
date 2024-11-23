
import tqdm
import pandas as pd
import numpy as np 

from utils import get_recommendation_raw
from surprise import Dataset, Reader

from metrics import avg_ndcg

reader = Reader(rating_scale = (0.5, 5.0))
RATING_COL = "rating"
USER_COL = "userId"
ITEM_COL = "movieId"

class w_u_i:
    def __init__(self, rating, time_delta, max_time_delta, mode='classic'):
        self.mode = mode
        self.rating = rating
        self.delta = time_delta
        self.max_time_delta = max_time_delta

    def weight(self):
        if self.mode == 'classic':
            return 1
        if self.mode == 'rating':
            return self.rating
        elif self.mode == 'rational':
            return self.generate_rational_weight()

        elif self.mode == 'exponential':
            return self.generate_exponential_weight()


    def generate_rational_weight(self):
        return self.rating / (self.delta + 1)

    def generate_exponential_weight(self):
      if self.max_time_delta == 0:
        return self.rating
      return self.rating * np.exp(-self.delta / self.max_time_delta)



def get_user_profile_distribution(df, userId, weight='exponential'):
    df_genres  = df[["movieId", "genres"]].drop_duplicates()

    genre_map = {i['movieId']:i['genres'].split("|") for i in df_genres[['movieId', 'genres']].to_dict('records')}
    
    user_profile_distribution = {}
    user_df = df[df['userId'] == userId]
    max_time_delta = user_df['age_days'].max()
    s = 0
    for _, row in user_df.iterrows():
      genre = genre_map.get(row.movieId, ["unk"])
      n_genres = len(genre)
      delta = row.age_days
      rating = row.rating
      p_g_i = 1/n_genres
      wui = w_u_i(rating=rating, time_delta=delta, max_time_delta=max_time_delta, mode=weight).weight()

      for genre in genre_map[row.movieId]:
          if genre not in user_profile_distribution:
              user_profile_distribution[genre] = 0
          user_profile_distribution[genre] += p_g_i * wui
      s += wui


    user_profile_distribution = {k: v/s for k, v in sorted(user_profile_distribution.items(), key=lambda item: item[1])}
    return user_profile_distribution


def get_gender_distribution_in_recommendation(prediction_df, userId, genre_map):
    user_rec_distribution = {}
    user_df = prediction_df[prediction_df['userId'] == userId]
    n = 0

    for _, row in user_df.iterrows():
        genre = genre_map.get(row.movieId, ["unk"])
        n_genres = len(genre)
        p_g_i = 1/n_genres
        rating = row.predicted_rating
        for genre in genre_map[row.movieId]:
            if genre not in user_rec_distribution:
                user_rec_distribution[genre] = 0
            user_rec_distribution[genre] += rating * p_g_i
        n += rating

    user_rec_distribution = {k: v/n for k, v in sorted(user_rec_distribution.items(), key=lambda item: item[1])}
    return user_rec_distribution



def rerank_recommendation(profile_dist, list_recomended_items, user, N, tradeoff, genre_map):
    re_ranked_list = []
    re_ranked_with_score = []

    for _ in range(N):

        max_mmr = -np.inf
        max_item = None
        max_item_rating = None

        for item, rating in list_recomended_items:
            if item in re_ranked_list:
                continue

            temporary_list = re_ranked_list + [item]
            temporary_list_with_score = re_ranked_with_score + [(item, rating)]

            weight_part = sum(
                recomendation[1]
                for recomendation in temporary_list_with_score
            )

            full_tmp_calib = calculate_calibration_sum(
                profile_dist,
                temporary_list_with_score,
                user,
                genre_map=genre_map
            )

            maximized = (1 - tradeoff)*weight_part - tradeoff*full_tmp_calib

            if maximized > max_mmr:
                max_mmr = maximized
                max_item = item
                max_item_rating = rating

        if max_item is not None:
            re_ranked_list.append(max_item)
            re_ranked_with_score.append((max_item, max_item_rating))

    return re_ranked_list, re_ranked_with_score


def preprocess_list_with_score(item_score_list, userId):
    list_with_user = [(r[0], r[1], userId) for r in item_score_list]
    df = pd.DataFrame(list_with_user, columns=["movieId", "predicted_rating", "userId"])
    return df


def calculate_calibration_sum(profile_dist, temporary_list_with_score, user, genre_map, alpha=0.001):
    kl_div = 0.0
    tmp_scored_df = preprocess_list_with_score(temporary_list_with_score, user)
    reco_distr = get_gender_distribution_in_recommendation(tmp_scored_df, user, genre_map=genre_map)
    for genre, p in profile_dist.items():
        q = reco_distr.get(genre, 0.0)
        til_q = (1 - alpha) * q + alpha * p

        if p == 0.0 or til_q == 0.0:
            kl_div = kl_div
        else:
            kl_div = kl_div + (p * np.log2(p / til_q))
    return kl_div


def get_recommendation_fairness(model, test, sample_size=None, lambda_=0.9, calibration_mode='exponential'):
    df_genres  = test[["movieId", "genres"]].drop_duplicates()

    genre_map = {i['movieId']:i['genres'].split("|") for i in df_genres[['movieId', 'genres']].to_dict('records')}
    prediction_user_map = {}
    full_df = pd.DataFrame(columns=["userId", "movieId", "predicted_rating"])
    movies_ids = list(set(test["movieId"].unique()))
    print(f"calibrating using {calibration_mode}")
    for user in tqdm.tqdm(test[USER_COL].unique()[:sample_size]):

        user_profile_distribution = get_user_profile_distribution(test, user, weight=calibration_mode)

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

        reranked_list, reranked_with_score = rerank_recommendation(
            user_profile_distribution,
            predictions[:100],
            user,
            10,
            lambda_,
            genre_map=genre_map
        )

        user_df = pd.DataFrame(reranked_with_score, columns=["movieId", "predicted_rating"])
        user_df["userId"] = user
        full_df = pd.concat([full_df, user_df])


    return full_df



def user_rank_miscalibration(user_profile_dist, rec_profile_dist, alpha=0.001):
    p_g_u = user_profile_dist
    q_g_u = rec_profile_dist

    Ckl = 0
    for genre, p in p_g_u.items():
        q = q_g_u.get(genre, 0.0)
        til_q = (1 - alpha) * q + alpha * p

        if til_q == 0 or p_g_u.get(genre, 0) == 0:
            Ckl = Ckl
        else:
            Ckl += p * np.log2(p / til_q)
    return Ckl

def get_user_miscalibration(recs, test, user, alpha=0.001):
    df_genres  = test[["movieId", "genres"]].drop_duplicates()
    genre_map = {i['movieId']:i['genres'].split("|") for i in df_genres[['movieId', 'genres']].to_dict('records')}
    user_profile_dist = get_user_profile_distribution(test, user)
    user_rec_dist = get_gender_distribution_in_recommendation(recs, user, genre_map)

    return user_rank_miscalibration(user_profile_dist, user_rec_dist, alpha=alpha)

def get_mean_rank_miscalibration(predictions_df, test, genre_map, calibrate=None):
    MRMC = 0

    for _, row in predictions_df.iterrows():
        user = row.userId
        predictions_user = predictions_df[predictions_df['userId'] == user]
        RMC = 0
        if calibrate == None:
          user_profile_dist = get_user_profile_distribution(test, user, weight='classic')
        else:
          user_profile_dist = get_user_profile_distribution(test, user, weight=calibrate)
        if user_profile_dist == {}:
            continue

        void = user_rank_miscalibration(user_profile_dist, {})
        N = len(predictions_user)
        for i in range(1, N):
            user_rec_dist = get_gender_distribution_in_recommendation(predictions_user.iloc[:i], user, genre_map=genre_map)
            kl = user_rank_miscalibration(user_profile_dist, user_rec_dist)
            RMC += kl/void

        MRMC += RMC/N

    return MRMC/len(predictions_df)




def evaluate(model, test, sample_size=None):
    if sample_size is None:
        sample_size = len(test)
    df_genres  = test[["movieId", "genres"]].drop_duplicates()
    genre_map = {i['movieId']:i['genres'].split("|") for i in df_genres[['movieId', 'genres']].to_dict('records')}

    baseline_metrics = {}
    calibrated_metrics = {}
    calibrations = ['classic', 'rating', 'rational', 'exponential']
    print("Generating baseline recommendations...")
    baseline_recs = get_recommendation_raw(model, test, sample_size=sample_size)
    print("Done!")
    baseline_metrics['ndcg'] = avg_ndcg(baseline_recs, test)
    baseline_metrics['mmr'] = get_mean_rank_miscalibration(baseline_recs, test, genre_map=genre_map, calibrate=None)
    for calibration_mode in calibrations:
        print("Generating calibrated recommendations...")
        fairness_recs = get_recommendation_fairness(model, test, calibration_mode=calibration_mode, sample_size=sample_size)
        print("Done!")

        calibrated_metrics[calibration_mode] = {}
        calibrated_metrics[calibration_mode]['ndcg'] = avg_ndcg(fairness_recs, test)
        calibrated_metrics[calibration_mode]['mmr'] = get_mean_rank_miscalibration(fairness_recs, test, genre_map=genre_map, calibrate=calibration_mode)
    return baseline_metrics, calibrated_metrics