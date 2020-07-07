import sys
import warnings

warnings.filterwarnings('ignore')

import pickle
import numpy as np
import scipy
import pandas as pd
import lightgbm as lgb


class ALS:
    def __init__(self,
                 csr_user_item_rate,
                 random_seed=16777216):
        self.R = csr_user_item_rate
        self.u_cnt = self.R.shape[0]
        self.i_cnt = self.R.shape[1]
        np.random.seed(random_seed)

    def fit(self,
            latent_features=10,
            lambda_coef=0.1,
            n_iter=5):
        U = scipy.sparse.csr_matrix(np.random.rand(self.u_cnt, latent_features))
        I = scipy.sparse.csr_matrix(np.random.rand(self.i_cnt, latent_features))
        eye = scipy.sparse.csr_matrix(np.eye(latent_features) * lambda_coef)

        for itr in range(n_iter):
            a = scipy.sparse.linalg.inv(U.T * U + eye)
            b = (self.R.T * U).T
            I = (a * b).T

            a = scipy.sparse.linalg.inv(I.T * I + eye)
            b = (self.R * I).T
            U = (a * b).T

        self.f_cnt = latent_features
        self.U = U
        self.I = I
        self.UTU = scipy.sparse.linalg.inv(U.T * U + eye)
        self.ITI = scipy.sparse.linalg.inv(I.T * I + eye)

    def recommend_train(self,
                        user=0,
                        item_list=None):
        if item_list is None:
            p = (self.U[user, :] * self.I.T)
        else:
            p = (self.U[user, :] * self.I[item_list, :].T)

        return p.toarray()

    def recommend(self,
                  csr_user_item_rate,
                  item_list=None):
        self.r = csr_user_item_rate
        self.r.resize((csr_user_item_rate.shape[0], self.i_cnt))
        self.u = (self.ITI * (self.r * self.I).T).T

        if item_list is None:
            p = (self.u * self.I.T)
        else:
            p = (self.u * self.I[item_list, :].T)

        return p.toarray()


np.random.seed(16777216)

filename = sys.argv[1]

try:
    n_recommendation = int(sys.argv[2])
except:
    n_recommendation = 10

rating = pd.read_csv(filename)
movie = pd.read_pickle('res/movie.pkl')
model_collaborative = pickle.load(open('res/model_collaborative.pkl', 'rb'))
model = lgb.Booster(model_file='res/model.pmml')
movie_index_to_id = pickle.load(open('res/movie_index_to_id.pkl', 'rb'))
movie_id_to_index = pickle.load(open('res/movie_id_to_index.pkl', 'rb'))
feats = pickle.load(open('res/feats.pkl', 'rb'))

rating['movieId_'] = rating['movieId'].map(movie_id_to_index)
rating.dropna(inplace=True)
rating['movieId_'] = rating['movieId_'].astype(int)
rating['userId_'] = 0

csr_user_item_rate = scipy.sparse.csr_matrix((rating['rating'], (rating['userId_'], rating['movieId_'])))
model_collaborative.recommend(csr_user_item_rate, 0)
lf = model_collaborative.ITI.shape[0]

user_features = pd.DataFrame(model_collaborative.u.toarray()).reset_index()
user_features.columns = ['userId_'] + ['uf' + str(i) for i in range(lf)]

test = movie.merge(rating[['movieId_', 'userId_']],
                   how='left',
                   on='movieId_')
test = test[~(test['userId_'] == 0)]
test['userId_'] = 0

test = test.merge(user_features,
                  on='userId_')
for i in range(lf):
    test['ff' + str(i)] = test['uf' + str(i)] * test['mf' + str(i)]

pred = model.predict(test[feats])
recommendation = [movie_index_to_id[i] for i in (-pred).argsort()[:n_recommendation]]

for r in recommendation:
    print(r)
