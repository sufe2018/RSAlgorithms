# encoding:utf-8
import sys

sys.path.append("..")

import numpy as np
from mf import MF


class BiasSVD(MF):
    """
    docstring for BiasSVD
    implement the BiasSVD

    Koren Y, Bell R, Volinsky C. Matrix factorization techniques for recommender systems[J]. Computer, 2009, 42(8).
    """

    def __init__(self):
        super(BiasSVD, self).__init__()
        self.config.lambdaB = 0.001  # 偏置项系数    ·超参数
        # self.init_model()

    def init_model(self, k):
        super(BiasSVD, self).init_model(k)
        self.Bu = np.random.rand(self.rg.get_train_size()[0]) / (self.config.factor ** 0.5)  # bias value of user ·user的bias
        self.Bi = np.random.rand(self.rg.get_train_size()[1]) / (self.config.factor ** 0.5)  # bias value of item ·item的bias

    def train_model(self, k):
        super(BiasSVD, self).train_model(k)
        iteration = 0
        while iteration < self.config.maxIter:
            self.loss = 0
            for index, line in enumerate(self.rg.trainSet()):
                user, item, rating = line   #line是数据
                u = self.rg.user[user]
                i = self.rg.item[item]
                error = rating - self.predict(user, item)  #eui = rui - rui_estimate
                self.loss += error ** 2 #损失函数
                p, q = self.P[u], self.Q[i] #latent vector 应该是向量
                # update latent vectors

                self.Bu[u] += self.config.lr * (error - self.config.lambdaB * self.Bu[u]) #bu和bi是个体打分情况的偏差bias
                self.Bi[i] += self.config.lr * (error - self.config.lambdaB * self.Bi[i])

                self.P[u] += self.config.lr * (error * q - self.config.lambdaP * p) #随机梯度下降
                self.Q[i] += self.config.lr * (error * p - self.config.lambdaQ * q)

            self.loss += self.config.lambdaP * (self.P * self.P).sum() + self.config.lambdaQ * (self.Q * self.Q).sum() \
                         + self.config.lambdaB * ((self.Bu * self.Bu).sum() + (self.Bi * self.Bi).sum())
            iteration += 1
            if self.isConverged(iteration):
                break

    def predict(self, u, i):
        # super(BasicMFwithR, self).predict()
        if self.rg.containsUser(u) and self.rg.containsItem(i):
            u = self.rg.user[u]
            i = self.rg.item[i]
            return self.P[u].dot(self.Q[i]) + self.rg.globalMean + self.Bi[i] + self.Bu[u] #rui_estimate = u +bi + bu + qiTpu
        else:
            return self.rg.globalMean


if __name__ == '__main__':

    rmses = []
    maes = []
    bmf = BiasSVD()
    bmf.config.k_fold_num = 1
    # print(bmf.rg.trainSet_u[1])
    for i in range(bmf.config.k_fold_num):
        bmf.train_model(i)
        rmse, mae = bmf.predict_model()
        print("current best rmse is %0.5f, mae is %0.5f" % (rmse, mae))
        rmses.append(rmse)
        maes.append(mae)
    rmse_avg = sum(rmses) / 5
    mae_avg = sum(maes) / 5
    print("the rmses are %s" % rmses)
    print("the maes are %s" % maes)
    print("the average of rmses is %s " % rmse_avg)
    print("the average of maes is %s " % mae_avg)
