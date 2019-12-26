import numpy as np
from scipy.special import erf as sperf

def LAD_fit(x, y, tol=1e-8, max_iter=5000):
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape

    #s_sample, s_target = len(y),1 # 2019.12.26: Tai -- only one target

    w_sol = 0.0*(np.random.rand(s_pred,s_target) - 0.5)
    b_sol = np.random.rand(1,s_target) - 0.5

#     print(weights.shape)
    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]

#         print(cov.shape, cov_xx.shape, cov_xy.shape)
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1
            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
#             old_error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
#             print(w_sol[:,index].shape, npl.solve(cov_xx, cov_xy).reshape(s_pred).shape)
            w_sol[:,index] = np.linalg.solve(cov_xx,cov_xy).reshape(s_pred)
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std(weights)
            error = np.mean(np.abs(weights))
#             error = np.mean(np.abs(b_sol[0,index] + x_test.dot(w_sol[:,index]) - y_test[:,index]))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit
            weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
#             print(old_error,error)
    return b_sol,w_sol

