from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

from data.sets.urban.stanford_campus_dataset.scripts.relations import Loader
from data.sets.urban.stanford_campus_dataset.scripts.post_processing import PostProcessing
from data.sets.urban.stanford_campus_dataset.scripts.relations import Route
# from data.sets.urban.stanford_campus_dataset.scripts.train_model import series_to_supervised
from pandas import DataFrame

""" This is code for simple GP regression. It assumes a zero mean GP Prior """


# This is the true unknown function we are trying to approximate
#f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


path = "../annotations/hyang/video0/"
loader = Loader(path)
south = np.array([720, 1920])
north = np.array([720, 0])
east = np.array([720*2, 1920/2])
route1 = Route(south, north)
route2 = Route(east, south)
loader.make_obj_dict_by_route(route1, route1, True, 'Biker', compute_all_routes=False)
postprocessor = PostProcessing(loader)
raw = DataFrame()
raw['xdot'] = [x for x in postprocessor.dx]
#raw['ydot'] = [x for x in postprocessor.dy]
values = raw.values



# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 1000      # number of training points.
n = 500         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points.
X = np.random.uniform(0, N, size=(N,1))
X = np.linspace(0, N, N).reshape(-1,1)
#y = f(X) + s*np.random.randn(N)
y = np.squeeze(values[0:N])
print(y.shape)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(0, N, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)


# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, y[::int(len(y)/n)], 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-0, N, -10, 10])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,N)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-0, N, -10, 10])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,N)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-0, N, -10, 10])
pl.savefig('post.png', bbox_inches='tight')

pl.show()

# from keras.layers import Input, SimpleRNN
# from keras.optimizers import Adam
#
# from kgp.layers import GP
# from kgp.models import Model
# from kgp.losses import gen_gp_loss
#
# input_shape = (10, 2)  # 10 time steps, 2 dimensions
# batch_size = 32
# nb_train_samples = 512
# gp_hypers = {'lik': -2.0, 'cov': [[-0.7], [0.0]]}
#
# # Build the model
# inputs = Input(shape=input_shape)
# rnn = SimpleRNN(32)(inputs)
# gp = GP(gp_hypers,
#         batch_size=batch_size,
#         nb_train_samples=nb_train_samples)
# outputs = [gp(rnn)]
# model = Model(inputs=inputs, outputs=outputs)
#
# # Compile the model
# loss = [gen_gp_loss(gp) for gp in model.output_layers]
# model.compile(optimizer=Adam(1e-2), loss=loss)