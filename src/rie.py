import numpy as np
import tensorflow as tf
import tqdm
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU
import scipy.io
import os
from glob import glob
from keras import backend as k

"""
along the lines of 
Fast Approximate Geodesics for Deep Generative Models
Nutan Chen, Francesco Ferroni, Alexej Klushyn, Alexandros Paraschos, Justin Bayer, Patrick van der Smagt
"""

input_path = '../dataset/Human3.6m/latent_nth'
#subs = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
subs = ['S1']

class RiemannianMetric(object):
    def __init__(self, x, z, session):
        self.x = x
        self.z = z
        self.session = session


    def create_tf_graph(self):
        """
        creates the metric tensor (J^T J and J being the jacobian of the decoder), 
        which can be evaluated at any point in Z
        and
        the magnification factor
        """

        # the metric tensor
        output_dim = self.x.shape[1].value
        print('h')
        # derivative of each output dim wrt to input (tf.gradients would sum over the output)
        J = [tf.gradients(self.x[:, _], self.z)[0] for _ in range(output_dim)]
        J = tf.stack(J, axis=1)  # batch x output x latent
        self.J = J
        print('hh')

        G = tf.transpose(J, [0, 2, 1]) @ J  # J^T \cdot J
        self.G = G
        print('hhh')

        # magnification factor
        MF = tf.sqrt(tf.linalg.det(G))
        self.MF = MF

    def riemannian_distance_along_line(self, z1, z2, n_steps):
        """
        calculates the riemannian distance between two near points in latent space on a straight line
        the formula is L(z1, z2) = \int_0^1 dt \sqrt(\dot \gamma^T J^T J \dot gamma)
        since gamma is a straight line \gamma(t) = t z_1 + (1-t) z_2, we get
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T J^T J [z1-z2])
        L(z1, z2) = \int_0^1 dt \sqrt([z_1 - z2]^T G [z1-z2])
        z1: starting point
        z2: end point
        n_steps: number of discretization steps of the integral
        """

        # discretize the integral aling the line
        t = np.linspace(0, 1, n_steps)
        dt = t[1] - t[0]
        the_line = np.concatenate([_ * z1 + (1 - _) * z2 for _ in t])

        if True:
            # for weird reasons it seems to be alot faster to first eval G then do matrix mutliple outside of TF
            G_eval = self.session.run(self.G, feed_dict={self.z: the_line})

            # eval the integral at discrete point
            L_discrete = np.sqrt((z1-z2) @ G_eval @ (z1-z2).T)
            L_discrete = L_discrete.flatten()

            L = np.sum(dt * L_discrete)

        else:
            # THIS IS ALOT (10x) slower, although its all in TF
            DZ = tf.constant(z1 - z2)
            DZT = tf.constant((z1 - z2).T)
            tmp_ = tf.tensordot(self.G, DZT, axes=1)
            tmp_ = tf.einsum('j,ijk->ik', DZ[0], tmp_ )
            # tmp_ = tf.tensordot(DZ, tmp_, axes=1)

            L_discrete = tf.sqrt(tmp_)  # this is a function of z, since G(z)

            L_eval = self.session.run(L_discrete, feed_dict={self.z: the_line})
            L_eval = L_eval.flatten()
            L = np.sum(dt * L_eval)

        return L


class RiemannianTree(object):
    """docstring for RiemannianTree"""

    def __init__(self, riemann_metric):
        super(RiemannianTree, self).__init__()
        self.riemann_metric = riemann_metric  # decoder input (tf_variable)


    def create_riemannian_graph(self, z, n_steps, n_neighbors):

        n_data = len(z)
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(z)

        G = nx.Graph()

        # Nodes
        for i in range(n_data):
            n_attr = {f'z{k}': float(z[i, k]) for k in range(z.shape[1])}
            G.add_node(i, **n_attr)

        # edges
        for i in tqdm.trange(n_data):
            distances, indices = knn.kneighbors(z[i:i+1])
            # first dim is for samples (z), but we only have one
            distances = distances[0]
            indices = indices[0]

            for ix, dist in zip(indices, distances):
                # calculate the riemannian distance of z[i] and its nn

                # save some computation if we alrdy calculated the other direction
                if (i, ix) in G.edges or (ix, i) in G.edges or i == ix:
                    continue

                L_riemann = self.riemann_metric.riemannian_distance_along_line(z[i:i+1], z[ix:ix+1], n_steps=n_steps)
                L_euclidean = dist

                # note nn-distances are NOT symmetric
                edge_attr = {'weight': float(1/L_riemann),
                             'weight_euclidean': float(1/L_euclidean),
                             'distance_riemann': float(L_riemann),
                             'distance_euclidean': float(L_euclidean)}
                G.add_edge(i, ix, **edge_attr)
        return G



def main():
    batch_size = 1
    z_dim = 512
    pca_dim = 2048
    x_dim = 70720

    z0 = '../dataset/Human3.6m/latent_nth/S1/Directions/5.mat'
    z1 = '../dataset/Human3.6m/latent_nth/S1/Directions/185.mat'

    m = Sequential()
    m.add(Dense(pca_dim, input_dim = z_dim, name = 'fc5'))
    m.add(LeakyReLU(alpha = 0.01))
    m.add(Dense(pca_dim, name = 'fc6'))
    m.add(LeakyReLU(alpha = 0.01))
    m.add(Dense(pca_dim, name = 'fc7'))
    m.add(LeakyReLU(alpha = 0.01))
    m.add(Dense(x_dim, activation = 'sigmoid', name = 'fc8'))
    m.load_weights('../dataset/Human3.6m/vae_weights_bias.h5')

    #read z0 and z1
    s_z0 = z0.split('/')[-3]
    s_z1 = z1.split('/')[-3]
    act_z0 = z0.split('/')[-2]
    act_z1 = z1.split('/')[-2]
    name_z0 = z0.split('/')[-1][:-4]
    name_z1 = z1.split('/')[-1][:-4]

    # plot the model real quick
    z = []
    z_name = []
    z0 = []
    z1 = []
    idx = 0
    for s in subs:
        print('processing', s + ':')
        acts = os.listdir(os.path.join(input_path, s))
        for act in acts:
            if act != 'Directions':
                continue
            print('processing', s + '-' + act)
            file_list = glob(os.path.join(input_path, s, act) + '/*.mat')
            for f in file_list:
                data = scipy.io.loadmat(f)['latent'][0]
                z.append(data)
                z_name.append(f.split('/')[-2] + '/' + f.split('/')[-1])
                name = f.split('/')[-1][:-4]
                if s == s_z0 and act == act_z0 and name == name_z0:
                    z0.append(idx)
                if s == s_z1 and act == act_z1 and name == name_z1:
                    z1.append(idx)
                idx = idx + 1
    if len(z0) > 1 or len(z1) > 1:
        assert('multiple z0 or z1')

    print('predict model ...')
    z = np.asarray(z)
    x = m.predict(z)


    session = tf.Session()
    #session = k.get_session()
    session.run(tf.global_variables_initializer())
    print(session.run(m.input[0]), '---')

    print('create riemannian metric ...')
    rmetric = RiemannianMetric(x=m.output, z=m.input, session=session)
    rmetric.create_tf_graph()

    #mf = session.run(rmetric.MF, {rmetric.z: inp})

    # for steps in [100,1_000,10_000,100_000]:
    #     q = r.riemannian_distance_along_line(z1, z2, n_steps=steps)
    #     print(q)

    rTree = RiemannianTree(rmetric)
    print('create riemannian graph ...')
    G = rTree.create_riemannian_graph(z, n_steps=100, n_neighbors=10)

    # can use G to do shortest path finding now
    print('find shortest path ...')
    path = nx.shortest_path(G, source = z0[0], target = z1[0], weight = 'weight')
    print(path,' !!!')
    print([z_name[idx] for idx in path], ' !!')

if __name__ == '__main__':
    main()