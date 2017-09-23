import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import icp

# Constants
N = 1000                                    # number of random points in the dataset
num_tests = 100                             # number of test iterations
dim = 3                                     # number of dimensions of the points
noise_sigma = .01                           # standard deviation error to be added
translation = .1                            # max translation of the test set
rotation = .1                               # max rotation (radians) of the test set


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def test_best_fit():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B = B + t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand()*rotation)
        B = np.dot(R, B.T).T

        # Find best fit transform
        start = time.time()
        T, R1, t1 = icp.best_fit_transform(B, A)
        total_time += time.time() - start

        # make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = B

        assert np.allclose(np.dot(T, C.T).T[:,0:3], A)  # T should transform B to A
        assert np.allclose(-t1, t)                      # t and t1 should be inverses
        assert np.allclose(R1.T, R)                     # R and R1 should be inverses

    print('best fit time: {:.3}'.format(total_time/num_tests))

    return

def test_icp():

    # Generate a random dataset
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(num_tests):

        B = np.copy(A)

        # Translate
        t = np.random.rand(dim)*translation
        B = B + t

        # Rotate
        R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
        B = np.dot(R, B.T).T

        # Add noise
        B = B + np.random.randn(N, dim) * noise_sigma

        # Run ICP
        start = time.time()
        T, d = icp.icp(B, A, tolerance=0.000001)
        total_time += time.time() - start

        # Make C a homogeneous representation of B
        C = np.ones((N, 4))
        C[:,0:3] = np.copy(B)

        # Transform C
        C = np.dot(T, C.T).T

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(A[:,0], A[:,1], A[:,2], c='r', marker='o')
        # ax.scatter(B[:,0], B[:,1], B[:,2], c='b', marker='o')
        # ax.scatter(C[:,0], C[:,1], C[:,2], edgecolors='g', marker='s', facecolors='none', s=80)
        # plt.show()

        assert np.allclose(C[:,0:3], A, atol=6*noise_sigma)       # T should transform B (or C) to A
        assert np.allclose(T[0:3,0:3].T, R, atol=6*noise_sigma)   # T and R should be inverses
        assert np.allclose(-T[0:3,3], t, atol=6*noise_sigma)      # T and t should be inverses

    print('icp time: {:.3}'.format(total_time/num_tests))

    return


if __name__ == "__main__":
    test_best_fit()
    test_icp()