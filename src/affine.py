import numpy as np
import ransac
import affine


def affine_transformation_estimation(kp1, kp2, matches_pos):
    # we obtain the keypoints that match between the reference image and the current frame
    kp1_m = kp1[:, matches_pos[:, 0]]
    kp2_m = kp2[:, matches_pos[:, 1]]

    # we apply ransac to estimate an affine transformation from the matches
    m, b, inliers = ransac.do_ransac(kp1_m, kp2_m)

    # we obtain the inliers
    kp1_in = kp1_m[:, inliers[0]]
    kp2_in = kp2_m[:, inliers[0]]

    # we obtain the final affine matrix from the inliers
    m, b = affine.affine_transformation_hypothesis(kp1_in, kp2_in)
    affine_matrix = np.hstack((m, b))

    return affine_matrix


def affine_transformation_hypothesis(kp1, kp2):
    x = np.zeros((2 * len(kp1[0]), 6))

    m = None
    b = None

    # we construct a general matrix to estimate an affine transformation from at least three points
    for i in range(len(kp1[0])):
        x[2 * i: 2 * i + 2, :] = [[kp1[0,i], kp1[1,i],        0,        0, 1, 0],
                                  [       0,        0, kp1[0,i], kp1[1,i], 0, 1]]

    y = np.transpose(kp2).reshape((2 * len(kp2[0]), 1))

    # we solve this equation from matrix operations
    try:
        xt = np.transpose(x)
        a = np.dot(np.linalg.inv(np.dot(xt, x)), np.dot(xt, y))
        m = a[:4].reshape((2, 2))
        b = a[4:]
    except:
        pass

    return m, b