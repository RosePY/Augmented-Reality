import numpy as np
import affine


# we determine the root squared error from the estimated keypoints using the affine transformation and the actual keypoints
def root_squared_error(m, b, kp1, kp2):
    res = None
    if not (m is None) and not (b is None):
        kp2_est = np.dot(m, kp1) + b
        diff_square = (kp2_est - kp2) * (kp2_est - kp2)
        res = np.sqrt(np.sum(diff_square, axis=0))
    return res


def do_ransac(kp1, kp2, iter_num=2000, n_random=3, threshold=1):
    m = None
    b = None
    inliers_idx = None
    n_inliers = 0

    # for a given number of iterations we estimate an affine transformation
    for i in range(iter_num):
        # we pick arbitrarily three points from the matches
        idx = np.random.randint(0, len(kp1[0]), (n_random, 1))

        # we estimate the affine transformation matrix
        m_tmp, b_tmp = affine.affine_transformation_hypothesis(kp1[:, idx], kp2[:, idx])

        # we determine the root squared error from the estimated points
        sq_er = root_squared_error(m_tmp, b_tmp, kp1, kp2)

        if not(sq_er is None):
            # if the error is below a certain threshold then we update the values of the affine transformation
            inliers_idxp = np.where(sq_er < threshold)

            if len(inliers_idxp[0]) > n_inliers:
                n_inliers = len(inliers_idxp[0])
                inliers_idx = inliers_idxp
                m = m_tmp
                b = b_tmp
        else:
            continue

    return m, b, inliers_idx