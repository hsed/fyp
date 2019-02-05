import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def saveKeypoints(filename, keypoints):
    # Reshape one sample keypoints into one line
    keypoints = keypoints.reshape(keypoints.shape[0], -1)
    np.savetxt(filename, keypoints, fmt='%0.4f')

# from v2v_posenet
def computeDistAcc(pred, gt, dist):
        '''
        pred: (N, K, 3)
        gt: (N, K, 3)
        dist: (M, )
        return acc: (K, M)
        '''
        assert(pred.shape == gt.shape)
        assert(len(pred.shape) == 3)

        N, K = pred.shape[0], pred.shape[1]
        err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2))  # (N, K)

        acc = np.zeros((K, dist.shape[0]))

        for i, d in enumerate(dist):
            acc_d = (err_dist < d).sum(axis=0) / N
            acc[:,i] = acc_d

        return acc