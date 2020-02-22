import numpy as np
import cv2

# hamming distance with a double penalization
def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return np.sum(2 * np.logical_not(np.equal(s1, s2)))

# brute force matching using the hamming distance
def match(des1, des2, threshold=40):
    matches = []
    matches_pos = np.array([], dtype=np.int32).reshape((0, 2))
    for i in range(len(des1)):
        match = cv2.DMatch()
        for j in range(len(des2)):
            distance = hamming_distance(des1[i], des2[j])
            if distance < match.distance and distance < threshold:
                match.distance = distance
                match.queryIdx = i
                match.trainIdx = j
        if match.queryIdx != -1:
            matches.append(match)
            matches_pos = np.vstack((matches_pos, np.array([match.queryIdx, match.trainIdx])))
    return matches, matches_pos
