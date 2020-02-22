import numpy as np
import cv2

# extracting descriptors from images using ORB
def extract_features(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kpo, des = orb.detectAndCompute(grayscale, None)
    kp = np.array([pt.pt for pt in kpo]).T
    return kp, kpo, des

# pasting overlying image
def pasting_overlay(source, target, overlay, M):
    rows, cols, _ = target.shape

    h, w = source.shape[:2]
    # obtaining corners from reference image
    corners = np.array([[0, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1], [w - 1, 0, 1]])

    corner_dst = []

    # using the affine matrix we obtain the correspondent corners for
    for corner in corners:
        corner_dst.append(np.dot(corner, M.T))

    corner_dst = np.array(corner_dst)

    pts_2 = np.float32([[0, 0], [overlay.shape[1], 0], [overlay.shape[1], overlay.shape[0]], [0, overlay.shape[0]]])
    pts_1 = np.float32([[int(corner_dst[3, 0]), int(corner_dst[3, 1])], [int(corner_dst[0, 0]), int(corner_dst[0, 1])],
                        [int(corner_dst[1, 0]), int(corner_dst[1, 1])], [int(corner_dst[2, 0]), int(corner_dst[2, 1])]])

    # getting perspective transform matrix from corners
    pt = cv2.getPerspectiveTransform(pts_2, pts_1)
    # warping overlying image
    dst_image = cv2.warpPerspective(overlay, pt, (target.shape[1], target.shape[0]))
    # creating mask for the overlying image and pasting over the target frame
    dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(dst_image_gray, 0, 255, cv2.THRESH_BINARY_INV)
    image_masked = cv2.bitwise_and(target, target, mask=mask)
    frame = cv2.add(dst_image, image_masked)

    return cv2.polylines(frame, [np.int32(corner_dst)], True, (0, 255, 255), 1)