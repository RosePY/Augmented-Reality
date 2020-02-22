import cv2
import utils
import matching
import affine


def main():
    img_ref = cv2.imread('../input/target.jpeg')
    overlay = cv2.imread('../input/overlay.jpg')
    cap = cv2.VideoCapture('../input/input.mp4')

    video_array = []

    # extracting descriptors for reference image
    kp1, kpo1, des1 = utils.extract_features(img_ref)

    frame_counter = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        # reading each frame of the video
        ret, frame = cap.read()

        # save the video when reading is over
        if ret is False or (cv2.waitKey(1) & 0xFF == ord('q')):
            out = cv2.VideoWriter('../output/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15,
                                  (video_array[0].shape[1], video_array[0].shape[0]))

            # save each frame to video
            for i in range(len(video_array)):
                out.write(video_array[i])
            out.release()

            break

        # extracting descriptors for each frame of the video
        kp2, kpo2, des2 = utils.extract_features(frame)

        # matching the image descriptors
        matches, matches_pos = matching.match(des1, des2)

        # if number of matches it not at least three
        if len(matches) < 3:
            continue

        # obtaining the final affine matrix
        affine_matrix = affine.affine_transformation_estimation(kp1, kp2, matches_pos)

        # pasting the overlying image on each frame
        final_frame = utils.pasting_overlay(img_ref, frame, overlay, affine_matrix)

        video_array.append(final_frame)
        frame_counter += 1
        print('processing frame', frame_counter, 'from', total_frames)


if __name__ == '__main__':
    main()
