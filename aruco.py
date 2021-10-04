import numpy as np
import cv2
import random


def tag_hamming_distance(tag_0, tag_1):
    hamming_distance = tag_0.shape[0] * tag_0.shape[1]
    for rot in range(0, 4):
        tag_rot_1 = np.rot90(tag_1, rot)
        tag_diff = tag_0 ^ tag_rot_1
        hamming_distance = min(hamming_distance, np.count_nonzero(tag_diff))
    return hamming_distance


def check_hamming_distance(tags, distance):
    for tag_id_0, tag_0 in enumerate(tags):
        for tag_id_1 in range(tag_id_0 + 1, len(tags)):
            tag_1 = tags[tag_id_1]
            if tag_hamming_distance(tag_0, tag_1) < distance:
                return False
    return True


def find_set_with_hamming_distance(tags, num, distance, num_tries):
    tag_ids = list(range(0, len(tags)))
    for _ in range(0, num_tries):
        selected_ids = random.sample(tag_ids, num)
        selected_tags = list(map(lambda idx: tags[idx], selected_ids))
        if check_hamming_distance(selected_tags, distance):
            return selected_tags, selected_ids
    return None, None


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16H5)

all_tags = []

for tag_id in range(0, 30):
    img = cv2.aruco.drawMarker(aruco_dict, tag_id, 10 * (4 + 2))
    tag = cv2.aruco.drawMarker(aruco_dict, tag_id, 6) // int(255)
    all_tags.append(tag)
    #    print(tag)
    # cv2.imshow("aruco marker {}".format(tag_id), img)

random.seed(0)

tags, ids = find_set_with_hamming_distance(all_tags, 6, 6, 10000000)
for tag_id in ids:
    tag_img = cv2.aruco.drawMarker(aruco_dict, tag_id, 6 * 40)
    cv2.imshow("AprilTag 16h5-{}".format(tag_id), tag_img)
print(tags, ids)

cv2.waitKey()

# distances = np.zeros((len(all_tags), len(all_tags)), dtype=np.uint32)
# for tag_id_0, tag_0 in enumerate(all_tags):
#    for tag_id_1, tag_1 in enumerate(all_tags):
#        distances[tag_id_0, tag_id_1] = tag_hamming_distance(tag_0, tag_1)
# print(distances)


# cv2.waitKey()
