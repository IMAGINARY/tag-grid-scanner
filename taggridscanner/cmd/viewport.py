import sys
import cv2
import numpy
import numpy as np
from taggridscanner.aux.threading import WorkerThreadWithResult, ThreadSafeContainer
from taggridscanner.pipeline.preprocess import Preprocess
from taggridscanner.pipeline.retrieve_image import RetrieveImage
from taggridscanner.pipeline.view_image import ViewImage

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_16H5)
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

detector_parameters = cv2.aruco.DetectorParameters_create()
detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
# detector_parameters.minMarkerDistanceRate = 0.01
#
# detector_parameters.perspectiveRemovePixelPerCell = 2
# detector_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.05


def viewport(args):
    config_with_defaults = args["config-with-defaults"]

    retrieve_image = RetrieveImage.create_from_config(config_with_defaults)
    retrieve_image.scale = (1.0, 1.0)
    retrieve_image_worker = WorkerThreadWithResult(retrieve_image)

    retrieve_image_worker.start()

    preprocess = Preprocess.create_from_config(config_with_defaults)

    view_image = ViewImage("Snapshot")

    print(
        "Press ESC or q to quit.",
        file=sys.stderr,
    )

    #    frame = preprocess(retrieve_image_worker.result.retrieve())
    frame = cv2.imread("marker/marker1-2-undistorted.png")

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
        frame, aruco_dict, parameters=detector_parameters
    )

    print("corners:", corners)

    enlarged_corners = []
    centroid_of_centroids = np.array([[[0.0, 0.0]]])
    for marker_corners in corners:
        print("points", marker_corners)
        centroid = np.mean(marker_corners, axis=1)
        centroid_of_centroids += centroid
        print("centroid", centroid)

        enlarged_corners.append(
            np.add(np.subtract(marker_corners, centroid) * (8.0 / 6.0), centroid)
        )
    centroid_of_centroids /= len(corners)
    print("centroid_of_centroids", centroid_of_centroids)

    outer_corners = []
    for marker_corners in enlarged_corners:
        dists = np.linalg.norm(marker_corners - centroid_of_centroids, ord=2, axis=2)
        argmax = np.argmax(dists, axis=1)
        print("dist", dists, "argmax", argmax)
        outer_corner = marker_corners[0][argmax]
        outer_corners.append(outer_corner[0])
    print("outer_corners", outer_corners)
    outer_corners = np.array(outer_corners, dtype=np.int32)

    print("outer_corners", outer_corners)

    cv2.aruco.drawDetectedMarkers(frame, enlarged_corners, ids)

    cv2.polylines(frame, [outer_corners], isClosed=True, color=(255, 0, 0))

    print(ids, enlarged_corners)
    print()

    view_image(frame)

    cv2.imwrite("roi_from_markers.jpg", frame)

    cv2.waitKey()
