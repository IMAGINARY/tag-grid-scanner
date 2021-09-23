import cv2


def save_coefficients(mtx, dist, path):
    """Save the camera matrix and the distortion coefficients to given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    """Loads camera matrix and distortion coefficients."""
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def compute_abs_roi_size(frame_size, margin_trbl):
    roi_width = frame_size[0] - (margin_trbl[1] + margin_trbl[3])
    roi_height = frame_size[1] - (margin_trbl[0] + margin_trbl[2])
    return roi_width, roi_height


def compute_rel_roi_size(frame_size, margin_trbl):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return abs_roi_size[0] / frame_size[0], abs_roi_size[0] / frame_size[0]


def compute_rel_margin_trbl(frame_size, margin_trbl):
    return (
        margin_trbl[0] / frame_size[0],
        margin_trbl[1] / frame_size[1],
        margin_trbl[2] / frame_size[0],
        margin_trbl[3] / frame_size[1],
    )


def compute_rel_gap(frame_size, margin_trbl, gaps):
    abs_roi_size = compute_abs_roi_size(frame_size, margin_trbl)
    return gaps[0] / abs_roi_size[0], gaps[1] / abs_roi_size[1]


def get_rotate_code(degrees):
    rotate_codes = {
        0: None,
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return rotate_codes.get(degrees)


def get_flip_code(flip_h, flip_v):
    if flip_v and not flip_h:
        return 0
    elif not flip_v and flip_h:
        return 1
    elif flip_v and flip_h:
        return -1
    else:
        return None


def create_linear_transformer(rotate, flip_h, flip_v):
    rotate_code = get_rotate_code(rotate)
    flip_code = get_flip_code(flip_h, flip_v)

    def linear_transformer(img):
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
        return img

    return linear_transformer


def create_inverse_linear_transformer(rotate, flip_h, flip_v):
    rotate_code = get_rotate_code((360 - rotate) % 360)
    flip_code = get_flip_code(flip_h, flip_v)

    def inverse_linear_transformer(img):
        if flip_code is not None:
            img = cv2.flip(img, flip_code)
        if rotate_code is not None:
            img = cv2.rotate(img, rotate_code)
        return img

    return inverse_linear_transformer


def create_preprocessor(camera_config):
    camera_matrix, distortion_coefficients = (
        load_coefficients(camera_config["calibration"])
        if "calibration" in camera_config
        else (None, None)
    )

    linear_transform = create_linear_transformer(
        camera_config["rotate"], camera_config["flipH"], camera_config["flipV"]
    )

    def preprocess(img):
        if camera_matrix is not None and distortion_coefficients is not None:
            img = cv2.undistort(img, camera_matrix, distortion_coefficients, None, None)
        return linear_transform(img)

    return preprocess
