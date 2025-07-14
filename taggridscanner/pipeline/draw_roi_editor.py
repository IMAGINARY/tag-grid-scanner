import cv2

from taggridscanner.pipeline.draw_roi import DrawROI


class DrawROIEditor(DrawROI):
    def __init__(self, rel_vertices, active_vertex=0):
        super().__init__(rel_vertices)
        self.active_vertex_idx = active_vertex

    def __call__(self, image, marker_homography_matrix):
        image = super().__call__(image, marker_homography_matrix)
        self.draw_vertices(image)
        self.label_vertices(image)
        return image

    def draw_vertices(self, image):
        circle_scale = max(image.shape[1] / 1920, image.shape[0] / 1080)
        r = round(10 * circle_scale)
        t = round(2 * circle_scale)

        outline_vertices = self.outline_vertices(image.shape)
        for idx, p in enumerate(outline_vertices):
            c = (0, 0, 255) if idx == self.active_vertex_idx else (0, 255, 0)
            cv2.circle(image, p, radius=r, color=c, thickness=t)

    def label_vertices(self, image):
        self.label_vertex(image, 0, True, True, prefix=" ")
        self.label_vertex(image, 1, False, True, suffix=" ")
        self.label_vertex(image, 2, False, False, suffix=" ")
        self.label_vertex(image, 3, True, False, prefix=" ")

    def label_vertex(self, image, idx, left, top, prefix="", suffix=""):
        rv = self.abs_vertices(image.shape)
        text = prefix + "{0:d} [{1[0]:.2f}, {1[1]:.2f}]".format(idx, rv[idx]) + suffix
        pos = self.outline_vertices(image.shape)[idx]
        color = (0, 0, 255) if idx == self.active_vertex_idx else (0, 255, 0)
        label(image, text, pos, left, top, color)


def label(img, text, pos, left, top, color):
    font_scale = max(img.shape[1] / 1920, img.shape[0] / 1080)

    text_size, baseline = cv2.getTextSize(
        text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=round(2 * font_scale),
    )

    text_x = pos[0] if left else pos[0] - text_size[0]
    text_y = pos[1] + text_size[1] + baseline if top else pos[1] - baseline

    cv2.putText(
        img,
        text,
        (int(text_x), int(text_y)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=color,
        thickness=round(2 * font_scale),
    )
