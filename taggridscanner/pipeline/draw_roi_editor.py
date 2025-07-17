import cv2

from taggridscanner.pipeline.draw_roi import DrawROI


class DrawROIEditor(DrawROI):
    def __init__(self, active_vertex=0):
        super().__init__()
        self.active_vertex_idx = active_vertex

    def __call__(self, image, roi_vertices):
        image = super().__call__(image, roi_vertices)
        self.draw_vertices(image, roi_vertices)
        self.label_vertices(image, roi_vertices)
        return image

    def draw_vertices(self, image, roi_vertices):
        circle_scale = max(image.shape[1] / 1920, image.shape[0] / 1080)
        r = round(10 * circle_scale)
        t = round(2 * circle_scale)

        outline_vertices = self.outline_vertices(roi_vertices)
        for idx, p in enumerate(outline_vertices):
            c = (0, 0, 255) if idx == self.active_vertex_idx else (0, 255, 0)
            cv2.circle(image, p, radius=r, color=c, thickness=t)

    def label_vertices(self, image, roi_vertices):
        self.label_vertex(image, roi_vertices, 0, True, True, prefix=" ")
        self.label_vertex(image, roi_vertices, 1, False, True, suffix=" ")
        self.label_vertex(image, roi_vertices, 2, False, False, suffix=" ")
        self.label_vertex(image, roi_vertices, 3, True, False, prefix=" ")

    def label_vertex(self, image, roi_vertices, idx, left, top, prefix="", suffix=""):
        text = prefix + "{0:d} [{1[0]:.2f}, {1[1]:.2f}]".format(idx, roi_vertices[idx]) + suffix
        roi_outline_vertices = DrawROI.outline_vertices(roi_vertices)
        pos = roi_outline_vertices[idx]
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
