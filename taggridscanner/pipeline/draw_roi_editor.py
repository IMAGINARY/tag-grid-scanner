import cv2

from taggridscanner.pipeline.draw_roi import DrawROI


class DrawROIEditor(DrawROI):
    def __init__(self, vertices, active_vertex=0):
        super().__init__(vertices)
        self.active_vertex = active_vertex

    def __call__(self, image):
        image = super().__call__(image)
        self.draw_vertices(image)
        self.label_vertices(image)
        return image

    def draw_vertices(self, image):
        for p in self.outline_vertices:
            cv2.circle(image, p, radius=10, color=(0, 255, 0), thickness=2)
        cv2.circle(
            image,
            self.outline_vertices[self.active_vertex],
            radius=10,
            color=(0, 0, 255),
            thickness=2,
        )

    def label_vertices(self, image):
        self.label_vertex(image, 0, True, True)
        self.label_vertex(image, 1, False, True)
        self.label_vertex(image, 2, False, False)
        self.label_vertex(image, 3, True, False)

    def label_vertex(self, image, idx, left, top):
        text = " {}: {}".format(idx, self.vertices[idx])
        pos = self.outline_vertices[idx]
        color = (0, 0, 255) if idx == self.active_vertex else (0, 255, 0)
        label(image, text, pos, left, top, color)


def label(img, text, pos, left, top, color):
    text_size, baseline = cv2.getTextSize(
        text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=2,
    )

    text_x = pos[0] if left else pos[0] - text_size[0]
    text_y = pos[1] + text_size[1] + baseline if top else pos[1] - baseline

    cv2.putText(
        img,
        text,
        (int(text_x), int(text_y)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness=2,
    )
