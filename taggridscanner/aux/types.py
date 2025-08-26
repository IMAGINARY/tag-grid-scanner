from typing import TypedDict, Tuple

MarkerIds = (int, int, int, int)
Point2f = (float, float)
Point2f4 = (Point2f, Point2f, Point2f, Point2f)


class MarkerIdWithCenter:
    """Class for storing the center a marker."""

    def __init__(self, id: int, center: Point2f):
        self.id = id
        self.center = center

    id: int
    center: Point2f

    def __str__(self):
        return f"MarkerIdWithCenter(id={self.id}, center={self.center})"

    def __repr__(self):
        return self.__str__()


class MarkerIdWithCorners(MarkerIdWithCenter):
    """Class for storing the corners of a marker."""

    def __init__(self, id: int, corners: Point2f4):
        """Initializes a MarkerIdWithCorners instance. Corners must be given in clockwise order (that's what OpenCV provides)."""
        super().__init__(
            id,
            MarkerIdWithCorners.projected_quad_center(corners),
        )
        self.corners = corners

    corners: Point2f4

    @staticmethod
    def intersect_lines(l1_p1: Point2f, l1_p2: Point2f, l2_p1: Point2f, l2_p2: Point2f) -> Point2f:
        """Finds the intersection point of two lines defined by points p1, p2 and p3, p4.
        Each line is defined by two points (p1, p2) and (p3, p4).
        Returns the intersection point as a tuple (x, y).
        If the lines are parallel, returns None.
        """
        x1, y1 = l1_p1
        x2, y2 = l1_p2
        x3, y3 = l2_p1
        x4, y4 = l2_p2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Lines are parallel

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

        return px, py

    @staticmethod
    def projected_quad_center(points: Point2f4) -> Point2f:
        """Computes the center of a projected quadrilateral defined by four points by intersection its diagonals."""
        if points[0] == points[1]:
            # points[0] and points[1] form a vanishing point
            return [points[0][0], points[0][1]]

        if points[1] == points[2]:
            # points[1] and points[2] form a vanishing point
            return [points[1][0], points[1][1]]

        if points[2] == points[3]:
            # points[2] and points[3] form a vanishing point
            return [points[2][0], points[2][1]]

        if points[3] == points[0]:
            # points[3] and points[0] form a vanishing point
            return [points[3][0], points[3][1]]

        # The case that all points are the same is also covered above
        center = MarkerIdWithCorners.intersect_lines(points[0], points[2], points[1], points[3])
        assert center is not None, "The quads diagonals should not be parallel"

        return center

    def __str__(self):
        return f"MarkerIdWithCorners(id={self.id}, corners={self.corners}, center={self.center})"

    def __repr__(self):
        return self.__str__()


ROIMarkers = Tuple[MarkerIdWithCorners, MarkerIdWithCorners, MarkerIdWithCorners, MarkerIdWithCorners]


class MarkersForVis(TypedDict):
    matched: list[MarkerIdWithCorners]
    remaining: list[MarkerIdWithCorners]
    not_on_hull: list[MarkerIdWithCorners]
