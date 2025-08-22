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
        super().__init__(
            id,
            (
                (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4,
                (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4,
            ),
        )
        self.corners = corners

    corners: Point2f4

    def __str__(self):
        return f"MarkerIdWithCorners(id={self.id}, corners={self.corners}, center={self.center})"

    def __repr__(self):
        return self.__str__()


ROIMarkers = Tuple[MarkerIdWithCorners, MarkerIdWithCorners, MarkerIdWithCorners, MarkerIdWithCorners]


class MarkersForVis(TypedDict):
    matched: list[MarkerIdWithCorners]
    remaining: list[MarkerIdWithCorners]
    not_on_hull: list[MarkerIdWithCorners]
