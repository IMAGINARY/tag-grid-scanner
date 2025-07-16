from typing import Union

from taggridscanner.aux.utils import Functor
from taggridscanner.aux.types import ROIMarkers, MarkersForVis


class TrackNoMarkers(Functor):
    # This class is used in the pipeline when no markers are to be tracked.
    def __init__(self):
        super().__init__()

    def __call__(self, image, roi_markers: ROIMarkers) -> (Union[ROIMarkers, None], MarkersForVis):
        return None, {"matched": [], "remaining": [], "not_on_hull": []}
