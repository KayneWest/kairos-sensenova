from __future__ import annotations

from typing import Any


class ObservationAdapter:
    """
    Preprocesses incoming camera observations before encoding.

    The initial implementation is intentionally conservative: it leaves the
    frame unchanged unless a subclass overrides it.
    """

    def preprocess_frame(self, frame_rgb: Any) -> Any:
        return frame_rgb
