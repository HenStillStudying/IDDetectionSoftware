"""Runs PaddleOCR over an image and normalizes its output into a flat list
of text lines with geometry, so the rest of the service doesn't need to know
about PaddleOCR's result-dict shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2


def run_ocr(ocr_engine, image: PILImage) -> list[TextLine]:
    results = ocr_engine.predict(np.array(image.convert("RGB")))
    lines: list[TextLine] = []
    for page in results:
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        boxes = page.get("rec_boxes")
        polys = page.get("rec_polys")

        for i, text in enumerate(texts):
            if boxes is not None:
                x1, y1, x2, y2 = (float(v) for v in boxes[i])
            else:
                poly = np.asarray(polys[i])
                x1, y1 = poly[:, 0].min(), poly[:, 1].min()
                x2, y2 = poly[:, 0].max(), poly[:, 1].max()

            lines.append(
                TextLine(
                    text=text,
                    confidence=float(scores[i]),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
    return lines
