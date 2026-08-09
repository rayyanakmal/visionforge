"""Pydantic models for VisionForge inputs: ground truth + predictions (COCO-style)."""

from pydantic import BaseModel, Field, field_validator


class GTImage(BaseModel):
    """One image in the ground truth dataset."""

    id: int
    file_name: str
    width: int
    height: int


class GTAnnotation(BaseModel):
    """One ground truth object: a bounding box in [x, y, w, h] format."""

    id: int
    image_id: int
    category_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def bbox_non_negative(cls, v: list[float]) -> list[float]:
        if any(x < 0 for x in v):
            raise ValueError(f"bbox values must be non-negative, got {v}")
        return v


class GTCategory(BaseModel):
    """One object category."""

    id: int
    name: str


class GroundTruth(BaseModel):
    """COCO-style ground truth: images, annotations, categories."""

    images: list[GTImage]
    annotations: list[GTAnnotation]
    categories: list[GTCategory]


class Prediction(BaseModel):
    """One model prediction: image, category, bbox, confidence score."""

    image_id: int
    category_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    score: float = Field(ge=0.0, le=1.0)


class PredictionsFile(BaseModel):
    """Container for a predictions file (COCO results format).

    Accepts either a bare list of detection dicts (standard COCO results
    format, what YOLO/DETR/etc export) or an object with a "predictions" key.
    """

    predictions: list[Prediction]

    @classmethod
    def load(cls, data: dict | list) -> "PredictionsFile":
        if isinstance(data, list):
            return cls(predictions=data)
        return cls.model_validate(data)


def load_predictions(path: str) -> PredictionsFile:
    """Load a predictions file from JSON, accepting COCO results (bare list)
    or wrapped {"predictions": [...]} format."""
    import json

    with open(path) as f:
        data = json.load(f)
    return PredictionsFile.load(data)


def load_ground_truth(path: str) -> GroundTruth:
    """Load a COCO-style ground truth JSON file."""
    import json

    with open(path) as f:
        data = json.load(f)
    return GroundTruth.model_validate(data)


def validate_consistency(gt: GroundTruth, preds: PredictionsFile) -> list[str]:
    """Warn when predictions reference categories/images missing from GT.

    This catches the silent zero-match trap: a prediction with a category_id
    that doesn't exist in GT categories will never match, producing misleading
    metrics with no error.
    """
    warnings: list[str] = []
    gt_cat_ids = {c.id for c in gt.categories}
    gt_img_ids = {i.id for i in gt.images}
    pred_cat_ids = {p.category_id for p in preds.predictions}
    pred_img_ids = {p.image_id for p in preds.predictions}

    unknown_cats = pred_cat_ids - gt_cat_ids
    if unknown_cats:
        warnings.append(
            f"predictions reference category_ids {sorted(unknown_cats)} "
            f"not present in GT categories — these will never match"
        )

    unknown_imgs = pred_img_ids - gt_img_ids
    if unknown_imgs:
        warnings.append(
            f"predictions reference image_ids {sorted(unknown_imgs)} "
            f"not present in GT images — these will never match"
        )

    return warnings
