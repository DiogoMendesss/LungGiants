"""Shared class semantics for indexing, transforms, training, and reporting."""

CLASS_CHOICES = ("nodule", "lung+nodule")
LUNG_LOBE_LABELS = (28, 29, 30, 31, 32)


def validate_classes(classes):
    if classes not in CLASS_CHOICES:
        raise ValueError(f"classes must be one of {CLASS_CHOICES}, got {classes!r}")
    return classes


def class_ids(classes, include_background=False):
    """Return class names and target IDs in a stable order."""
    validate_classes(classes)
    ids = {"nodule": 1} if classes == "nodule" else {"lung": 1, "nodule": 2}
    return {"background": 0, **ids} if include_background else ids


def output_channels(classes):
    validate_classes(classes)
    return 1 if classes == "nodule" else 3


def mask_keys(classes):
    validate_classes(classes)
    return ["nod_mask"] if classes == "nodule" else ["nod_mask", "ts_mask"]


def predict_labels(logits, classes):
    """Decode raw logits to [B, 1, X, Y, Z] integer class maps."""
    if logits.shape[1] != output_channels(classes):
        raise ValueError(f"Unexpected output channels for {classes}: {logits.shape[1]}")
    if classes == "nodule":
        return (logits > 0).long()
    return logits.argmax(dim=1, keepdim=True)
