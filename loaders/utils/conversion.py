import torch

__all__ = ['normalize_image']

MEANS = torch.tensor([0.485, 0.456, 0.406]).reshape((3, 1, 1))
STDS = torch.tensor([[0.229, 0.224, 0.225]]).reshape((3, 1, 1))


def normalize_image(image: torch.tensor) -> torch.tensor:
    """ Check for usefulness in morphology, because this immediately makes it
    correlation-like values. """
    image -= MEANS
    return image / STDS
