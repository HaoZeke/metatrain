"""UPET: torch.compile compatible PET for metatrain."""

from .model import UPETModel
from .trainer import Trainer

__model__ = UPETModel
__trainer__ = Trainer

__authors__ = [
    ("Rohit Goswami <rgoswami@ieee.org>", "@HaoZeke"),
    ("Marcel Langer <mail@marcel.science>", "@sirmarcel"),
]

__maintainers__ = [
    ("Rohit Goswami <rgoswami@ieee.org>", "@HaoZeke"),
    ("Marcel Langer <mail@marcel.science>", "@sirmarcel"),
]
