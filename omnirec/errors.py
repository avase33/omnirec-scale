"""Exception hierarchy for OmniRec-Scale."""

from __future__ import annotations


class OmniRecError(Exception):
    """Base class for all OmniRec errors."""


class ConfigError(OmniRecError):
    """Invalid configuration."""


class NotTrainedError(OmniRecError):
    """A model was used for inference before being trained/loaded."""


class IndexError_(OmniRecError):
    """Vector-index failure."""


class FeatureStoreError(OmniRecError):
    """Feature-store read/write failure."""


class ServingError(OmniRecError):
    """Serving/funnel failure."""
