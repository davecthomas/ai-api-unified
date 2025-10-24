"""
audio_models.py
---------------

Defines the AudioFormat model used to describe supported audio
output formats (e.g., mp3_44100_128).
"""

from typing import Final

from pydantic import BaseModel, Field, ConfigDict


class AudioFormat(BaseModel):
    """
    Immutable record describing an audio output format.

    Attributes
    ----------
    key : str
        Vendor-agnostic identifier, for example "mp3_24000".
    description : str
        Human-readable label for UI drop-downs.
    file_extension : str
        Extension beginning with a dot, for example ".mp3".
    sample_rate_hz : int | None
        Nominal sample rate in hertz.  Use None when the provider
        ignores the value (e.g. mu-law 8 kHz).
    """

    key: str = Field(..., description="Canonical identifier.", examples=["mp3_24000"])
    description: str = Field(
        ..., description="Human-readable description.", examples=["MP3 - 24 kHz"]
    )
    file_extension: str = Field(
        ...,
        pattern=r"^\.[a-z0-9]+$",
        description="File extension including the leading dot.",
        examples=[".mp3"],
    )
    sample_rate_hz: int | None = Field(
        default=None,
        description="Sample rate in hertz, or None if not applicable.",
        examples=[24000],
    )

    # Pydantic-v2 configuration ----------------------------------------------
    model_config: Final[ConfigDict] = ConfigDict(
        frozen=True,  # instances are hashable / read-only
        extra="forbid",  # reject unexpected keys
        populate_by_name=True,  # allow aliases if you add them later
    )
