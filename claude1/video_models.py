"""Video generation model registry and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class VideoModelInfo:
    """Metadata for a video generation model."""

    model_id: str
    name: str
    capabilities: list[Literal["text-to-video", "image-to-video"]]
    description: str
    memory_gb: float  # Approximate VRAM requirement in GB
    recommended_frames: int
    recommended_height: int
    recommended_width: int
    recommended_fps: int


# Registry of available video generation models
VIDEO_MODELS: dict[str, VideoModelInfo] = {
    "damo-vilab/text-to-video-ms-1.7b": VideoModelInfo(
        model_id="damo-vilab/text-to-video-ms-1.7b",
        name="ModelScopeT2V",
        capabilities=["text-to-video"],
        description="Text-to-video model with 1.7B parameters, good quality for RTX 5090",
        memory_gb=8.0,
        recommended_frames=25,
        recommended_height=512,
        recommended_width=512,
        recommended_fps=8,
    ),
    "THUDM/CogVideoX-5b": VideoModelInfo(
        model_id="THUDM/CogVideoX-5b",
        name="CogVideoX",
        capabilities=["text-to-video"],
        description="CogVideoX 5B parameter model supporting text-to-video generation",
        memory_gb=12.0,
        recommended_frames=25,
        recommended_height=480,
        recommended_width=832,
        recommended_fps=8,
    ),
    "THUDM/CogVideoX-5b-I2V": VideoModelInfo(
        model_id="THUDM/CogVideoX-5b-I2V",
        name="CogVideoX-I2V",
        capabilities=["image-to-video"],
        description="CogVideoX image-to-video variant, generates video from input image",
        memory_gb=12.0,
        recommended_frames=25,
        recommended_height=480,
        recommended_width=832,
        recommended_fps=8,
    ),
    "stabilityai/stable-video-diffusion-img2vid-xt": VideoModelInfo(
        model_id="stabilityai/stable-video-diffusion-img2vid-xt",
        name="Stable Video Diffusion (SVD)",
        capabilities=["image-to-video"],
        description="Stable Video Diffusion XT - high quality image-to-video generation",
        memory_gb=10.0,
        recommended_frames=25,
        recommended_height=576,
        recommended_width=1024,
        recommended_fps=7,
    ),
    "cerspense/zeroscope-v2-576w": VideoModelInfo(
        model_id="cerspense/zeroscope-v2-576w",
        name="Zeroscope v2",
        capabilities=["text-to-video"],
        description="Zeroscope v2 text-to-video model, good quality and speed",
        memory_gb=9.0,
        recommended_frames=24,
        recommended_height=320,
        recommended_width=576,
        recommended_fps=8,
    ),
}


def get_model_info(model_id: str) -> VideoModelInfo | None:
    """Get model information by model ID."""
    return VIDEO_MODELS.get(model_id)


def list_models(capability: Literal["text-to-video", "image-to-video"] | None = None) -> list[VideoModelInfo]:
    """List all available models, optionally filtered by capability."""
    if capability is None:
        return list(VIDEO_MODELS.values())
    return [model for model in VIDEO_MODELS.values() if capability in model.capabilities]


def get_default_model(capability: Literal["text-to-video", "image-to-video"]) -> str:
    """Get the default model ID for a given capability."""
    from claude1.config import DEFAULT_VIDEO_MODEL_T2V, DEFAULT_VIDEO_MODEL_I2V

    if capability == "text-to-video":
        return DEFAULT_VIDEO_MODEL_T2V
    return DEFAULT_VIDEO_MODEL_I2V

