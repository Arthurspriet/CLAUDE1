"""Video generation tools for LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claude1.config import (
    DEFAULT_VIDEO_FRAMES,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_MODEL_T2V,
    DEFAULT_VIDEO_MODEL_I2V,
)
from claude1.tools.base import BaseTool

# Lazy import to avoid breaking if dependencies aren't installed
try:
    from claude1.video_generation import VideoGenerator
    from claude1.video_models import get_model_info, list_models
    _VIDEO_DEPS_AVAILABLE = True
except ImportError:
    _VIDEO_DEPS_AVAILABLE = False
    VideoGenerator = None  # type: ignore
    get_model_info = None  # type: ignore
    list_models = None  # type: ignore


class TextToVideoTool(BaseTool):
    """Tool for generating videos from text prompts."""

    def __init__(self, working_dir: str):
        """Initialize the text-to-video tool."""
        super().__init__(working_dir)
        if not _VIDEO_DEPS_AVAILABLE:
            raise ImportError(
                "Video generation dependencies not installed. "
                "Install with: pip install diffusers torch torchvision transformers accelerate imageio[ffmpeg]"
            )
        self._generator = VideoGenerator()

    @property
    def name(self) -> str:
        return "text_to_video"

    @property
    def description(self) -> str:
        return (
            "Generate a video from a text prompt using AI video generation models. "
            "Supports multiple models optimized for RTX 5090 GPU. "
            "Videos are saved to ~/.claude1/videos/ and returned as file paths. "
            "This operation can take 30-120 seconds depending on model and settings."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the video to generate (required)",
                },
                "model": {
                    "type": "string",
                    "description": f"Model ID to use (optional, defaults to {DEFAULT_VIDEO_MODEL_T2V}). "
                    "Available models: damo-vilab/text-to-video-ms-1.7b, THUDM/CogVideoX-5b, cerspense/zeroscope-v2-576w",
                },
                "num_frames": {
                    "type": "integer",
                    "description": f"Number of frames to generate (default: {DEFAULT_VIDEO_FRAMES})",
                    "minimum": 1,
                    "maximum": 100,
                },
                "height": {
                    "type": "integer",
                    "description": f"Video height in pixels (default: {DEFAULT_VIDEO_HEIGHT})",
                    "minimum": 128,
                    "maximum": 1024,
                },
                "width": {
                    "type": "integer",
                    "description": f"Video width in pixels (default: {DEFAULT_VIDEO_WIDTH})",
                    "minimum": 128,
                    "maximum": 1024,
                },
                "fps": {
                    "type": "integer",
                    "description": f"Frames per second (default: {DEFAULT_VIDEO_FPS})",
                    "minimum": 1,
                    "maximum": 30,
                },
                "output_filename": {
                    "type": "string",
                    "description": "Optional output filename without extension (auto-generated if not provided)",
                },
            },
            "required": ["prompt"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        """Execute text-to-video generation."""
        prompt = kwargs.get("prompt", "")
        if not prompt:
            return "Error: prompt is required"

        model_id = kwargs.get("model")
        num_frames = kwargs.get("num_frames")
        height = kwargs.get("height")
        width = kwargs.get("width")
        fps = kwargs.get("fps")
        output_filename = kwargs.get("output_filename")

        try:
            output_path = self._generator.generate_text_to_video(
                prompt=prompt,
                model_id=model_id,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                output_filename=output_filename,
            )

            return f"Video generated successfully: {output_path}\nFile size: {output_path.stat().st_size / (1024*1024):.2f} MB"

        except RuntimeError as e:
            return f"Error generating video: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"


class ImageToVideoTool(BaseTool):
    """Tool for generating videos from input images."""

    def __init__(self, working_dir: str):
        """Initialize the image-to-video tool."""
        super().__init__(working_dir)
        if not _VIDEO_DEPS_AVAILABLE:
            raise ImportError(
                "Video generation dependencies not installed. "
                "Install with: pip install diffusers torch torchvision transformers accelerate imageio[ffmpeg]"
            )
        self._generator = VideoGenerator()

    @property
    def name(self) -> str:
        return "image_to_video"

    @property
    def description(self) -> str:
        return (
            "Generate a video from an input image using AI video generation models. "
            "Supports multiple models optimized for RTX 5090 GPU. "
            "Videos are saved to ~/.claude1/videos/ and returned as file paths. "
            "This operation can take 30-120 seconds depending on model and settings."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image file (required). Can be relative to working directory or absolute.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional text prompt to guide video generation (not all models use this)",
                },
                "model": {
                    "type": "string",
                    "description": f"Model ID to use (optional, defaults to {DEFAULT_VIDEO_MODEL_I2V}). "
                    "Available models: stabilityai/stable-video-diffusion-img2vid-xt, THUDM/CogVideoX-5b-I2V",
                },
                "num_frames": {
                    "type": "integer",
                    "description": f"Number of frames to generate (default: {DEFAULT_VIDEO_FRAMES})",
                    "minimum": 1,
                    "maximum": 100,
                },
                "height": {
                    "type": "integer",
                    "description": f"Video height in pixels (default: model-specific)",
                    "minimum": 128,
                    "maximum": 1024,
                },
                "width": {
                    "type": "integer",
                    "description": f"Video width in pixels (default: model-specific)",
                    "minimum": 128,
                    "maximum": 1024,
                },
                "fps": {
                    "type": "integer",
                    "description": f"Frames per second (default: {DEFAULT_VIDEO_FPS})",
                    "minimum": 1,
                    "maximum": 30,
                },
                "output_filename": {
                    "type": "string",
                    "description": "Optional output filename without extension (auto-generated if not provided)",
                },
            },
            "required": ["image_path"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        """Execute image-to-video generation."""
        image_path = kwargs.get("image_path", "")
        if not image_path:
            return "Error: image_path is required"

        # Resolve image path
        image_path = self._resolve_path(image_path)

        prompt = kwargs.get("prompt")
        model_id = kwargs.get("model")
        num_frames = kwargs.get("num_frames")
        height = kwargs.get("height")
        width = kwargs.get("width")
        fps = kwargs.get("fps")
        output_filename = kwargs.get("output_filename")

        try:
            output_path = self._generator.generate_image_to_video(
                image_path=image_path,
                prompt=prompt,
                model_id=model_id,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                output_filename=output_filename,
            )

            return f"Video generated successfully: {output_path}\nFile size: {output_path.stat().st_size / (1024*1024):.2f} MB"

        except RuntimeError as e:
            return f"Error generating video: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"

