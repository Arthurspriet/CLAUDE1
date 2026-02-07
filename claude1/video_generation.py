"""Core video generation service using Hugging Face Diffusers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from claude1.config import VIDEOS_DIR
from claude1.video_models import get_model_info, get_default_model


class VideoGenerator:
    """Service for generating videos using Hugging Face Diffusers models."""

    def __init__(self):
        """Initialize the video generator."""
        self._pipelines: dict[str, DiffusionPipeline] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

    def _get_pipeline(self, model_id: str, capability: str) -> DiffusionPipeline:
        """Get or load a pipeline for the given model."""
        cache_key = f"{model_id}:{capability}"

        if cache_key in self._pipelines:
            return self._pipelines[cache_key]

        # Load the appropriate pipeline based on capability
        if capability == "text-to-video":
            try:
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self._dtype,
                    variant="fp16" if self._dtype == torch.float16 else None,
                )
            except Exception as e:
                # Fallback to default pipeline loading
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self._dtype,
                )
        elif capability == "image-to-video":
            # For image-to-video, try specific pipeline classes
            try:
                from diffusers import StableVideoDiffusionPipeline

                if "stable-video" in model_id.lower():
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self._dtype,
                        variant="fp16" if self._dtype == torch.float16 else None,
                    )
                else:
                    # Generic pipeline for other I2V models
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self._dtype,
                    )
            except ImportError:
                # Fallback to generic pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self._dtype,
                )
        else:
            raise ValueError(f"Unknown capability: {capability}")

        # Move to device and enable optimizations
        pipeline = pipeline.to(self._device)

        # Enable memory optimizations for RTX 5090
        if self._device == "cuda":
            try:
                # Enable VAE slicing and tiling for memory efficiency
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                if hasattr(pipeline, "enable_vae_tiling"):
                    pipeline.enable_vae_tiling()
                # Enable CPU offloading if available
                if hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()
            except Exception:
                # If optimizations fail, continue without them
                pass

        # Cache the pipeline
        self._pipelines[cache_key] = pipeline
        return pipeline

    def generate_text_to_video(
        self,
        prompt: str,
        model_id: str | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        output_filename: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Path:
        """Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate
            model_id: Model to use (defaults to DEFAULT_VIDEO_MODEL_T2V)
            num_frames: Number of frames to generate
            height: Video height in pixels
            width: Video width in pixels
            fps: Frames per second
            output_filename: Optional output filename (without extension)
            progress_callback: Optional callback for progress updates (0.0-1.0)

        Returns:
            Path to the generated video file

        Raises:
            RuntimeError: If generation fails
        """
        if model_id is None:
            model_id = get_default_model("text-to-video")

        model_info = get_model_info(model_id)
        if model_info is None:
            raise RuntimeError(f"Unknown model: {model_id}")

        if "text-to-video" not in model_info.capabilities:
            raise RuntimeError(f"Model {model_id} does not support text-to-video")

        # Use model defaults if not specified
        num_frames = num_frames or model_info.recommended_frames
        height = height or model_info.recommended_height
        width = width or model_info.recommended_width
        fps = fps or model_info.recommended_fps

        try:
            pipeline = self._get_pipeline(model_id, "text-to-video")

            if progress_callback:
                progress_callback(0.1)

            # Generate video frames
            if progress_callback:
                progress_callback(0.3)

            # Different models may have different API signatures
            # ModelScope T2V returns frames in a specific format
            try:
                # Try standard text-to-video API
                result = pipeline(
                    prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    num_inference_steps=50,  # Default inference steps
                )
            except TypeError:
                # Fallback for models with different signatures
                result = pipeline(prompt, num_frames=num_frames)
            
            # For ModelScope pipeline, result is typically a dict with "frames" key
            # The frames are a list of PIL Images or numpy arrays

            if progress_callback:
                progress_callback(0.8)

            # Extract frames - handle different pipeline output formats
            frames = None
            if isinstance(result, dict):
                # Most pipelines return a dict with "frames" or "images" key
                if "frames" in result:
                    frames = result["frames"]
                elif "images" in result:
                    frames = result["images"]
                elif "videos" in result:
                    frames = result["videos"]
            elif hasattr(result, "frames"):
                frames = result.frames
            elif hasattr(result, "images"):
                frames = result.images
            elif isinstance(result, (list, tuple)):
                # Some pipelines return frames directly as a list
                frames = result
            else:
                # Try to get frames from the result object
                frames = result

            # Ensure frames is a list/array and convert to numpy if needed
            import numpy as np
            from PIL import Image

            if frames is None:
                raise RuntimeError("Could not extract frames from pipeline output")

            # Convert to list of numpy arrays with proper shape (H, W, 3)
            frame_list = []
            
            # Helper function to normalize a single frame
            def normalize_frame(frame):
                """Convert a frame to numpy array with shape (H, W, 3)."""
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                elif isinstance(frame, torch.Tensor):
                    if frame.is_cuda:
                        frame = frame.cpu()
                    frame = frame.detach().numpy()
                elif not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                
                # Handle different shapes
                if len(frame.shape) == 4:
                    # (batch, ...) - take first
                    frame = frame[0]
                if len(frame.shape) == 3:
                    if frame.shape[0] <= 4:
                        # (channels, height, width) - transpose to (height, width, channels)
                        frame = np.transpose(frame, (1, 2, 0))
                    # Ensure 3 channels
                    if frame.shape[2] == 1:
                        # Grayscale - convert to RGB
                        frame = np.repeat(frame, 3, axis=2)
                    elif frame.shape[2] > 3:
                        # Too many channels - take RGB
                        frame = frame[:, :, :3]
                elif len(frame.shape) == 2:
                    # Grayscale - convert to RGB
                    frame = np.stack([frame] * 3, axis=-1)
                
                # Normalize to uint8
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                
                # Final validation
                if len(frame.shape) != 3 or frame.shape[2] not in [1, 3, 4]:
                    raise RuntimeError(f"Invalid frame shape: {frame.shape}, expected (H, W, 3)")
                
                # Ensure exactly 3 channels (RGB)
                if frame.shape[2] != 3:
                    if frame.shape[2] == 1:
                        frame = np.repeat(frame, 3, axis=2)
                    elif frame.shape[2] == 4:
                        frame = frame[:, :, :3]  # Remove alpha channel
                
                return frame
            
            # Process frames
            # ModelScope returns frames as (batch, frames, height, width, channels) or (frames, height, width, channels)
            try:
                if isinstance(frames, np.ndarray):
                    if len(frames.shape) == 5:
                        # Shape: (batch, frames, height, width, channels)
                        # Remove batch dimension and extract each frame
                        frames = frames[0]  # Remove batch: now (frames, height, width, channels)
                        for i in range(frames.shape[0]):
                            frame_list.append(normalize_frame(frames[i]))
                    elif len(frames.shape) == 4:
                        # Could be (frames, channels, height, width) or (frames, height, width, channels)
                        # Check which dimension is channels
                        if frames.shape[1] <= 4:
                            # Likely (frames, channels, height, width) - transpose each frame
                            for i in range(frames.shape[0]):
                                frame = frames[i]
                                frame = np.transpose(frame, (1, 2, 0))  # (height, width, channels)
                                frame_list.append(normalize_frame(frame))
                        else:
                            # Likely (frames, height, width, channels)
                            for i in range(frames.shape[0]):
                                frame_list.append(normalize_frame(frames[i]))
                    elif len(frames.shape) == 3:
                        # Single frame: (channels, height, width) or (height, width, channels)
                        frame_list.append(normalize_frame(frames))
                    else:
                        frame_list.append(normalize_frame(frames))
                elif isinstance(frames, (list, tuple)):
                    for frame in frames:
                        frame_list.append(normalize_frame(frame))
                else:
                    frame_list.append(normalize_frame(frames))
            except Exception as e:
                # Add debugging info
                frames_type = type(frames).__name__
                frames_shape = getattr(frames, 'shape', 'no shape')
                if isinstance(frames, (list, tuple)) and len(frames) > 0:
                    first_frame_type = type(frames[0]).__name__
                    first_frame_shape = getattr(frames[0], 'shape', 'no shape')
                    raise RuntimeError(
                        f"Frame processing error: {e}. "
                        f"Frames type: {frames_type}, shape: {frames_shape}, "
                        f"First frame type: {first_frame_type}, shape: {first_frame_shape}"
                    )
                else:
                    raise RuntimeError(
                        f"Frame processing error: {e}. "
                        f"Frames type: {frames_type}, shape: {frames_shape}"
                    )
            
            frames = frame_list

            # Generate output filename
            if output_filename is None:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (" ", "-", "_")).strip()
                safe_prompt = safe_prompt.replace(" ", "_")
                output_filename = f"video_{timestamp}_{safe_prompt}"

            output_path = VIDEOS_DIR / f"{output_filename}.mp4"

            # Export video - export_to_video expects a list of numpy arrays with shape (H, W, 3)
            export_to_video(frames, str(output_path), fps=fps)

            if progress_callback:
                progress_callback(1.0)

            return output_path

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {e}") from e

    def generate_image_to_video(
        self,
        image_path: str | Path,
        prompt: str | None = None,
        model_id: str | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
        output_filename: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Path:
        """Generate a video from an input image.

        Args:
            image_path: Path to the input image
            prompt: Optional text prompt (not all models use this)
            model_id: Model to use (defaults to DEFAULT_VIDEO_MODEL_I2V)
            num_frames: Number of frames to generate
            height: Video height in pixels
            width: Video width in pixels
            fps: Frames per second
            output_filename: Optional output filename (without extension)
            progress_callback: Optional callback for progress updates (0.0-1.0)

        Returns:
            Path to the generated video file

        Raises:
            RuntimeError: If generation fails
        """
        if model_id is None:
            model_id = get_default_model("image-to-video")

        model_info = get_model_info(model_id)
        if model_info is None:
            raise RuntimeError(f"Unknown model: {model_id}")

        if "image-to-video" not in model_info.capabilities:
            raise RuntimeError(f"Model {model_id} does not support image-to-video")

        # Use model defaults if not specified
        num_frames = num_frames or model_info.recommended_frames
        height = height or model_info.recommended_height
        width = width or model_info.recommended_width
        fps = fps or model_info.recommended_fps

        # Load and prepare image
        from PIL import Image

        image_path = Path(image_path)
        if not image_path.exists():
            raise RuntimeError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            # Resize if needed
            if height and width:
                image = image.resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}") from e

        try:
            pipeline = self._get_pipeline(model_id, "image-to-video")

            if progress_callback:
                progress_callback(0.1)

            # Generate video frames
            if progress_callback:
                progress_callback(0.3)

            # Different models may have different API signatures
            try:
                # Try standard image-to-video API
                if prompt:
                    result = pipeline(
                        image,
                        prompt=prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=50,
                    )
                else:
                    result = pipeline(
                        image,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=50,
                    )
            except TypeError:
                # Fallback for models with different signatures
                if prompt:
                    result = pipeline(image, prompt=prompt, num_frames=num_frames)
                else:
                    result = pipeline(image, num_frames=num_frames)

            if progress_callback:
                progress_callback(0.8)

            # Extract frames - handle different pipeline output formats
            frames = None
            if isinstance(result, dict):
                # Most pipelines return a dict with "frames" or "images" key
                if "frames" in result:
                    frames = result["frames"]
                elif "images" in result:
                    frames = result["images"]
                elif "videos" in result:
                    frames = result["videos"]
            elif hasattr(result, "frames"):
                frames = result.frames
            elif hasattr(result, "images"):
                frames = result.images
            elif isinstance(result, (list, tuple)):
                # Some pipelines return frames directly as a list
                frames = result
            else:
                # Try to get frames from the result object
                frames = result

            # Ensure frames is a list/array and convert to numpy if needed
            import numpy as np
            from PIL import Image

            if frames is None:
                raise RuntimeError("Could not extract frames from pipeline output")

            # Convert to list of numpy arrays if needed
            if isinstance(frames, np.ndarray):
                # If it's a single array, check if it's 3D (frames, height, width, channels)
                if len(frames.shape) == 4:
                    frames = [frames[i] for i in range(frames.shape[0])]
                else:
                    frames = [frames]
            elif isinstance(frames, (list, tuple)):
                # Convert each frame to numpy array if it's a PIL Image or tensor
                frame_list = []
                for frame in frames:
                    if isinstance(frame, Image.Image):
                        frame_list.append(np.array(frame))
                    elif isinstance(frame, torch.Tensor):
                        # Convert tensor to numpy
                        if frame.is_cuda:
                            frame = frame.cpu()
                        frame_np = frame.numpy()
                        # Handle different tensor formats
                        if len(frame_np.shape) == 4:
                            frame_np = frame_np[0]  # Remove batch dimension
                        if len(frame_np.shape) == 3:
                            # Convert CHW to HWC if needed
                            if frame_np.shape[0] == 3 or frame_np.shape[0] == 1:
                                frame_np = np.transpose(frame_np, (1, 2, 0))
                            # Normalize if values are in [0, 1] range
                            if frame_np.max() <= 1.0:
                                frame_np = (frame_np * 255).astype(np.uint8)
                            else:
                                frame_np = frame_np.astype(np.uint8)
                        frame_list.append(frame_np)
                    elif isinstance(frame, np.ndarray):
                        frame_list.append(frame)
                    else:
                        # Try to convert to numpy
                        frame_list.append(np.array(frame))
                frames = frame_list
            else:
                # Try to convert to numpy array
                frames = [np.array(frames)]

            # Generate output filename
            if output_filename is None:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = image_path.stem
                output_filename = f"video_{timestamp}_{stem}"

            output_path = VIDEOS_DIR / f"{output_filename}.mp4"

            # Export video - export_to_video expects a list of numpy arrays
            export_to_video(frames, str(output_path), fps=fps)

            if progress_callback:
                progress_callback(1.0)

            return output_path

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {e}") from e

    def clear_cache(self):
        """Clear cached pipelines to free memory."""
        for pipeline in self._pipelines.values():
            del pipeline
        self._pipelines.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

