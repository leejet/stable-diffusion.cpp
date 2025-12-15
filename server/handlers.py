"""
Image generation and editing handlers
Combines ImageGenerator and ImageEditor classes
"""

import asyncio
from image_generator import ImageGenerator
from image_editor import ImageEditor
from models import SD_BINARY, OUTPUT_DIR

# Global GPU semaphore (shared across all handlers)
gpu_semaphore = asyncio.Semaphore(1)

# Initialize generator
image_generator = ImageGenerator(SD_BINARY, OUTPUT_DIR, gpu_semaphore)

# Initialize editor
image_editor = ImageEditor(SD_BINARY, OUTPUT_DIR, gpu_semaphore)

# Export for app.py
__all__ = ["image_generator", "image_editor", "gpu_semaphore"]
