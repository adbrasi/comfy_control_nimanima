"""
ComfyUI-EasyControl-Anima
Custom nodes for EasyControl spatial conditioning on the Anima model.

Nodes:
  - LoadEasyControl: Load an EasyControl LoRA adapter
  - ApplyEasyControlCondition: Apply a condition image with an EasyControl adapter to a model
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
