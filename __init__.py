"""
WanSoundTrajectory - Audio-driven path modulation for WanMove

A ComfyUI custom node that modulates SplineEditor paths based on audio analysis.
Connect between SplineEditor and WanMove nodes to add audio-reactive camera/object movement.
"""

from .wan_sound_trajectory import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
