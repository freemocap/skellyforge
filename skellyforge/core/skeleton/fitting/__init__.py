"""Connected sequence fitting backed by SkellyForge's native Ceres extension."""
from .window_sequence import WindowSequenceFit, fit_windows, refine_window_result

__all__ = ["WindowSequenceFit", "fit_windows", "refine_window_result"]
