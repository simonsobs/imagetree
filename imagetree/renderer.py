"""
Renderer for buffers to images.
"""

from typing import Any, BinaryIO, Optional, Tuple, Union

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm


class Renderer:
    cmap: Union[str, ListedColormap]
    "Color map to use for rendering, defaults to 'viridis', and may not be used if RGBA buffers are provided."
    vmin: Optional[float]
    "Color map range minimum, defaults to None."
    vmax: Optional[float]
    "Color map range maximum, defaults to None."
    log_norm: Optional[bool]
    "Whether to use a log normalization, defaults to False."
    clip: Optional[bool]
    "Whether to clip values outside of the range, defaults to True."
    format: Optional[str]
    "Format to render images to, defaults to 'webp'."
    pil_kwargs: Optional[dict[str, Any]]
    "Keyword arguments to pass to PIL for rendering, defaults to None."

    def __init__(
        self,
        cmap: Union[str, ListedColormap] = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        log_norm: Optional[bool] = False,
        clip: Optional[bool] = True,
        format: Optional[str] = "webp",
        pil_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.log_norm = log_norm
        self.clip = clip
        self.format = format
        self.pil_kwargs = pil_kwargs

        return

    @property
    def norm(self) -> plt.Normalize:
        if self.log_norm:
            return LogNorm(vmin=self.vmin, vmax=self.vmax, clip=self.clip)
        else:
            return plt.Normalize(vmin=self.vmin, vmax=self.vmax, clip=self.clip)

    def render(self, fname: Union[str, Path, BinaryIO], buffer: np.ndarray):
        """
        Renders the buffer to the given file.

        Parameters
        ----------
        fname : Union[str, Path, BinaryIO]
            Output for the rendering.
        buffer : np.ndarray
            Buffer to render to disk or IO.

        Notes
        -----

        Buffer is transposed in x, y to render correctly within this function.
        """

        if buffer.ndim == 2:
            # Render with colour mapping, this is 'raw data'.
            plt.imsave(
                fname,
                self.norm(buffer.T),
                cmap=self.cmap,
                pil_kwargs=self.pil_kwargs,
                format=self.format,
                vmin=0.0,
                vmax=1.0,
            )
        else:
            # Direct rendering
            plt.imsave(
                fname,
                np.ascontiguousarray(buffer.swapaxes(0, 1)),
                pil_kwargs=self.pil_kwargs,
                format=self.format,
            )

        return
