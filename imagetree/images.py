"""
Objects for loading and saving different image formats.
"""

from typing import Any, Optional, Union

from pathlib import Path

import numpy as np
from astropy.io import fits

from imagetree.tree import QuadTree, TreeConfiguration


class FITSImage:
    metadata: dict[str, Any]
    "Metadata read from the FITS file (header information)."
    tree: QuadTree
    "Quadtree to load tiles from."
    filename: Path
    "Path to original data file. Data from file is not kept in memory."
    hdu: int
    "The HDU that was loaded."
    data_index: Optional[Union[int, slice]]
    "Which data slice to use, if any, from the HDU."

    def __init__(
        self,
        filename: Union[str, Path],
        hdu: int = 0,
        data_index: Optional[Union[int, slice]] = None,
    ):
        """
        Object for loading FITS images from disk.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to file to open.
        hdu : int, optional
            Which HDU to load, by default loads primary (0).
        data_index : int, slice, optional
            Which index to load from the data array, by default loads all.
            Some data provides different challens in the same HDU, for instance
            the ACT maps have I, Q, and U in the same HDU. Image objects should
            be created for each channel.
        """

        if isinstance(filename, str):
            self.filename = Path(filename)
        else:
            self.filename = filename

        self.hdu = hdu

        return

    def load_file(self) -> np.ndarray:
        """
        Loads data from file, including metadata.

        Returns
        -------
        np.ndarray
            Full data array read from disk for use in building tree.
        """

        with fits.open(self.filename) as handle:
            self.metadata = dict(handle[self.hdu].header)

            if self.data_index is None:
                return handle[self.hdu].data
            else:
                return handle[self.hdu].data[self.data_index]

    def build_tree(self, configuration: TreeConfiguration):
        """
        Builds the tree by loading the file.

        Parameters
        ----------
        configuration : TreeConfiguration
            Configuration for the tree.
        """

        tree = QuadTree(configuration=configuration)

        raw_file_data = self.load_file()

        tree.initialize_from_array(data=raw_file_data)
        tree.walk_tree_and_populate()

        return
