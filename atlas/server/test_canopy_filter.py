"""Unit tests for the atlas-only canopy display filter."""

from __future__ import annotations

import unittest

import numpy as np

from server.app import CANOPY_NODATA, suppress_small_canopy_components


class CanopyDisplayFilterTests(unittest.TestCase):
    def test_removes_components_smaller_than_six_pixels(self) -> None:
        values = np.zeros((8, 8), dtype=np.uint8)
        values[2, 2:5] = 255
        values[3, 2:4] = 255

        cleaned = suppress_small_canopy_components(values)

        self.assertEqual(int((cleaned == 255).sum()), 0)

    def test_retains_six_diagonally_connected_pixels(self) -> None:
        values = np.zeros((8, 8), dtype=np.uint8)
        for index in range(1, 7):
            values[index, index] = 255

        cleaned = suppress_small_canopy_components(values)

        self.assertEqual(int((cleaned == 255).sum()), 6)

    def test_preserves_nodata_and_does_not_fill_small_holes(self) -> None:
        values = np.full((7, 7), 255, dtype=np.uint8)
        values[0, :] = CANOPY_NODATA
        values[3, 3] = 0

        cleaned = suppress_small_canopy_components(values)

        self.assertTrue(np.all(cleaned[0, :] == CANOPY_NODATA))
        self.assertEqual(int(cleaned[3, 3]), 0)


if __name__ == "__main__":
    unittest.main()
