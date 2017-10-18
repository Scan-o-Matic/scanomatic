import numpy as np
from scanomatic.image_analysis import signal


class TestGetSignalEdges:

    def test_works_with_useful_data(self):
        observed_to_expected_index_map = np.array([
            0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8,
            9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16,
            16, 16, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20,
            20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22,
            22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23
        ])

        deltas = np.array([
            0.05, 1.05, 2.05, 3.05, 4.05, 5.05, 6.05, 2.75, 2.45, 3.45, 3.15,
            2.85, 3.85, 3.55, 4.55, 3.25, 4.25, 3.95, 4.95, 3.65, 4.65, 5.65,
            4.35, 5.35, 5.05, 6.05, 4.75, 5.75, 6.75, 5.45, 6.45, 6.15, 7.15,
            5.85, 6.85, 7.85, 6.55, 7.55, 6.25, 7.25, 8.25, 6.05, 6.95, 7.95,
            np.nan, 7.65, 8.65, np.nan, 7.35, 8.35, np.nan, 1.05, 2.05, 3.05,
            7.05, 8.05, np.nan, np.nan, 4.25, 2.25, 1.25, 0.25, 0.75, 4.75,
            6.75, 7.75, 8.75, np.nan, np.nan, np.nan, np.nan, 7.55, 7.45, 8.45,
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            3.85,
        ])

        observed_spikes = np.array([
            0, 1, 2, 3, 4, 5, 6, 32, 61, 62, 91, 120, 121, 150, 151, 179, 180,
            209, 210, 238, 239, 240, 268, 269, 298, 299, 327, 328, 329, 357,
            358, 387, 388, 416, 417, 418, 446, 447, 475, 476, 477, 492, 505,
            506, 507, 535, 536, 537, 564, 565, 566, 587, 588, 589, 593, 594,
            595, 605, 611, 613, 614, 615, 616, 620, 622, 623, 624, 626, 630,
            631, 635, 637, 652, 653, 654, 655, 657, 658, 660, 662, 663, 664,
            670
        ])

        number_of_segments = 23

        edges = signal.get_signal_edges(
            observed_to_expected_index_map,
            deltas,
            observed_spikes,
            number_of_segments)

        assert len(edges) == 24
        assert np.isfinite(edges).sum() == 17

    def test_no_signal_returns_none(self):

        edges = signal.get_signal_edges(
            np.array([]),
            np.array([]),
            np.array([]),
            23)

        assert len(edges) == 24
        assert not np.isfinite(edges).any()

    def test_no_finite_deltas(self):

        observed_to_expected_index_map = np.array([
            0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8,
            9, 9, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16,
            16, 16, 17, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20,
            20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22,
            22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23
        ])

        deltas = np.ones(observed_to_expected_index_map.shape) * np.nan

        observed_spikes = np.array([
            0, 1, 2, 3, 4, 5, 6, 32, 61, 62, 91, 120, 121, 150, 151, 179, 180,
            209, 210, 238, 239, 240, 268, 269, 298, 299, 327, 328, 329, 357,
            358, 387, 388, 416, 417, 418, 446, 447, 475, 476, 477, 492, 505,
            506, 507, 535, 536, 537, 564, 565, 566, 587, 588, 589, 593, 594,
            595, 605, 611, 613, 614, 615, 616, 620, 622, 623, 624, 626, 630,
            631, 635, 637, 652, 653, 654, 655, 657, 658, 660, 662, 663, 664,
            670
        ])

        number_of_segments = 23

        edges = signal.get_signal_edges(
            observed_to_expected_index_map,
            deltas,
            observed_spikes,
            number_of_segments)

        assert len(edges) == 24
        assert not np.isfinite(edges).any()
