import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import pytest

# add this path in order to import from modules in the root or in src
REPO_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT_DIR))

from src.data import (
    DATASET_PATHS,
    DEFAULT_CACHE_DIR,
    CacheHelper,
    NLBDataHandler,
)

MODULE_UNDER_TEST = "src.data"


class NLBDataHandlerTests(TestCase):
    """Tests for the NLBDataHandler class"""

    def setUp(self):
        with mock.patch(f"{MODULE_UNDER_TEST}.CacheHelper"):
            self.sut = NLBDataHandler()
            self.mock_cache = self.sut._cache_helper

    @mock.patch(f"{MODULE_UNDER_TEST}.NWBDataset")
    def test_load_dataset_opens_dataset_with_correct_path(self, mock_dataset):
        """Verify that the correct path from DATASET_PATHS is used."""
        result = self.sut._load_dataset(name="area2_bump", bin_ms=5)
        expected_path = DATASET_PATHS["area2_bump"]
        self.assertEqual(result, mock_dataset.return_value)
        mock_dataset.assert_called_once_with(expected_path)

    @mock.patch(f"{MODULE_UNDER_TEST}.NWBDataset")
    def test_load_dataset_resamples_to_correct_bin_size(self, mock_dataset):
        """Verify that the dataset is resampled to the requested bin size."""
        bin_size = 17
        result = self.sut._load_dataset(name="area2_bump", bin_ms=bin_size)
        self.assertEqual(result, mock_dataset.return_value)
        mock_result = mock_dataset.return_value
        mock_result.resample.assert_called_once_with(bin_size)

    def test_load_dataset_with_wrong_name(self):
        """Verify that a ValueError is raised when a nonexistant
        dataset is requested.
        """
        with pytest.raises(ValueError):
            _ = self.sut._load_dataset(name="bad_name", bin_ms=5)

    @mock.patch(f"{MODULE_UNDER_TEST}.NWBDataset")
    def test_load_dataset_lru_cache_hit(self, mock_dataset):
        """Verify that the lru_cache is used when the same call is
        made consecutively.
        """
        _ = self.sut._load_dataset(name="mc_maze_medium", bin_ms=5)
        _ = self.sut._load_dataset(name="mc_maze_medium", bin_ms=5)
        mock_dataset.assert_called_once()

    @mock.patch(f"{MODULE_UNDER_TEST}.NWBDataset")
    def test_load_dataset_lru_cache_miss(self, mock_dataset):
        """Verify that the lru_cache is not used when different calls are
        made consecutively.
        """
        _ = self.sut._load_dataset(name="mc_maze_medium", bin_ms=5)
        _ = self.sut._load_dataset(name="mc_maze_medium", bin_ms=3)
        _ = self.sut._load_dataset(name="mc_maze_large", bin_ms=3)
        self.assertEqual(mock_dataset.call_count, 3)

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._get_tensors_for_phase")
    def test_get_train_tensors(self, mock_get_phase):
        """Verify that get_validation_tensors correctly requests train data
        via _get_tensors_for_phase.
        """
        heldin, heldout = ("fake_X", "fake_y")
        mock_get_phase.return_value = (heldin, heldout)
        result = self.sut.get_train_tensors(
            "foo", phase="bar", bin_ms=7, use_cache=False
        )
        self.assertEqual(result, (heldin, heldout))
        mock_get_phase.assert_called_once_with(
            phase="bar",
            tensor_set="train",
            dataset_name="foo",
            bin_ms=7,
            use_cache=False,
        )

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._get_tensors_for_phase")
    def test_get_validation_tensors(self, mock_get_phase):
        """Verify that get_validation_tensors correctly requests validation data
        via _get_tensors_for_phase.
        """
        heldin, heldout = ("fake_X", "fake_y")
        mock_get_phase.return_value = (heldin, heldout)
        result = self.sut.get_eval_tensors(
            "bar", phase="bar", bin_ms=11, use_cache=False
        )
        self.assertEqual(result, (heldin, heldout))
        mock_get_phase.assert_called_once_with(
            phase="bar",
            tensor_set="eval",
            dataset_name="bar",
            bin_ms=11,
            use_cache=False,
        )

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._get_forward_tensors")
    def test_get_train_forward_tensors(self, mock_get_forward):
        """Verify that get_train_forward_tensors correctly requests train data
        via _get_forward_tensors
        """
        heldin, heldout = ("fake_X", "fake_y")
        mock_get_forward.return_value = (heldin, heldout)
        result = self.sut.get_train_forward_tensors(
            "foo", phase="bar", bin_ms=11, use_cache=False
        )
        self.assertEqual(result, (heldin, heldout))
        mock_get_forward.assert_called_once_with(
            dataset_name="foo",
            phase="bar",
            tensor_set="train",
            bin_ms=11,
            use_cache=False,
        )

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._get_forward_tensors")
    def test_get_eval_forward_tensors(self, mock_get_forward):
        """Verify that get_eval_forward_tensors correctly requests eval data
        via _get_forward_tensors
        """
        heldin, heldout = ("fake_X", "fake_y")
        mock_get_forward.return_value = (heldin, heldout)
        result = self.sut.get_eval_forward_tensors(
            "foo", phase="bar", bin_ms=11, use_cache=False
        )
        self.assertEqual(result, (heldin, heldout))
        mock_get_forward.assert_called_once_with(
            dataset_name="foo",
            phase="bar",
            tensor_set="eval",
            bin_ms=11,
            use_cache=False,
        )

    def test_get_tensors_for_phase_with_cache_hit(self):
        """Verify that if the cache is hit, the tensors are returned from the cache."""
        self.mock_cache.cached_files_exist.return_value = True
        result = self.sut._get_tensors_for_phase(
            phase="a_phase",
            tensor_set="a_set",
            dataset_name="foo",
            bin_ms=7,
            use_cache=True,
        )
        self.assertEqual(result, self.mock_cache.load_tensors_from_cache.return_value)

    def test_get_forward_tensors_with_cache_hit(self):
        """Verify that if the cache is hit, the tensors are returned from the cache."""
        self.mock_cache.cache_files_exist.return_value = True
        result = self.sut._get_forward_tensors(
            dataset_name="foo",
            phase="a_phase",
            tensor_set="a_set",
            bin_ms=7,
            use_cache=True,
        )
        self.assertEqual(result, self.mock_cache.load_tensors_from_cache.return_value)

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._build_tensor_for_phase")
    def test_get_tensors_for_phase_without_cache(self, mock_build):
        """Verify that if use_cache is False, the cache is ignored."""
        mock_build.return_value = ("a mock", "two-tuple")
        result = self.sut._get_tensors_for_phase(
            phase="a_phase",
            tensor_set="a_set",
            dataset_name="foo",
            bin_ms=7,
            use_cache=False,
        )
        self.mock_cache.cached_files_exist.assert_not_called()
        self.mock_cache.load_tensors_from_cache.assert_not_called()
        self.mock_cache.cache_arrays.assert_not_called()
        self.assertEqual(result, mock_build.return_value)

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._build_forward_tensors")
    def test_get_forward_tensors_without_cache(self, mock_build):
        """Verify that if use_cache is False, the cache is ignored."""
        mock_build.return_value = ("a mock", "two-tuple")
        result = self.sut._get_forward_tensors(
            phase="a_phase",
            tensor_set="a_set",
            dataset_name="foo",
            bin_ms=7,
            use_cache=False,
        )
        self.mock_cache.cached_files_exist.assert_not_called()
        self.mock_cache.load_tensors_from_cache.assert_not_called()
        self.mock_cache.cache_arrays.assert_not_called()
        self.assertEqual(result, mock_build.return_value)

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._build_tensor_for_phase")
    def test_get_tensors_for_phase_with_cache_miss(self, mock_build):
        """Verify that if use_cache is True but the cache is missed, the loader will
        cache and return the tensors.
        """
        heldin, heldout = ("a mock", "two-tuple")
        mock_build.return_value = (heldin, heldout)
        self.mock_cache.cached_files_exist.return_value = False
        result = self.sut._get_tensors_for_phase(
            phase="a_phase",
            tensor_set="a_set",
            dataset_name="foo",
            bin_ms=7,
            use_cache=True,
        )
        self.mock_cache.load_tensors_from_cache.assert_not_called()
        self.mock_cache.cache_arrays.assert_called_once_with(
            "foo",
            bin_ms=7,
            phase="a_phase",
            tensor_set="a_set",
            heldin=heldin,
            heldout=heldout,
        )
        self.assertEqual(result, (heldin, heldout))

    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._build_forward_tensors")
    def test_get_forward_tensors_with_cache_miss(self, mock_build):
        """Verify that if use_cache is True but the cache is missed, the loader will
        cache and return the tensors.
        """
        heldin, heldout = ("a mock", "two-tuple")
        mock_build.return_value = (heldin, heldout)
        self.mock_cache.cached_files_exist.return_value = False
        result = self.sut._get_forward_tensors(
            phase="a_phase",
            tensor_set="a_set",
            dataset_name="foo",
            bin_ms=7,
            use_cache=True,
        )
        self.mock_cache.load_tensors_from_cache.assert_not_called()
        self.mock_cache.cache_arrays.assert_called_once_with(
            "foo",
            bin_ms=7,
            phase="a_phase",
            tensor_set="a_set",
            heldin=heldin,
            heldout=heldout,
            forward=True,
        )
        self.assertEqual(result, (heldin, heldout))

    @mock.patch(f"{MODULE_UNDER_TEST}.make_train_input_tensors")
    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._load_dataset")
    def test_build_tensor_for_phase_for_train(self, mock_loader, mock_train_tensors):
        """Verify that the train tensors are loaded and extracted from the nlb
        train_dict properly
        """
        heldin_fake, heldout_fake = get_fake_arrays()
        fake_train_dict = {
            "train_spikes_heldin": heldin_fake,
            "train_spikes_heldout": heldout_fake,
        }
        mock_train_tensors.return_value = fake_train_dict
        heldin, heldout = self.sut._build_tensor_for_phase(
            phase="val", tensor_set="train", dataset_name="foo", bin_ms=7
        )
        mock_loader.assert_called_once_with(name="foo", bin_ms=7)
        mock_train_tensors.assert_called_once_with(
            mock_loader.return_value,
            dataset_name="foo",
            trial_split="train",
            save_file=False,
        )
        assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
        assert_array_equal_but_with_upgraded_dtype(heldout, heldout_fake)

    @mock.patch(f"{MODULE_UNDER_TEST}.make_eval_input_tensors")
    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._load_dataset")
    def test_build_tensor_for_phase_for_validation(
        self, mock_loader, mock_eval_tensors
    ):
        """Verify that the train tensors are loaded and extracted from the nlb
        val_dict properly
        """
        heldin_fake, heldout_fake = get_fake_arrays()
        fake_eval_dict = {
            "eval_spikes_heldin": heldin_fake,
            "eval_spikes_heldout": heldout_fake,
        }
        mock_eval_tensors.return_value = fake_eval_dict
        heldin, heldout = self.sut._build_tensor_for_phase(
            phase="val", tensor_set="eval", dataset_name="foo", bin_ms=7
        )
        mock_loader.assert_called_once_with(name="foo", bin_ms=7)
        mock_eval_tensors.assert_called_once_with(
            mock_loader.return_value,
            dataset_name="foo",
            trial_split="val",
            save_file=False,
        )
        assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
        assert_array_equal_but_with_upgraded_dtype(heldout, heldout_fake)

    def test_build_tensor_for_phase_for_unknown_phase(self):
        """Verify that the _build_tensor_for_phase raises a ValueError for an unknown
        phase
        """
        with pytest.raises(ValueError):
            self.sut._build_tensor_for_phase(
                phase="unknown", tensor_set="train", dataset_name="foo", bin_ms=7
            )

    @mock.patch(f"{MODULE_UNDER_TEST}.make_train_input_tensors")
    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._load_dataset")
    def test_build_forward_tensors_for_train(self, mock_loader, mock_train_tensors):
        """Verify that the forward train tensors are loaded and extracted from the nlb
        train_dict properly
        """
        heldin_fake, heldout_fake = get_fake_arrays()
        fake_train_dict = {
            "train_spikes_heldin_forward": heldin_fake,
            "train_spikes_heldout_forward": heldout_fake,
        }
        mock_train_tensors.return_value = fake_train_dict
        heldin, heldout = self.sut._build_forward_tensors(
            phase="val", tensor_set="train", dataset_name="foo", bin_ms=7
        )
        mock_loader.assert_called_once_with(name="foo", bin_ms=7)
        mock_train_tensors.assert_called_once_with(
            mock_loader.return_value,
            dataset_name="foo",
            trial_split="train",
            save_file=False,
            include_forward_pred=True,
        )
        assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
        assert_array_equal_but_with_upgraded_dtype(heldout, heldout_fake)

    @mock.patch(f"{MODULE_UNDER_TEST}.make_train_input_tensors")
    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._load_dataset")
    def test_build_forward_tensors_for_train_test_phase(
        self, mock_loader, mock_train_tensors
    ):
        """Verify that the forward train tensors are loaded and extracted from the nlb
        train_dict properly
        """
        heldin_fake, heldout_fake = get_fake_arrays()
        fake_train_dict = {
            "train_spikes_heldin_forward": heldin_fake,
            "train_spikes_heldout_forward": heldout_fake,
        }
        mock_train_tensors.return_value = fake_train_dict
        heldin, heldout = self.sut._build_forward_tensors(
            phase="test", tensor_set="train", dataset_name="foo", bin_ms=7
        )
        mock_loader.assert_called_once_with(name="foo", bin_ms=7)
        mock_train_tensors.assert_called_once_with(
            mock_loader.return_value,
            dataset_name="foo",
            trial_split=["train", "val"],
            save_file=False,
            include_forward_pred=True,
        )
        assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
        assert_array_equal_but_with_upgraded_dtype(heldout, heldout_fake)

    @mock.patch(f"{MODULE_UNDER_TEST}.make_eval_target_tensors")
    @mock.patch(f"{MODULE_UNDER_TEST}.NLBDataHandler._load_dataset")
    def test_build_forward_tensors_for_validation(self, mock_loader, mock_eval_tensors):
        """Verify that the train tensors are loaded and extracted from the nlb
        val_dict properly
        """
        heldin_fake, heldout_fake = get_fake_arrays()

        fake_eval_dict = {
            "foo_7": {
                "eval_spikes_heldin_forward": heldin_fake,
                "eval_spikes_heldout_forward": heldout_fake,
            }
        }
        mock_loader.return_value = SimpleNamespace(bin_width=7)
        mock_eval_tensors.return_value = fake_eval_dict
        heldin, heldout = self.sut._build_forward_tensors(
            phase="val", tensor_set="eval", dataset_name="foo", bin_ms=7
        )

        mock_loader.assert_called_once_with(name="foo", bin_ms=7)
        mock_eval_tensors.assert_called_once_with(
            mock_loader.return_value,
            dataset_name="foo",
            save_file=False,
        )
        assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
        assert_array_equal_but_with_upgraded_dtype(heldout, heldout_fake)

    def test_build_forward_tensors_for_unknown_phase(self):
        """Verify that the _build_tensor_for_phase raises a ValueError for an unknown
        phase
        """
        with pytest.raises(ValueError):
            self.sut._build_forward_tensors(
                phase="unknown", tensor_set="train", dataset_name="foo", bin_ms=7
            )


def get_fake_arrays():
    """Returns fake heldin and heldout numpy arrays which are float16"""
    heldin_fake = np.array([1.0], dtype=np.float16)
    heldout_fake = np.array([2.0], dtype=np.float16)
    return heldin_fake, heldout_fake


def assert_array_equal_but_with_upgraded_dtype(arr, arr_orig):
    """Assert that the two arrays are equal and that `arr` has a dtype
    of float (=np.float64)
    """
    np.testing.assert_array_equal(arr, arr_orig)
    assert arr.dtype == float


class CacheHelperTests(TestCase):
    """Tests for the CacheHelper class"""

    def setUp(self):
        with mock.patch("pathlib.Path.mkdir"):
            self.sut = CacheHelper()

    @mock.patch("pathlib.Path.mkdir")
    def test_default_cache_dir(self, mock_mkdir):
        """Verify that the default cache path is used when no
        constructor arguments are given.
        """
        sut = CacheHelper()
        expected = DEFAULT_CACHE_DIR
        self.assertEqual(sut.cache_path, expected)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @mock.patch("pathlib.Path.mkdir")
    def test_custom_cache_dir(self, mock_mkdir):
        """Verify that a custom path is correctly set when
        passed as a constructor argument.
        """
        test_path = Path("/a/b/c")
        sut = CacheHelper(cache_path=test_path)
        self.assertEqual(sut.cache_path, test_path)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_get_cache_filenames(self):
        """Verify that the cache filenames are correct"""
        test_cases = [
            # all combinations of phase and tensor_set
            ("val", "train", True),
            ("val", "eval", True),
            ("test", "train", True),
            ("test", "eval", True),
            ("val", "train", False),
            ("val", "eval", False),
            ("test", "train", False),
            ("test", "eval", False),
        ]
        for phase, tensor_set, forward in test_cases:
            with mock.patch.object(self.sut, "cache_path", Path("/a/b/c")):
                result = self.sut._get_cache_filepaths(
                    "a_dataset",
                    bin_ms=7,
                    phase=phase,
                    tensor_set=tensor_set,
                    forward=forward,
                )
            suffix = "_forward" if forward else ""
            expected = [
                Path(f"/a/b/c/a_dataset_7ms_{phase}_{tensor_set}_heldin{suffix}.npy"),
                Path(f"/a/b/c/a_dataset_7ms_{phase}_{tensor_set}_heldout{suffix}.npy"),
            ]
            if phase == "test" and tensor_set == "eval":
                expected = expected[:1]
            self.assertListEqual(result, expected)

    def test_cached_files_exist_checks_all_paths(self):
        """Verify that cached_files_exist uses the correct boolean logic on all paths

        Uses two mock paths where the return value for `Path.exists` is set
        according to the test case, and then validates that the return value
        of the target function is as expected from the test case.
        """
        test_cases = [
            # we expect AND logic. return value is True iff both paths exist
            ((True, True), True),
            ((True, False), False),
            ((False, True), False),
            ((False, False), False),
        ]
        for files_exist, expected in test_cases:
            with self._prepare_mock_get_cache_filepaths(
                files_exist
            ) as mock_filenames_fn:
                result = self.sut.cached_files_exist(
                    "foo", bin_ms=7, phase="a_phase", tensor_set="a_set", forward=False
                )
                assert result == expected
                mock_filenames_fn.assert_called_once_with(
                    "foo", bin_ms=7, phase="a_phase", tensor_set="a_set", forward=False
                )

    def _prepare_mock_get_cache_filepaths(self, files_exist=None):
        mock_paths = [mock.Mock(Path), mock.Mock(Path)]
        if files_exist:
            p1_exists, p2_exists = files_exist
            mock_paths[0].exists.return_value = p1_exists
            mock_paths[1].exists.return_value = p2_exists
        return mock.patch.object(
            self.sut, "_get_cache_filepaths", return_value=mock_paths
        )

    @mock.patch(f"{MODULE_UNDER_TEST}.np.load")
    def test_load_tensors_from_cache_uses_correct_paths(self, mock_np_load):
        """Verifies that load_tensors_from_cache uses the correct paths"""
        with self._prepare_mock_get_cache_filepaths() as mock_filenames_fn:
            expected_paths = mock_filenames_fn.return_value
            heldin, heldout = self.sut.load_tensors_from_cache(
                "foo",
                bin_ms=7,
                phase="a_phase",
                tensor_set="a_set",
            )
            mock_filenames_fn.assert_called_once_with(
                "foo", bin_ms=7, phase="a_phase", tensor_set="a_set", forward=False
            )
            mock_np_load.assert_has_calls(
                [mock.call(expected_paths[0]), mock.call(expected_paths[1])],
                any_order=True,
            )

    @mock.patch(f"{MODULE_UNDER_TEST}.np.load")
    def test_load_tensors_from_cache_formats_arrays_correctly(self, mock_np_load):
        """Verifies that load_tensors_from_cache sets the array type correctly"""
        heldin_fake, heldout_fake = get_fake_arrays()
        mock_np_load.side_effect = (heldin_fake, heldout_fake)
        with self._prepare_mock_get_cache_filepaths():
            heldin, heldout = self.sut.load_tensors_from_cache(
                "foo", bin_ms=7, phase="a_phase", tensor_set="a_set"
            )
            assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
            assert_array_equal_but_with_upgraded_dtype(heldout, heldout_fake)

    @mock.patch(f"{MODULE_UNDER_TEST}.np.load")
    def test_load_tensors_from_cache_loads_none_for_test_holdout(self, mock_np_load):
        """Verifies that load_tensors_from_cache handles holdout tensors correctly"""
        heldin_fake, _ = get_fake_arrays()
        mock_np_load.side_effect = heldin_fake

        with self._prepare_mock_get_cache_filepaths():
            heldin, heldout = self.sut.load_tensors_from_cache(
                "foo", bin_ms=7, phase="test", tensor_set="eval"
            )
            assert_array_equal_but_with_upgraded_dtype(heldin, heldin_fake)
            assert heldout is None

    @mock.patch(f"{MODULE_UNDER_TEST}.np.save")
    def test_cache_arrays(self, mock_save):
        """Verify that cache_arrays saves each array as expected"""
        heldin_fake, heldout_fake = "heldin_fake", "heldout_fake"
        with self._prepare_mock_get_cache_filepaths() as mock_filenames_fn:
            expected_paths = mock_filenames_fn.return_value
            self.sut.cache_arrays(
                "foo",
                bin_ms=7,
                phase="a_phase",
                tensor_set="a_set",
                heldin=heldin_fake,
                heldout=heldout_fake,
            )
            mock_filenames_fn.assert_called_once_with(
                "foo", bin_ms=7, phase="a_phase", tensor_set="a_set", forward=False
            )
        mock_save.assert_has_calls(
            [
                mock.call(expected_paths[0], heldin_fake),
                mock.call(expected_paths[1], heldout_fake),
            ]
        )

    @mock.patch(f"{MODULE_UNDER_TEST}.np.save")
    def test_cache_arrays_does_not_save_test_set_holdout(self, mock_save):
        """Verify that cache_arrays saves each array as expected"""
        heldin_fake, heldout_fake = "heldin_fake", None
        with self._prepare_mock_get_cache_filepaths() as mock_filenames_fn:
            expected_paths = mock_filenames_fn.return_value
            self.sut.cache_arrays(
                "foo",
                bin_ms=7,
                phase="test",
                tensor_set="eval",
                heldin=heldin_fake,
                heldout=heldout_fake,
            )
            mock_filenames_fn.assert_called_once_with(
                "foo", bin_ms=7, phase="test", tensor_set="eval", forward=False
            )

        mock_save.assert_called_with(expected_paths[0], heldin_fake)
        with pytest.raises(AssertionError):
            mock_save.assert_called_with(expected_paths[1], heldout_fake)
