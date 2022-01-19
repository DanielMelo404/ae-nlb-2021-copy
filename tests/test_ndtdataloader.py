import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# add this path in order to import from modules in the root or in src
REPO_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT_DIR))

from src import data
from src.data import NDTDataHandler, NLBDataHandler

MODULE_UNDER_TEST = "src.data"


# Test Config
# ===========
def fake_tensors(*args, **kwargs):
    """Create fake tensors for testing"""
    phase = kwargs.get("phase")
    if phase == "val":
        return np.array([0, 1]), np.array([2, 3])
    elif phase == "test":
        return np.array([0, 1, 2]), np.array([3, 4, 5])


def fake_path():
    """Create a fake path"""
    return Path("/a/b/c")


@pytest.fixture(autouse=True)
def mock_get_tensors(monkeypatch):
    """Replace NLBDataHandler functions with fake tensors"""
    monkeypatch.setattr(NLBDataHandler, "get_train_tensors", fake_tensors)
    monkeypatch.setattr(NLBDataHandler, "get_train_forward_tensors", fake_tensors)
    monkeypatch.setattr(NLBDataHandler, "get_eval_tensors", fake_tensors)
    monkeypatch.setattr(NLBDataHandler, "get_eval_forward_tensors", fake_tensors)


@pytest.fixture(autouse=True)
def mock_data_dir(monkeypatch):
    """Replace the data directory with a fake path"""
    monkeypatch.setattr(data, "DEFAULT_NDT_DATA_DIR", fake_path())


@pytest.fixture(autouse=True)
def mock_mkdir(monkeypatch):
    """Replace the call to mkdirs in Pathlib"""

    def do_nothing(*args, **kwargs):
        pass

    monkeypatch.setattr(Path, "mkdir", do_nothing)


@pytest.fixture
def mock_train_data():
    """Create fake training data dictionary"""
    heldin, heldout = fake_tensors(phase="val")
    return dict(
        train_data_heldin=heldin,
        train_data_heldout=heldout,
        train_data_heldin_forward=heldin,
        train_data_heldout_forward=heldout,
        eval_data_heldin=heldin,
        eval_data_heldout=heldout,
        eval_data_heldin_forward=heldin,
        eval_data_heldout_forward=heldout,
    )


@pytest.fixture
def mock_test_data():
    """Create fake test data dictionary"""
    heldin, heldout = fake_tensors(phase="test")
    return dict(
        train_data_heldin=heldin,
        train_data_heldout=heldout,
        train_data_heldin_forward=heldin,
        train_data_heldout_forward=heldout,
        eval_data_heldin=heldin,
    )


# Test Public API
# ===============
def test_make_train_data(monkeypatch, mock_train_data):
    """Verify make_train_data passes correct tensors and filename for saving"""
    sut = NDTDataHandler()
    expected_data = mock_train_data
    expected_filename = "/a/b/c/foo_train.h5"

    def fake_save(data, filename, *args, **kwargs):
        np.testing.assert_equal(data, expected_data)
        assert filename == expected_filename

    monkeypatch.setattr(data, "save_to_h5", fake_save)
    sut.make_train_data("foo")


def test_make_test_data(monkeypatch, mock_test_data):
    """Verify make_test_data passes correct tensors and filename for saving"""
    sut = NDTDataHandler()
    expected_data = mock_test_data
    expected_filename = "/a/b/c/foo_test.h5"

    def fake_save(data, filename, *args, **kwargs):
        np.testing.assert_equal(data, expected_data)
        assert filename == expected_filename

    monkeypatch.setattr(data, "save_to_h5", fake_save)
    sut.make_test_data("foo")


def test_get_filename():
    """Verify the saved filename is correct"""
    sut = NDTDataHandler()
    test_cases = [("val", "train"), ("test", "test")]

    for phase, name in test_cases:
        result = sut.get_filename(dataset_name="foo", phase=phase)
        expected = f"/a/b/c/foo_{name}.h5"
        assert result == expected


# Test Private API
# ================
def test_build_data_dict(mock_train_data, mock_test_data):
    """Test that the data dictionary is created correctly"""
    test_cases = ["val", "test"]
    sut = NDTDataHandler()
    for case in test_cases:
        result = sut._build_data_dict("foo", phase=case)
        if case == "val":
            expected = mock_train_data
        if case == "test":
            expected = mock_test_data

        np.testing.assert_equal(result, expected)


@mock.patch(f"{MODULE_UNDER_TEST}.NDTDataHandler._file_exists")
@mock.patch(f"{MODULE_UNDER_TEST}.save_to_h5")
def test_make_data_does_not_overwrite(mock_save, mock_exists):
    """Verify that files are note overwritten"""
    sut = NDTDataHandler()
    mock_exists.return_value = True
    sut.make_train_data("foo", overwrite=False)
    mock_exists.return_value = True
    sut.make_test_data("foo", overwrite=False)
    mock_save.assert_not_called()


@mock.patch(f"{MODULE_UNDER_TEST}.NDTDataHandler._file_exists")
@mock.patch(f"{MODULE_UNDER_TEST}.save_to_h5")
def test_make_data_does_overwrite_h5(mock_save, mock_exists):
    """Verify that files are overwritten when specified"""
    sut = NDTDataHandler()
    mock_exists.return_value = True
    sut.make_train_data("foo", overwrite=True)
    sut.make_test_data("foo", overwrite=True)
    assert mock_save.call_count == 2
