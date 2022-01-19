import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from nlb_tools.make_tensors import (
    make_eval_input_tensors,
    make_eval_target_tensors,
    make_train_input_tensors,
    save_to_h5,
)
from nlb_tools.nwb_interface import NWBDataset

from src.config.settings import settings

logger = logging.getLogger(__name__)

RAW_DATA_DIR = settings.NLB_DATA_RAW

DATASET_PATHS = {
    "mc_maze": RAW_DATA_DIR / "000128" / "sub-Jenkins",
    "mc_rtt": RAW_DATA_DIR / "000129" / "sub-Indy",
    "area2_bump": RAW_DATA_DIR / "000127" / "sub-Han",
    "dmfc_rsg": RAW_DATA_DIR / "000130" / "sub-Haydn",
    "mc_maze_large": RAW_DATA_DIR / "000138" / "sub-Jenkins",
    "mc_maze_medium": RAW_DATA_DIR / "000139" / "sub-Jenkins",
    "mc_maze_small": RAW_DATA_DIR / "000140" / "sub-Jenkins",
}

DEFAULT_CACHE_DIR = settings.NLB_CACHE_DIR
DEFAULT_NDT_DATA_DIR = settings.NDT_CACHE_DIR

TwoArrayTuple = Tuple[np.ndarray, Optional[np.ndarray]]


class NLBDataHandler:
    """Loads raw NLB data and converts into train and val tensors.

    Simplifies the nlb_tools API, and optionally allows for caching
    pre-built tensors to disk.

    Use NLBDataHandler.get_train_tensors and NLBDataHandler.get_eval_tensors
    to get train and eval tensors as numpy arrays.  In each of these methods
    set the phase="val" (default) to get training and validation data and
    set the phase="test" to get test data.

    Note the NLB challenge uses the following terminology:
        Tensor set: ['train', 'eval']
        Phase: ['val', 'test']
        Split: ['train','val','test']

    NLBDatahandler.get_train_tensors(phase='val') will return the train tensor
    set from the `val` phase (the standard training data)

    NLBDatahandler.get_train_tensors(phase='test') will return the train
    tensor set for the 'test' phase (used for metrics other than co-bps)

    Similar for `get_eval_tensors`.
    """

    def __init__(self, *, cache_path: Path = None):
        """Instantiate a NLBDataHandler.

        Args:
            cache_path (Path, optional): A path to the directory where files
                should be cached. Defaults to None.
        """
        self._cache_helper = CacheHelper(cache_path=cache_path)

    def get_train_tensors(
        self,
        dataset_name: str,
        *,
        phase: str = "val",
        bin_ms: int = 5,
        use_cache: bool = True,
    ) -> TwoArrayTuple:
        """Returns heldin and heldout training arrays.

        Args:
            dataset_name (str): The name of the NLB dataset to return.
            phase (str): The phase, one of ["val", "test"]
            bin_ms (int, optional): The desired bin width in ms. Defaults to 5.
            use_cache (bool, optional): Whether or not to use the cache. If
                False, the data will be loaded from raw NLB data. If true and
                cache files do not exist, the arrays will be loaded from the
                raw NLB data but then cached to file before returning.
                Defaults to True.

        Returns:
            (np.ndarray, np.ndarray): heldin and heldout training arrays
        """
        return self._get_tensors_for_phase(
            tensor_set="train",
            phase=phase,
            dataset_name=dataset_name,
            bin_ms=bin_ms,
            use_cache=use_cache,
        )

    def get_eval_tensors(
        self,
        dataset_name: str,
        *,
        phase: str = "val",
        bin_ms: int = 5,
        use_cache: bool = True,
    ) -> TwoArrayTuple:
        """Returns heldin and heldout validation arrays.

        Note: when phase="test" there is no heldout evaluation set
        so the return value will have type (np.ndarray, None).

        Args:
            dataset_name (str): The name of the NLB dataset to return.
            phase (str): The desired phase. One of ["val", "test"]
            bin_ms (int, optional): The desired bin width in ms. Defaults to 5.
            use_cache (bool, optional): Whether or not to use the cache. If
                False, the data will be loaded from raw NLB data. If true and
                cache files do not exist, the arrays will be loaded from the
                raw NLB data but then cached to file before returning.
                Defaults to True.

        Returns:
            (np.ndarray, Optional[np.ndarray]): heldin and heldout validation arrays
        """
        return self._get_tensors_for_phase(
            tensor_set="eval",
            phase=phase,
            dataset_name=dataset_name,
            bin_ms=bin_ms,
            use_cache=use_cache,
        )

    def get_train_forward_tensors(
        self,
        dataset_name,
        *,
        phase: str = "val",
        bin_ms: int = 5,
        use_cache: bool = True,
    ) -> TwoArrayTuple:
        """Returns heldin and heldout forward train tensors

        Args:
            dataset_name (str): The name of the NLB dataset to return
            phase (str): The desired phase. One of ["val", "test"]
            bin_ms (int, optional): The desired bin width in ms. Defaults to 5.
            use_cache (bool, optional): Whether or not to use the cache. If
                False, the data will be loaded from raw NLB data. If true and
                cache files do not exist, the arrays will be loaded from the
                raw NLB data but then cached to file before returning.
                Defaults to True.

        Returns:
            (np.ndarray, np.ndarray): the heldin and heldout forward tensors

        """
        return self._get_forward_tensors(
            dataset_name=dataset_name,
            phase=phase,
            tensor_set="train",
            bin_ms=bin_ms,
            use_cache=use_cache,
        )

    def get_eval_forward_tensors(
        self,
        dataset_name,
        *,
        phase: str = "val",
        bin_ms: int = 5,
        use_cache: bool = True,
    ) -> TwoArrayTuple:
        """Returns heldin and heldout forward eval tensors

        Note:
            `phase` has no effect on this function and is only an option
            to provide a consistent interface.  The "eval" forward tensors
            are always the same - they are computed via
            `nlb_tools.make_eval_target_tensors`.

        Args:
            dataset_name (str): The name of the NLB dataset to return
            phase (str): The desired phase. Ignored.
            bin_ms (int, optional): The desired bin width in ms. Defaults to 5.
            use_cache (bool, optional): Whether or not to use the cache. If
                False, the data will be loaded from raw NLB data. If true and
                cache files do not exist, the arrays will be loaded from the
                raw NLB data but then cached to file before returning.
                Defaults to True.

        Returns:
            (np.ndarray, np.ndarray): the heldin and heldout forward tensors

        """
        return self._get_forward_tensors(
            dataset_name=dataset_name,
            phase=phase,
            tensor_set="eval",
            bin_ms=bin_ms,
            use_cache=use_cache,
        )

    def _get_forward_tensors(
        self,
        *,
        dataset_name: str,
        phase: str,
        tensor_set: str,
        bin_ms: int,
        use_cache: bool,
    ) -> TwoArrayTuple:
        """Returns the requested forward tensors, optionally using the cache"

        If use_cache is True and the cached files exist, returns heldin and heldout from
        the cache. If use_cache is False, builds and returns heldin and heldout. If
        use_cache is True but the cached files do not exist, builds heldin and heldout
        before caching and returning them.

        Args:
            dataset_name (str): The name of the NLB dataset to return.
            phase (str): The desired phase. One of ['val','test']
            tensor_set (str): The desired set of tensors. One of ['train', 'eval']
            bin_ms (int): The desired bin width in ms.
            use_cache (bool): Whether or not to use the cache. If False, the
                data will be loaded from raw NLB data. If true and cache files
                do not exist, the arrays will be loaded from the raw NLB data
                but then cached to file before returning.

        Returns:
            (np.ndarray,np.ndarray): heldin and heldout arrays
                for the requested phase and tensor set
        """
        if use_cache and self._cache_helper.cached_files_exist(
            dataset_name,
            phase=phase,
            bin_ms=bin_ms,
            tensor_set=tensor_set,
            forward=True,
        ):
            return self._cache_helper.load_tensors_from_cache(
                dataset_name,
                bin_ms=bin_ms,
                tensor_set=tensor_set,
                phase=phase,
                forward=True,
            )

        heldin, heldout = self._build_forward_tensors(
            tensor_set=tensor_set,
            phase=phase,
            dataset_name=dataset_name,
            bin_ms=bin_ms,
        )
        if use_cache:
            self._cache_helper.cache_arrays(
                dataset_name,
                bin_ms=bin_ms,
                phase=phase,
                tensor_set=tensor_set,
                heldin=heldin,
                heldout=heldout,
                forward=True,
            )
        return heldin, heldout

    def _build_forward_tensors(
        self, *, dataset_name: str, phase: str, tensor_set: str, bin_ms: int
    ):
        """Return the requested heldin and heldout forward tensors

        Args:
            dataset_name (str): The name of the NLB dataset to return.
            phase (str): The desired phase. One of ['val','test'].  This is
                ignored if `tensor_set=="eval"`
            tensor_set (str): The desired set of tensors. One of ['train', 'eval']
            bin_ms (int): The desired bin width in ms.

        Returns:
            (np.ndarray, np.ndarray): The requested forward tensors
        """
        dataset = self._load_dataset(name=dataset_name, bin_ms=bin_ms)
        if tensor_set == "train":
            if phase == "val":
                trial_split = "train"  # type: Union[str, List[str]]
            elif phase == "test":
                trial_split = ["train", "val"]
            else:
                raise ValueError(
                    f"phase must be one of ['val','test'] but given {phase}"
                )
            tensor_dict = make_train_input_tensors(
                dataset,
                dataset_name=dataset_name,
                trial_split=trial_split,
                save_file=False,
                include_forward_pred=True,
            )
            heldin, heldout = (
                tensor_dict["train_spikes_heldin_forward"].astype(float),
                tensor_dict["train_spikes_heldout_forward"].astype(float),
            )
        elif tensor_set == "eval":
            tensor_dict = make_eval_target_tensors(
                dataset, dataset_name=dataset_name, save_file=False
            )
            # need to extract tensors from second level of dict
            # noqa: see https://github.com/neurallatents/nlb_tools/blob/main/nlb_tools/make_tensors.py#L618-L626
            suf = "" if (dataset.bin_width == 5) else f"_{dataset.bin_width}"
            dataset_key = dataset_name + suf
            heldin, heldout = (
                tensor_dict[dataset_key]["eval_spikes_heldin_forward"].astype(float),
                tensor_dict[dataset_key]["eval_spikes_heldout_forward"].astype(float),
            )
        else:
            raise ValueError(
                f"Unknown tensor_set: {tensor_set}.  Must be one of ['train','eval']"
            )

        return heldin, heldout

    def _get_tensors_for_phase(
        self,
        *,
        phase: str,
        tensor_set: str,
        dataset_name: str,
        bin_ms: int,
        use_cache: bool,
    ) -> TwoArrayTuple:
        """Returns requested heldin and heldout tensors, optionally using the cache.

        If use_cache is True and the cached files exist, returns heldin and heldout from
        the cache. If use_cache is False, builds and returns heldin and heldout. If
        use_cache is True but the cached files do not exist, builds heldin and heldout
        before caching and returning them.

        Args:
            phase (str): The desired phase. One of ['val', 'test']
            tensor_set (str): The desired set of tensors. One of ['train', 'eval']
            dataset_name (str): The name of the NLB dataset to return.
            bin_ms (int): The desired bin width in ms.
            use_cache (bool): Whether or not to use the cache. If False, the
                data will be loaded from raw NLB data. If true and cache files
                do not exist, the arrays will be loaded from the raw NLB data
                but then cached to file before returning.

        Returns:
            (np.ndarray, Optional[np.ndarray]): heldin and heldout arrays
                for the requested phase and tensor set
        """
        if use_cache and self._cache_helper.cached_files_exist(
            dataset_name,
            bin_ms=bin_ms,
            phase=phase,
            tensor_set=tensor_set,
        ):
            return self._cache_helper.load_tensors_from_cache(
                dataset_name, bin_ms=bin_ms, phase=phase, tensor_set=tensor_set
            )

        heldin, heldout = self._build_tensor_for_phase(
            phase=phase,
            tensor_set=tensor_set,
            dataset_name=dataset_name,
            bin_ms=bin_ms,
        )
        if use_cache:
            self._cache_helper.cache_arrays(
                dataset_name,
                bin_ms=bin_ms,
                phase=phase,
                tensor_set=tensor_set,
                heldin=heldin,
                heldout=heldout,
            )
        return heldin, heldout

    def _build_tensor_for_phase(
        self, *, phase: str, tensor_set: str, dataset_name: str, bin_ms: int
    ) -> TwoArrayTuple:
        """Builds heldin and heldout for the requested phase

        This is called when either the cache is missed or is not used.

        Args:
            phase (str): The desired phase. One of ['val', 'test']
            tensor_set (str): The desired tensor set. One of ['train', 'eval']
            dataset_name (str): The name of the NLB dataset to return.
            bin_ms (int): The desired bin width in ms.

        Raises:
            ValueError: if the requested tensor_set is not one
                of ['train', 'eval'].

        Returns:
            (np.ndarray, Optional[np.ndarray]): heldin and heldout arrays
                for the requested phase and tensor set
        """
        if tensor_set == "train":
            heldin, heldout = self._build_train_tensors(
                dataset_name, phase=phase, bin_ms=bin_ms
            )
        elif tensor_set == "eval":
            heldin, heldout = self._build_validation_tensors(
                dataset_name, phase=phase, bin_ms=bin_ms
            )
        else:
            raise ValueError(
                f"Unknown tensor_set: {tensor_set}.  Must be one of ['train','eval']"
            )
        return heldin, heldout

    def _build_train_tensors(
        self, dataset_name: str, *, phase: str, bin_ms: int
    ) -> TwoArrayTuple:
        """Builds training heldin and heldout from the NLB dataset.

        Args:
            dataset_name (str): The name of the NLB dataset to return.
            phase (str): The desired phase. One of ['val', 'test']
            bin_ms (int): The desired bin width in ms.
        Raises:
            ValueError: if the requested phase is not one
                of ['val', 'test'].
        Returns:
            (np.ndarray, np.ndarray): heldin and heldout train arrays
        """
        dataset = self._load_dataset(name=dataset_name, bin_ms=bin_ms)
        if phase == "val":
            train_dict = make_train_input_tensors(
                dataset, dataset_name=dataset_name, trial_split="train", save_file=False
            )
        elif phase == "test":
            train_dict = make_train_input_tensors(
                dataset,
                dataset_name=dataset_name,
                trial_split=["train", "val"],
                save_file=False,
            )
        else:
            raise ValueError(f"Unknown phase: {phase}. Must be on of ['val', 'test']")

        heldin_train = train_dict["train_spikes_heldin"].astype(float)
        heldout_train = train_dict["train_spikes_heldout"].astype(float)
        return heldin_train, heldout_train

    def _build_validation_tensors(
        self, dataset_name: str, *, phase: str, bin_ms: int
    ) -> TwoArrayTuple:
        """Builds validation heldin and heldout from the NLB dataset.

        Args:
            dataset_name (str): The name of the NLB dataset to return.
            phase (str): The desired phase. One of ['val', 'test']
            bin_ms (int): The desired bin width in ms.

        Raises:
            ValueError: if the requested phase is not one
                of ['val', 'test'].

        Returns:
            (np.ndarray, Optional[np.ndarray]): heldin and heldout validation arrays
        """
        dataset = self._load_dataset(name=dataset_name, bin_ms=bin_ms)
        if phase == "val":
            val_dict = make_eval_input_tensors(
                dataset, dataset_name=dataset_name, trial_split="val", save_file=False
            )
        elif phase == "test":
            val_dict = make_eval_input_tensors(
                dataset, dataset_name=dataset_name, trial_split="test", save_file=False
            )
        else:
            raise ValueError(f"Unknown phase: {phase}. Must be on of ['val', 'test']")

        heldin_val = val_dict["eval_spikes_heldin"].astype(float)
        # if phase == test there are no holdout rates
        heldout_val = val_dict.get("eval_spikes_heldout", None)
        heldout_val = heldout_val if heldout_val is None else heldout_val.astype(float)
        return heldin_val, heldout_val

    @lru_cache(maxsize=1)
    def _load_dataset(self, *, name: str, bin_ms: int) -> NWBDataset:
        """Returns a resampled NLB dataset loaded from disk or memory.

        Uses an in-memory cache (lru_cache) to speed up duplicate consecutive
        requests. Uses maxsize=1 because each dataset can be really big, so we
        want to avoid holding more than one dataset in memory at a time.

        Args:
            name (str): The name of the NLB dataset to return.
            bin_ms (int): The desired bin width in ms.

        Raises:
            ValueError: If the dataset name is unknown

        Returns:
            NWBDataset: the resampled NLB dataset object.
        """
        dataset_names = list(DATASET_PATHS.keys())
        if name not in dataset_names:
            raise ValueError(
                f"unknown dataset: {name}. Expected one of: {dataset_names}"
            )
        dataset = NWBDataset(DATASET_PATHS[name])
        dataset.resample(bin_ms)
        return dataset


class CacheHelper:
    """A class which assists in caching tensors to disk."""

    def __init__(self, *, cache_path: Path = None):
        """Instantiate a CacheHelper.

        If cache_path is None, DEFAULT_CACHE_DIR is used as the default.

        Args:
            cache_path (Path, optional): A path to the directory where files
                should be cached. Defaults to None.
        """
        if cache_path is None:
            self.cache_path = DEFAULT_CACHE_DIR
        else:
            self.cache_path = cache_path
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def cached_files_exist(
        self,
        dataset_name: str,
        *,
        bin_ms: int,
        phase: str,
        tensor_set: str,
        forward: bool = False,
    ) -> bool:
        """Checks to see if the required cache files exist on disk.

        Args:
            dataset_name (str): The name of the cached dataset.
            bin_ms (int): The requested bin width in ms.
            phase (str): The requested phase. One of ['val', 'test']
            tensor_set (str): The desired set of tensors. One of ['train', 'eval']
            forward (bool): If true, check for forward tensors

        Returns:
            bool: True if all required tensors exist on disk, otherwise False.
        """
        for file_path in self._get_cache_filepaths(
            dataset_name,
            bin_ms=bin_ms,
            phase=phase,
            tensor_set=tensor_set,
            forward=forward,
        ):
            if not file_path.exists():
                return False
        return True

    def load_tensors_from_cache(
        self,
        dataset_name: str,
        *,
        bin_ms: int,
        phase: str,
        tensor_set: str,
        forward: bool = False,
    ) -> TwoArrayTuple:
        """Loads the cached tensors from disk.

        Args:
            dataset_name (str): The name of the cached dataset.
            bin_ms (int): The requested bin width in ms.
            phase (str): The requested phase. One of ['val', 'test']
            tensor_set (str): The desired set of tensors. One of ['train', 'eval']
            forward (bool): If true, load the forward tensors from the cache

        Returns:=
            (np.ndarray, Optional[np.ndarray]): heldin and heldout arrays
                loaded from cache.
        """
        file_paths = self._get_cache_filepaths(
            dataset_name,
            bin_ms=bin_ms,
            phase=phase,
            tensor_set=tensor_set,
            forward=forward,
        )
        if phase == "test" and tensor_set == "eval":
            heldin = np.load(file_paths[0]).astype(float)
            heldout = None
        else:
            heldin, heldout = [
                np.load(save_path).astype(float) for save_path in file_paths
            ]
        return heldin, heldout

    def _get_cache_filepaths(
        self,
        dataset_name: str,
        *,
        bin_ms: int,
        phase: str,
        tensor_set: str,
        forward: bool,
    ) -> List[Path]:
        """Returns paths to the requested tensor cache files.

        Args:
            dataset_name (str): The name of the cached dataset.
            bin_ms (int): The requested bin width in ms.
            phase (str): The requested phase. One of ['val', 'test']
            tensor_set (str): The desired set of tensors. One of ['train', 'eval']
            forward (bool): If true, get the filenames for the forward tensors

        Returns:
            List[Path]: the paths where the specified tensors would be cached.
        """
        suffix = "_forward" if forward else ""
        filenames = [
            self.cache_path
            / f"{dataset_name}_{bin_ms}ms_{phase}_{tensor_set}_heldin{suffix}.npy",
            self.cache_path
            / f"{dataset_name}_{bin_ms}ms_{phase}_{tensor_set}_heldout{suffix}.npy",
        ]
        if phase == "test" and tensor_set == "eval":
            filenames = filenames[:1]
        return filenames

    def cache_arrays(
        self,
        dataset_name: str,
        *,
        bin_ms: int,
        phase: str,
        tensor_set: str,
        heldin: np.ndarray,
        heldout: Optional[np.ndarray],
        forward: bool = False,
    ) -> None:
        """Caches heldin and heldout tensors to disk.

        Args:
            dataset_name (str): The name of the cached dataset.
            bin_ms (int): The requested bin width in ms.
            phase (str): The requested phase.
            tensor_set (str): The desired set of tensors.
            heldin (np.ndarray): The heldin tensor to cache to disk.
            heldout (np.ndarray): The heldout tensor to cache to disk.
            forward (bool): If true, cache as a forward tensor
        """
        save_paths = self._get_cache_filepaths(
            dataset_name,
            bin_ms=bin_ms,
            phase=phase,
            tensor_set=tensor_set,
            forward=forward,
        )
        if phase == "test" and tensor_set == "eval":
            np.save(save_paths[0], heldin)
        else:
            for save_path, arr in zip(save_paths, [heldin, heldout]):
                np.save(save_path, arr)


class NDTDataHandler:
    """Create an NDT compatible h5 file for training and testing NDT models"""

    def __init__(self, data_dir: Optional[Path] = None, nlb_cache_path: Optional[Path] = None):
        """Initialize a NDTDataHandler

        Args:
            data_dir (Optional[Path]): The path to save the ndt compatible
                h5 datasets. Defaults to `DEFAULT_DATA_DIR`.
            nlb_cache_path (Optional[Path]): A path which is passed to NLBDataHandler on
                creation. Defaults to `None` which uses the default of NLBDataHandler.
        """
        self.nlb_data = NLBDataHandler(cache_path=nlb_cache_path)
        self.data_dir = data_dir or DEFAULT_NDT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)


    def make_train_data(self, dataset_name: str, *, overwrite: bool = False) -> None:
        """Save the training data to NDT compatible h5 file

        Args:
            dataset_name (str): The desired dataset
            overwrite (bool): If True overwrite an existing file. Defaults to False.
        """
        self._make_data_file(dataset_name, phase="val", overwrite=overwrite)

    def make_test_data(self, dataset_name: str, *, overwrite: bool = False) -> None:
        """Save the test data to NDT compatible h5 file

        Args:
            dataset_name (str): The desired dataset
            overwrite (bool): If True overwrite existing file. Defaults to False.
        """
        self._make_data_file(dataset_name, phase="test", overwrite=overwrite)

    def get_filename(self, dataset_name: str, *, phase: str) -> str:
        """Get the full filepath for a given dataset

        Args:
            dataset_name (str): The desired dataset.
            phase (str): The desired phase. One of ['val','test']

        Raises:
            ValueError: if the requested phase is not one
                of ['val', 'test'].

        Returns:
            str, the name of the associated file
        """
        if phase == "val":
            suffix = "train"
        elif phase == "test":
            suffix = "test"
        else:
            raise ValueError(
                f"The phase must be one of ['val','test'] but given {phase}"
            )
        return str(self.data_dir / f"{dataset_name}_{suffix}.h5")

    def _make_data_file(
        self, dataset_name: str, *, phase: str, overwrite: bool = False
    ) -> None:
        """Save the data to an NDT compatible h5 file

        Args:
            dataset_name (str): The desired dataset
            phase (str): The desired phase. One of ['val', 'test'].
            overwrite (bool): If True overwrite existing file. Defaults to False.

        """
        filename = self.get_filename(dataset_name=dataset_name, phase=phase)
        if not overwrite and self._file_exists(dataset_name=dataset_name, phase=phase):
            logger.info(
                f"File exists at {filename}. Skipping creating of training data"
            )
            return
        data_dict = self._build_data_dict(dataset_name, phase=phase)
        logger.info(f"Saving training data to {filename}")
        self._save_dict_to_h5(data_dict, dataset_name=dataset_name, phase=phase)

    def _build_data_dict(self, dataset_name: str, *, phase: str):
        """Create an NDT compatible dictionary or data

        Args:
            dataset_name (str): The desired dataset.
            phase (str): The desired phase. One of ['val', 'test'].
                Note that phase is ignored for making eval tensors.

        See Also:
            src.data.NLBDataHandler.get_eval_forward_tensors
        """
        train_heldin, train_heldout = self.nlb_data.get_train_tensors(
            dataset_name, phase=phase
        )
        (
            train_heldin_forward,
            train_heldout_forward,
        ) = self.nlb_data.get_train_forward_tensors(dataset_name, phase=phase)
        eval_heldin, eval_heldout = self.nlb_data.get_eval_tensors(
            dataset_name, phase=phase
        )
        (
            eval_heldin_forward,
            eval_heldout_forward,
        ) = self.nlb_data.get_eval_forward_tensors(dataset_name, phase=phase)

        data = dict(
            train_data_heldin=train_heldin,
            train_data_heldout=train_heldout,
            train_data_heldin_forward=train_heldin_forward,
            train_data_heldout_forward=train_heldout_forward,
            eval_data_heldin=eval_heldin,
            eval_data_heldout=eval_heldout,
            eval_data_heldin_forward=eval_heldin_forward,
            eval_data_heldout_forward=eval_heldout_forward,
        )
        if phase == "test":
            keep = [
                "train_data_heldin",
                "train_data_heldout",
                "train_data_heldin_forward",
                "train_data_heldout_forward",
                "eval_data_heldin",
            ]
            data = {k: data[k] for k in keep}
        return data

    def _save_dict_to_h5(self, data: dict, *, dataset_name: str, phase: str) -> None:
        """Save a dictionary to h5

        Args:
            data (dict): the dictionary of data to save
        """
        filename = self.get_filename(dataset_name=dataset_name, phase=phase)
        save_to_h5(data, filename, overwrite=True)

    def _file_exists(self, *, dataset_name: str, phase: str) -> bool:
        """Returns if the file exists already or not

        Args:
            dataset_name (str): The desired dataset.
            phase (str): The desired phase. One of ['val','test']

        Returns:
            bool, True if the file exists else False
        """
        filename = self.get_filename(dataset_name=dataset_name, phase=phase)
        return Path(filename).exists()
