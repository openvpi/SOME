import pathlib
import random
import warnings
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    """
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    """

    def __init__(self, config: dict, data_attrs=None):
        self.config = config
        self.raw_data_dirs = [pathlib.Path(d) for d in config['raw_data_dir']]
        self.binary_data_dir = pathlib.Path(self.config['binary_data_dir'])
        self.data_attrs = [] if data_attrs is None else data_attrs

        self.binarization_args = self.config['binarization_args']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.items = {}
        self.item_names: list = None
        self._train_item_names: list = None
        self._valid_item_names: list = None

        self.timestep = self.config['hop_size'] / self.config['audio_sample_rate']

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        raise NotImplementedError()

    def split_train_valid_set(self):
        """
        Split the dataset into training set and validation set.
        :return: train_item_names, valid_item_names
        """
        prefixes = set([str(pr) for pr in self.config['test_prefixes']])
        valid_item_names = set()
        # Add prefixes that specified speaker index and matches exactly item name to test set
        for prefix in deepcopy(prefixes):
            if prefix in self.item_names:
                valid_item_names.add(prefix)
                prefixes.remove(prefix)
        # Add prefixes that exactly matches item name without speaker id to test set
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.split(':')[-1] == prefix:
                    valid_item_names.add(name)
                    matched = True
            if matched:
                prefixes.remove(prefix)
        # Add names with one of the remaining prefixes to test set
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.startswith(prefix):
                    valid_item_names.add(name)
                    matched = True
            if matched:
                prefixes.remove(prefix)
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.split(':')[-1].startswith(prefix):
                    valid_item_names.add(name)
                    matched = True
            if matched:
                prefixes.remove(prefix)

        if len(prefixes) != 0:
            warnings.warn(
                f'The following rules in test_prefixes have no matching names in the dataset: {sorted(prefixes)}',
                category=UserWarning
            )
            warnings.filterwarnings('default')

        valid_item_names = sorted(list(valid_item_names))
        assert len(valid_item_names) > 0, 'Validation set is empty!'
        train_item_names = [x for x in self.item_names if x not in set(valid_item_names)]
        assert len(train_item_names) > 0, 'Training set is empty!'

        return train_item_names, valid_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._valid_item_names

    def meta_data_iterator(self, prefix):
        if prefix == 'train':
            item_names = self.train_item_names
        else:
            item_names = self.valid_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        # load each dataset
        for ds_id, data_dir in zip(range(len(self.raw_data_dirs)), self.raw_data_dirs):
            self.load_meta_data(pathlib.Path(data_dir), ds_id=ds_id)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._valid_item_names = self.split_train_valid_set()

        if self.binarization_args['shuffle']:
            random.seed(self.config['seed'])
            random.shuffle(self.item_names)

        self.binary_data_dir.mkdir(parents=True, exist_ok=True)
        self.check_coverage()

        # Process valid set and train set
        try:
            self.process_dataset('valid')
            self.process_dataset(
                'train',
                num_workers=int(self.binarization_args['num_workers']),
                apply_augmentation=True
            )
        except KeyboardInterrupt:
            exit(-1)

    def check_coverage(self):
        pass

    def process_dataset(self, prefix, num_workers=0, apply_augmentation=False):
        args = []
        builder = IndexedDatasetBuilder(self.binary_data_dir, prefix=prefix, allowed_attr=self.data_attrs)
        lengths = []
        total_sec = 0
        total_raw_sec = 0

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, apply_augmentation])

        def postprocess(_item, _is_raw=True):
            nonlocal total_sec, total_raw_sec
            if _item is None:
                return
            builder.add_item(_item)
            lengths.append(_item['length'])
            total_sec += _item['seconds']
            if _is_raw:
                total_raw_sec += _item['seconds']

        try:
            if num_workers > 0:
                # code for parallel processing
                for items in tqdm(
                        chunked_multiprocess_run(self.process_item, args, num_workers=num_workers),
                        total=len(list(self.meta_data_iterator(prefix)))
                ):
                    for i, item in enumerate(items):
                        postprocess(item, i == 0)
            else:
                # code for single cpu processing
                for a in tqdm(args):
                    items = self.process_item(*a)
                    for i, item in enumerate(items):
                        postprocess(item, i == 0)
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()
        with open(self.binary_data_dir / f'{prefix}.lengths', 'wb') as f:
            # noinspection PyTypeChecker
            np.save(f, lengths)

        if apply_augmentation:
            print(f'| {prefix} total duration (before augmentation): {total_raw_sec:.2f}s')
            print(
                f'| {prefix} total duration (after augmentation): {total_sec:.2f}s ({total_sec / total_raw_sec:.2f}x)')
        else:
            print(f'| {prefix} total duration: {total_raw_sec:.2f}s')

    def process_item(self, item_name, meta_data, allow_aug=False):
        raise NotImplementedError()
