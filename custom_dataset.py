import torch
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def scale_inputs(inputs_x, inputs_y, inputs_z, scale_fac = 101.937):
    """
    Scale inputs to actual acceleration values in m/s²
    :param inputs_x: Vibration data of x dimension
    :param inputs_y: Vibration data of y dimension
    :param inputs_z: Vibration data of z dimension
    """
    inputs_x_scaled = inputs_x / scale_fac
    inputs_y_scaled = inputs_y / scale_fac
    inputs_z_scaled = inputs_z / scale_fac

    return inputs_x_scaled, inputs_y_scaled, inputs_z_scaled

def load_and_log(path, dtype=float):
    print(f"reading {path} ...")
    return np.loadtxt(path, dtype=dtype)

def read_txt(input_x_path, input_y_path, input_z_path, gt_path):
    """
    Read inputs and ground truth stored in txt file
    :param input_x_path: Path of file containing vibration data in x dimension
    :param input_y_path: Path of file containing vibration data in y dimension
    :param input_z_path: Path of file containing vibration data in z dimension
    :param gt_path:  Path of file containing labels
    :return: Stacked inputs and labels as numpy array
    """

    inputs_x_np = load_and_log(input_x_path)
    inputs_y_np = load_and_log(input_y_path)
    inputs_z_np = load_and_log(input_z_path)
    gt_np = load_and_log(gt_path, dtype=int)

    inputs_x_np, inputs_y_np, inputs_z_np = scale_inputs(inputs_x_np, inputs_y_np, inputs_z_np)

    print("stacking inputs...")
    inputs_np = np.stack((inputs_x_np, inputs_y_np, inputs_z_np), axis=1)

    return inputs_np, gt_np


def print_parquet_data_information(pumpids, labels):
    """
    Prints some information about the pump parquet dataset
    :param pumpids: List of all pumpids
    :param labels: List of labels corresponding to each pump id
    """

    num_unique_pump_ids = np.unique(pumpids).size

    print("Number of different pumps: {}".format(num_unique_pump_ids))
    print("total normal labels: {}".format(np.size(labels[labels == "normal"])))
    print("total anormal labels: {}".format(np.size(labels[labels != "normal"])))

    for idx, pump_id in enumerate(np.unique(pumpids)):
        normal_labels = np.size(labels[np.logical_and(pumpids == pump_id, labels == "normal")])
        anormal_lables = np.size(labels[np.logical_and(pumpids == pump_id, labels != "normal")])

        print("Idx: {}, ID: {}, Total Labels: {}, Normal Labels: {}, Anormal Labels: {}.".format(idx, pump_id, labels[pumpids == pump_id].size, normal_labels, anormal_lables))

    print("labels: {}".format(labels))
    print("pump_ids: {}".format(pumpids))
    print("type(pump_ids): {}".format(type(pumpids)))


def read_parquet(path, print_data_info = True):
    """
    Read inputs and ground truth stored in parquet file
    :param path: Path of parquet file
    :param normalize:  Normalize inputs
    :param used_fixed_mean_std: Use precalculated mean and variance for normalization
    :param print_data_info: Print some information about the dataset
    :param value_normal_class: Value of normal class
    :param value_anormal_class: Value of anormal class
    :return: Stacked inputs and labels as numpy array
    """
    pyarrow_df = pq.read_table(source=path)

    # Count number of different pumps
    labels = pyarrow_df["Label"].to_numpy()
    inputs_x = pyarrow_df["values_x"].to_numpy()
    inputs_y = pyarrow_df["values_y"].to_numpy()
    inputs_z = pyarrow_df["values_z"].to_numpy()

    pumpids = pyarrow_df["pumpid"].to_numpy()

    if print_data_info:
        print_parquet_data_information(pumpids, labels)

    labels_np = np.asarray([0 if label == "normal" else 1 for label in labels], dtype=int)
    inputs_x_np = np.asarray([np.fromstring(input[1 : -1], dtype=float, sep=', ') for input in inputs_x])
    inputs_y_np = np.asarray([np.fromstring(input[1 : -1], dtype=float, sep=', ') for input in inputs_y])
    inputs_z_np = np.asarray([np.fromstring(input[1 : -1], dtype=float, sep=', ') for input in inputs_z])

    inputs_x_np, inputs_y_np, inputs_z_np = scale_inputs(inputs_x_np, inputs_y_np, inputs_z_np)

    inputs_np = np.stack((inputs_x_np, inputs_y_np, inputs_z_np), axis=1)

    return inputs_np, labels_np, pumpids




def get_num_pumps_in_parquet_dataset(parquet_path):

    inputs, gts, pump_ids = read_parquet(parquet_path)

    unique_pump_ids = np.unique(pump_ids)
    return len(unique_pump_ids)



# PumpVibrationDataset
class CustomDataset(Dataset):
    """
    Dataset containing vibration data of pumps
    """

    def __init__(self,
                 dataset_path,
                 validation_split=0.2,
                 device="cpu",
                 use_seed=False,
                 training_mode="sup",
                 test_pump_generalization = False,
                 pump_eval_indexes=0,
                 equally_distributed_train_data=False,
                 do_calc_mean_val=False,
                 do_remove_pumps_with_only_one_class=False):
        """
        Init dataset
        :param dataset_path: Path of dataset in parquet format
        :param validation_split: Percentage of dataset used for validation
        :param device: cpu or gpu
        :param use_seed: Use seed for random generator
        :param training_mode: Training mode "sup", "sup_using_g", "ae", "gan" ...
        :param test_pump_generalization: Test how well the model generalizes to unseen pumps
        :param pump_eval_indexes: Indexes of pumps to use for evaluation if test_pump_generalization is True
        :param value_normal_class: Value of normal class
        :param value_anormal_class: Value of anormal class
        :param equally_distributed_train_data: Distribute training data equally between classes
        :param do_calc_max_abs_val: Calculate max absolute value of normal inputs in x,y and z dimension for each pump
        :param do_calc_mean_val: Calculate mean value of normal inputs in x,y and z dimension for each pump
        :param do_remove_pumps_with_only_one_class: Remove pumps with only one class from dataset
        """

        self.device = device
        self.inputs, self.gts, self.pump_ids = read_parquet(dataset_path)

        if self.pump_ids is None and test_pump_generalization:
            raise NotImplementedError("Testing for generalization of pump ids only implemented for parquet dataset.")

        self.unique_pump_ids = np.unique(self.pump_ids) if self.pump_ids is not None else None
        self.selected_pump_indexes = pump_eval_indexes

        # Keep full dataset on cpu first, only pass to gpu when reading samples
        self.inputs = torch.tensor(self.inputs, device="cpu", dtype=torch.float32)
        self.num_samples = self.inputs.shape[0]
        self.sequence_len = self.inputs.shape[2]

        print("inputs shape: {}".format(self.inputs.shape))
        print("gts shape: {}".format(self.gts.shape))
        print("num_samples: {}".format(self.num_samples))
        print("selected_pump_indexes: {}".format(self.selected_pump_indexes))

        # Add channel dimension to inputs to be used in CNN
        if self.inputs.ndim == 2:
            self.inputs = self.inputs[:, None, :]

        # Keep full dataset on cpu first, only pass to gpu when reading samples
        self.gts = torch.tensor(self.gts, device="cpu", dtype=torch.float32)
        self.classes = torch.unique(self.gts)
        self.test_pump_generalization = test_pump_generalization

        for c in self.classes:
            print("Num samples with gt = {}: {}".format(c, torch.numel(self.gts[self.gts == c])))

        # self.dataset_selected_pump_sample indexes is a list of indexes of all samples of the selected pumps in the dataset
        self.train_indexes, self.eval_indexes, self.dataset_selected_pump_sample_indexes = self.gen_train_eval_indexes(validation_split, use_seed)

        if equally_distributed_train_data:
            self.equally_distribute_training_data()

        # Generate sampler from indexes
        self.train_sampler = SubsetRandomSampler(self.train_indexes)
        print("Number of training samples: {}".format(len(self.train_sampler)))

        self.eval_sampler = SubsetRandomSampler(self.eval_indexes)
        print("Number of evaluation samples: {}".format(len(self.eval_sampler)))

        if self.dataset_selected_pump_sample_indexes is not None:
            self.selected_pumps_sampler = SubsetRandomSampler(self.dataset_selected_pump_sample_indexes)
            print("Number of selected pump samples: {}".format(len(self.selected_pumps_sampler)))

        if do_calc_mean_val:
            self.mean_val = self.calc_mean_val()
        else:
            self.mean_val = None

        if do_remove_pumps_with_only_one_class:
            self.remove_pumps_with_only_one_class()

    def gen_normal_and_anormal_indexes(self, indexes):
        """
        Generate indexes of normal and abnormal samples
        """

        indexes_0 = (self.gts.index_select(0, torch.as_tensor(indexes, dtype=torch.int)) == self.classes[0]).nonzero().squeeze()
        indexes_1 = (self.gts.index_select(0, torch.as_tensor(indexes, dtype=torch.int)) == self.classes[1]).nonzero().squeeze()
        indexes_normal = [indexes[i] for i in indexes_0.tolist()]
        indexes_anormal = [indexes[i] for i in indexes_1.tolist()]

        return indexes_normal, indexes_anormal

    def gen_train_eval_indexes(self, validation_split, use_seed):
        """
        Gets indexes in complete dataset for training dataset, evaluation dataset and potentially selected pumps dataset
        """

        # Index of each sample in dataset
        all_indexes = list(range(self.num_samples))

        dataset_selected_pump_indexes = None
        if self.test_pump_generalization:
            # Split dataset into training set with all pumps except the selected ones and evaluation set with only the selected pumps

            # Indexes in dataset corresponding to selected pump indexes
            dataset_selected_pump_indexes, selected_pump_ids = self.get_dataset_idx_based_on_pump_idx(self.selected_pump_indexes)

            # Indexes in dataset corresponding NOT to selected pump indexes
            dataset_without_selected_pump_indexes = [index for index in all_indexes if index not in dataset_selected_pump_indexes]

            num_eval_samples = int(np.floor(validation_split * len(dataset_without_selected_pump_indexes)))

            # Split non-selected pumps in training and validation set
            np.random.shuffle(dataset_without_selected_pump_indexes)
            train_indexes = dataset_without_selected_pump_indexes[num_eval_samples:]
            eval_indexes = dataset_without_selected_pump_indexes[: num_eval_samples]
        else:
            num_eval_samples = int(np.floor(validation_split * self.num_samples))
            if use_seed:
                np.random.seed(1)

            np.random.shuffle(all_indexes)
            train_indexes, eval_indexes = all_indexes[num_eval_samples:], all_indexes[: num_eval_samples]

        return train_indexes, eval_indexes, dataset_selected_pump_indexes


    def remove_pumps_with_only_one_class(self):
        """
        Remove pumps with samples of only one class from dataset
        """

        print("Removing pumps with only one class from dataset...")
        train_indexes = list(iter(self.train_sampler))
        eval_indexes = list(iter(self.eval_sampler))

        for idx, pump_id in enumerate(self.unique_pump_ids):
            pump_indexes = self.pump_ids == pump_id

            # Remove pumps without normal class from indexes
            pump_inputs = self.inputs[pump_indexes]
            pump_inputs = pump_inputs[self.gts[pump_indexes] == self.classes[0]]
            if pump_inputs.shape[0] == 0:
                # Remove pump from train and eval indexes
                train_indexes = [index for index in train_indexes if self.pump_ids[index] != pump_id]
                eval_indexes = [index for index in eval_indexes if self.pump_ids[index] != pump_id]
                print("Removed pump with idx: {} and id: {}".format(idx, pump_id))

            # Remove pumps without anormal class from indexes
            pump_inputs = self.inputs[pump_indexes]
            pump_inputs = pump_inputs[self.gts[pump_indexes] == self.classes[1]]
            if pump_inputs.shape[0] == 0:
                # Remove pump from train and eval indexes
                train_indexes = [index for index in train_indexes if self.pump_ids[index] != pump_id]
                eval_indexes = [index for index in eval_indexes if self.pump_ids[index] != pump_id]
                print("Removed pump with idx: {} and id: {}".format(idx, pump_id))

        # Update sampler and indexes
        self.train_sampler = SubsetRandomSampler(train_indexes)
        self.eval_sampler = SubsetRandomSampler(eval_indexes)
        self.train_indexes = train_indexes
        self.eval_indexes = eval_indexes


    def equally_distribute_training_data(self):
        """
        Provide equal distribution of normal and abnormal labels for training
        """

        print("Distribution data equally between classes...")
        self.train_indexes_normal, self.train_indexes_anormal = self.gen_normal_and_anormal_indexes(self.train_indexes)
        # Repeat indexes of class 1 until it has the same number of samples as class 0
        if len(self.train_indexes_normal) > len(self.train_indexes_anormal):
            self.train_indexes_anormal = self.train_indexes_anormal * (len(self.train_indexes_normal) // len(self.train_indexes_anormal))
            # Add remaining samples of anormal class
            self.train_indexes_anormal += self.train_indexes_anormal[:len(self.train_indexes_normal) - len(self.train_indexes_anormal)]
        else:
            self.train_indexes_normal = self.train_indexes_normal * (len(self.train_indexes_anormal) // len(self.train_indexes_anormal))
            # Add remaining samples of normal class
            self.train_indexes_normal += self.train_indexes_normal[:len(self.train_indexes_anormal) - len(self.train_indexes_normal)]

        print("Number of normal samples: {}".format(len(self.train_indexes_normal)))
        print("Number of anormal samples: {}".format(len(self.train_indexes_anormal)))
        self.train_indexes = self.train_indexes_normal + self.train_indexes_anormal

    def calc_mean_val(self):
        """
        Calculate mean value of all normal inputs in x,y and z dimension for each pump
        """

        # Init mean value with nan
        mean_val = torch.full((len(self.unique_pump_ids), 3), float('nan'))

        for idx, pump_id in enumerate(self.unique_pump_ids):
            pump_indexes = self.pump_ids == pump_id
            pump_inputs = self.inputs[pump_indexes]
            pump_inputs = pump_inputs[self.gts[pump_indexes] == self.classes[0]]
            if pump_inputs.shape[0] != 0:
                for axis in range(3):
                    mean_val[idx, axis] = torch.mean(pump_inputs[:, axis])

        # create torch tensor with same length as input dataset including corresponding mean value
        mean_val_tensor = torch.zeros((self.num_samples, 3))
        for idx, pump_id in enumerate(self.unique_pump_ids):
            pump_indexes = self.pump_ids == pump_id
            mean_val_tensor[pump_indexes] = mean_val[idx]

        return mean_val_tensor


    def get_dataset_idx_based_on_pump_idx(self, pump_indexes):
        """
        Gets the index of the pump in the complete dataset.
        E.g. if pump_indexes is 1, it will get all the dataset-indexes that correspond to the same pump as the first one:
        1. id-2424
        2. id-1234
        3. id-9569
        4. id-2424
        5. id-2424
        6. id-0204

        if pump_indexes is 1 it will return 1, 4, 5

        :param pump_indexes: Index of the pump to retrieve all indexes from
        :return: all indexes of the specific pump in the dataset, the correpsponding pump_id
        """

        all_indexes = list(range(self.num_samples))

        # Convert pump indexes to list if it is not list already
        pump_indexes = pump_indexes if isinstance(pump_indexes, list) else [pump_indexes]

        # Set all indexes of corresponding samples to true
        is_indexed_sample = np.zeros_like(self.pump_ids, dtype=bool)
        pump_ids = []
        for index in pump_indexes:
            is_indexed_sample = np.logical_or(is_indexed_sample, self.pump_ids == self.unique_pump_ids[index])
            pump_ids.append(self.unique_pump_ids[index])

        # Get indexes of all samples that are not part of eval samples
        dataset_indexes = np.array(all_indexes)[is_indexed_sample].tolist()

        return dataset_indexes, pump_ids

    def __len__(self):
        """
        Get number of samples in dataset
        :return: Number of samples in dataset
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Get one sample of dataset
        :param idx: Index of sample in dataset
        :return: Sample
        """

        input_samples = self.inputs[idx].to(self.device)
        gt_samples = self.gts[idx].to(self.device)

        # Add additional input (mean val of specific pump) to data sample
        if self.mean_val is not None:
            if self.mean_val[idx].isnan().any():
                print("mean is nan for idx: {}, with pump id:{}".format(idx, self.pump_ids[idx]))
                exit(10)
            input_samples = (input_samples, self.mean_val[idx].to(self.device))

        return input_samples, gt_samples

    def change_eval_pump_id(self, new_eval_pump_id):
        """
        Changes the pump id of the eval set to the new_eval_pump_id
        """

        print("Changing eval pump id to: {}".format(new_eval_pump_id))
        indices = list(range(self.num_samples))

        if self.pump_ids is not None:
            eval_pump_ids = [new_eval_pump_id]  # list(range(0, 5))

            eval_pump_indices = np.zeros_like(self.pump_ids, dtype=bool)
            for id in eval_pump_ids:
                eval_pump_indices = np.logical_or(eval_pump_indices, self.pump_ids == self.unique_pump_ids[id])

            num_eval_samples = eval_pump_indices.sum()
            train_indices = np.array(indices)[np.logical_not(eval_pump_indices)].tolist()
            eval_indices = np.array(indices)[eval_pump_indices].tolist()
        else:
            raise NotImplementedError("Testing for generalization of pump ids only implemented for parquet data")

        print("num_eval_samples: {}".format(num_eval_samples))

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.eval_sampler = SubsetRandomSampler(eval_indices)

        print("Number of training samples: {}".format(len(self.train_sampler)))
        print("Number of evaluation samples: {}".format(len(self.eval_sampler)))








