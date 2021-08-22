from torch.utils.data import Dataset
import pandas as pd
from ReviewVectorizer import ReviewVectorizer


class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        :param review_df: the dataset
        :param vectorizer: vectorizer instantiated from dataset
        """

        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.validation_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """
        :param review_csv: location of the dataset
        :return: an instance of ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        """
        :return:
            return the vectorizer
        """
        return self._vectorizer

    def set_split(self, split="train"):
        """
        selects the splits in the dataset using a column in the dataframe

        :param split: one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """
        the primary entry point method for PyTorch datasets

        :param index: the index to the data point
        :return: a dict of the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        review_vector = self._vectorizer.vectorize(row.review)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)

        return {
            'x_data': review_vector,
            'y_target': rating_index
        }

    def get_num_batches(self, batch_size):
        """
        Given a batch size, return the number of batches in the dataset

        :param batch_size: (int)
        :return: number of batches in the dataset
        """

        return len(self) // batch_size
