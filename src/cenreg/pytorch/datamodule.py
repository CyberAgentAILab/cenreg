import torch
from torch.utils.data import DataLoader


class ProbDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def len_feature(self):
        return self.features.shape[1]


class ProbDataModule(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, features, targets):
        return DataLoader(
            ProbDataset(features, targets),
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self, features, targets):
        return DataLoader(
            ProbDataset(features, targets),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self, features, targets):
        return DataLoader(
            ProbDataset(features, targets),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )


class SurvDataset(torch.utils.data.Dataset):
    def __init__(self, features, observed_times, events):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.events = torch.tensor(events, dtype=torch.int64)
        self.observed_times = torch.tensor(observed_times, dtype=torch.float32)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.features[idx], self.observed_times[idx], self.events[idx]

    def len_feature(self):
        return self.features.shape[1]


class SurvDataModule(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self, features, observed_times, events):
        return DataLoader(
            SurvDataset(features, observed_times, events),
            batch_size=self.batch_size,
            num_workers=1,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self, features, observed_times, events):
        return DataLoader(
            SurvDataset(features, observed_times, events),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self, features, observed_times, events):
        return DataLoader(
            SurvDataset(features, observed_times, events),
            batch_size=99999999,
            num_workers=1,
            persistent_workers=True,
            shuffle=False,
        )
