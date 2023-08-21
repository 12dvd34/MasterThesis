class CombinedDataloader:
    def __init__(self, dataloader1, dataloader2):
        self.data = []
        self.batch_size = None
        self.data.extend(dataloader1)
        self.batch_size = self.data[0][0].size(0)
        assert dataloader2[0][0].size(0) == self.batch_size, "dataloaders batch size should match"
        self.data.extend(dataloader2)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)