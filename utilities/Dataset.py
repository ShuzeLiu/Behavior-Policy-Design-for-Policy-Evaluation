import torch

class OPEDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data):
        'Initialization'
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.data[index]
        # return (self.data[index][0],self.data[index][1]) , (self.data[index][2],self.data[index][3],self.data[index][4])