import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np
from dezero import cuda
import dezero.mlx


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False, mlx=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        if isinstance(gpu, str):
            device = gpu.lower()
            self.gpu = device in ('gpu', 'cuda')
            self.mlx = (device == 'mlx')
        else:
            self.gpu = bool(gpu)
            self.mlx = bool(mlx)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def _get_xp(self):
        if self.mlx:
            return dezero.mlx.xp_mlx
        elif self.gpu:
            return cuda.cupy
        return np

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = self._get_xp()
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False
        self.mlx = False

    def to_gpu(self):
        self.gpu = True
        self.mlx = False

    def to_mlx(self):
        self.gpu = False
        self.mlx = True

    def to_device(self, device='cpu'):
        device = str(device).lower()
        if device == 'mlx':
            self.to_mlx()
        elif device in ('gpu', 'cuda'):
            self.to_gpu()
        elif device == 'cpu':
            self.to_cpu()
        else:
            raise ValueError("Unsupported device: {}. Use 'cpu', 'gpu', or 'mlx'.".format(device))


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False, mlx=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         gpu=gpu, mlx=mlx)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = self._get_xp()
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t