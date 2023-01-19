from tqdm import tqdm
import numpy as np

class Test(object):
    def __init__(self) -> None:
        self.num = np.arange(24).reshape((8, -1))

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < 8:
            self.idx += 1
            return self.num[self.idx-1]
        raise StopIteration
    

test = Test()
for i, vec in enumerate(test):
    print(i, vec)

