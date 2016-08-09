import itertools as it
import numpy as np
import cv2


def get_cifar10(batch_size=16):
    from skdata.cifar10.dataset import CIFAR10
    cifar10 = CIFAR10()
    cifar10.fetch(True)

    trn_labels = []
    trn_pixels = []
    for i in range(1,6):
        data = cifar10.unpickle("data_batch_%d" % i)
        trn_pixels.append(data['data'])
        trn_labels.extend(data['labels'])

    trn_pixels = np.vstack(trn_pixels)

    tst_data = cifar10.unpickle("test_batch")
    tst_labels = tst_data["labels"]
    tst_pixels = tst_data["data"]

    trn_set = batch_iterator(it.cycle(zip(trn_pixels, trn_labels)), batch_size, batch_fn=lambda x: zip(*x))
    tst_set = (np.vstack(tst_pixels), np.array(tst_labels))

    return trn_set, tst_set

def batch_iterator(iterable, size, cycle=False, batch_fn=lambda x: x):
    """
    Iterate over a list or iterator in batches
    """
    batch = []

    # loop to begining upon reaching end of iterable, if cycle flag is set
    if cycle is True:
        iterable = it.cycle(iterable)

    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch_fn(batch)
            batch = []

    if len(batch) > 0:
        yield batch_fn(batch)





if __name__ == '__main__':
    trn, tst = get_cifar10()
    a = trn.next()
    img =  a[0][0]
    print img.shape
    show = img.reshape(3, 32, 32).transpose(1, 2, 0)
    show = cv2.resize(show, (0,0), fx=10, fy=10)
    cv2.imshow('img', show)
    cv2.waitKey(0)
