from model import MnistCNN
import torch

DEFAULT_MASKED_MODULES = (torch.nn.Conv2d,torch.nn.Linear)
# https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809
def to_sparse(x):
    """ converts dense tensor x to sparse format """
    if isinstance(x, DEFAULT_MASKED_MODULES):
        # x_typename = torch.typename(x).split('.')[-1]
        # sparse_tensortype = getattr(torch.sparse, x_typename)
        #
        # indices = torch.nonzero(x)
        # if len(indices.shape) == 0:  # if all elements are zeros
        #     return sparse_tensortype(*x.shape)
        # indices = indices.t()
        # values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        # return sparse_tensortype(indices, values, x.size())
        print(torch.nonzero(x))
        indices = torch.nonzero(x).t()

        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return torch.sparse.FloatTensor(indices, values, x.size())


def main():

    net = MnistCNN()
    net.apply(to_sparse)
    # for i in net.children()

    # https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809

if __name__ == '__main__':
    main()