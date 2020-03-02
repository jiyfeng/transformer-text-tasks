from argparse import ArgumentParser


def main(arg):
    """

    :param arg:
    :return:
    """

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
                        default=1_000_000, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=32, type=int)

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file. Will be read as a string of 8-bit characters.",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=256, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr of self-attention layers)",
                        default=12, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1500, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=100000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss. This can be a bit bigger than the training batch size.",
                        default=64, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=5000, type=int)

    parser.add_argument("--wide", dest="wide",
                        help="Use wide self attention instead of narrow self attention.",
                        action="store_true")

    options = parser.parse_args()
    print('OPTIONS ', options)

    main(options)
