# import sys
# sys.path.append('transformer')
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets
from argparse import ArgumentParser
# from torch.utils.tensorboard import SummaryWriter
import tqdm     # CLI progress bar

from transformer.tc_transformer import TCTransformer

TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(sequential=False)
NUM_CLASS = 2

"""
Creates and trains a basic transformer for the IMDB sentiment classification task.
"""
def main(arg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    # load the IMDB data
    if arg.final:
        train, test = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(train, max_size=arg.vocab_size - 2) # - 2 to make space for <unk> and <pad>
        LABEL.build_vocab(train)
        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=device)
    else:
        tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
        train, test = tdata.split(split_ratio=0.8)
        TEXT.build_vocab(train, max_size=arg.vocab_size - 2)
        LABEL.build_vocab(train)
        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=arg.batch_size, device=device)

    print('num of training examples: ', len(train_iter))
    print('num of test examples: ', len(test_iter))

    if arg.max_length < 0:
        mx = max([input.text[0].size(1) for input in train_iter])
        mx = mx * 2
    else:
        mx = arg.max_length
    print('maximum seq length: ', mx)

    # create the model
    model = TCTransformer(emb=arg.embedding_size,
                          heads=arg.num_heads,
                          depth=arg.depth,
                          seq_length=mx,
                          num_tokens=arg.vocab_size,
                          num_classes=NUM_CLASS,
                          dropout=arg.dropout)
    if device == 'cuda':
        model.cuda()

    optim = torch.optim.Adam(lr=arg.lr, params=model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # Training loop
    seen = 0
    for epoch in range(arg.num_epochs):
        print("\n Epoch: ", epoch)
        model.train(True)
        for batch in tqdm.tqdm(train_iter):
            optim.zero_grad()
            input = batch.text[0]
            label = batch.label - 1
            if input.size(1) > mx:
                input = input[:, :mx]
            output = model(input)
            loss = F.nll_loss(output, label)
            loss.backward()

            # clip gradients vector length if > 1 to 1
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            optim.step()
            sched.step()
            seen += input.size(0)

        with torch.no_grad():
            model.train(False)
            total, correct = 0.0, 0.0

            for batch in test_iter:
                inp = batch.text[0]
                label = batch.label - 1
                if inp.size(1) > mx:
                    inp = inp[:, :mx]
                out = model(inp).argmax(dim=1)

                total += float(inp.size(0))
                correct += float((label == out).sum().item())
            acc = correct / total
            print(f'-- {"Test" if arg.final else "Validation"} accuracy {acc:.3}')


if __name__ == '__main__':
    parser = ArgumentParser()

    """
    Arguments:
        --num_epochs,   -e  Number of epochs
        --batch-size,   -b  Batch size
        --learn-rate,   -l  Learning rate
        --tb_dir,       -T  Tensorboard logging directory 
        --final,        -f  Whether to run on the real test set (if not included, validation set is used)
        --max_pool,         Use max pooling in final classification layer
        --embedding     -E  Size of character embeddings
        --vocab-size    -V  Vocabulary size
        --max           -M  Max sequence length. Longer sequences are clipped (-1 for no limit)
        --heads         -H  Number of attention heads
        --depth         -d  Depth of the network (num of self-attention layers)
        --random-seed   -r  RNG seed
        --lr-warmup         Learning rate warmup
        --gradient-clipping Gradient clipping
        --dropout       -D  Dropout
    """

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=5, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0005, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=50, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("-D", "--dropout",
                        dest="dropout",
                        help="Dropout rate.",
                        default=0.2, type=float)

    options = parser.parse_args()
    print('OPTIONS', options)

    main(options)