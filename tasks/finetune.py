import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torchtext import data

TEXT = data.Field(
    lower=True, include_lengths=True, batch_first=True, fix_length=max_len
)
LABEL = data.Field(sequential=False)

if dataset == "IMDB":
    fields = {"text": ("text", TEXT), "label": ("label", LABEL)}

    train_data, valid_data = data.TabularDataset.splits(
        path=data_dir,
        train=train_file,
        test=valid_file,
        format="JSON",
        fields=fields,
    )
elif dataset == "AmazonReviewPolarity":
    fields = [("label", LABEL), ("name", None), ("text", TEXT)]

    train_data, valid_data = data.TabularDataset.splits(
        path=data_dir,
        train=train_file,
        test=valid_file,
        format="CSV",
        fields=fields,
    )
else:
    raise ValueError("Dataset: {} currently not supported".format(dataset))

TEXT.build_vocab(train_data, max_size=vocab_size - 2)
LABEL.build_vocab(train_data)

# Set device type
if cuda and torch.cuda.is_available():
    print("Running on GPU")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("-" * 84)
print("Running on device type: {}".format(device))

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=batch_size,
    sort_key=lambda x: x.text,
    sort_within_batch=False,
    device=device,
)


optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

# Start Training
print("-" * 84)
print("Start Training")
start_time = time.time()
for epoch in range(num_epochs):
    model.train()

    per_epoch_loss = 0
    for i, batch in enumerate(train_iterator):
        optimizer.zero_grad()

        # Get text and label
        text = batch.text[0]
        label = batch.label - 1

        # Trim input text
        if text.size(1) > max_len:
            text = text[:, :max_len]

        out = model(text)
        loss = nn.functional.nll_loss(out, label)

        per_epoch_loss += loss.item()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()