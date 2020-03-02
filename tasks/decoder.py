import torch
import torch.nn.functional as F


def top_k_sampling(probs, k):
    if k == 1:
        return torch.argmax(probs).item()
    top_k = torch.topk(probs, k=k)[1]
    top_k = top_k.type(torch.FloatTensor)
    return int(top_k[torch.multinomial(top_k, 1)].item())


def top_p_sampling(probs, p):

    sorted_probs, sorted_indexes = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # select indexes till prob > p
    selected = (cumulative_probs > p).nonzero()[0]
    top_p = sorted_indexes[: selected]
    if top_p.size()[0] == 0:
        return sorted_indexes[0].item()
    top_p = top_p.type(torch.FloatTensor)
    return int(top_p[torch.multinomial(top_p, 1)].item())


def beam_search(tokenizer, model, num_words, k, source_sent):
    i = 0
    indexed_tokens = tokenizer.encode(source_sent)
    # Store in queue - indexed_tokens
    is_first = True
    results, sequences = list(), list()
    queue = [indexed_tokens]
    while len(queue) and i < k**num_words:
        indexed_tokens = queue.pop(0)
        last_word = indexed_tokens[-1]
        if tokenizer.decode(last_word) == "<|endoftext|>":
            results.append(tokenizer.decode(indexed_tokens))
            continue
        if tokenizer.decode(last_word) == ".":
            if not is_first:
                is_first = False
                continue

        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0][0][-1]
            probs = F.softmax(predictions)

        candidates = list()
        seq, score = list(), 0
        for j in range(len(probs)):
            candidate = (seq + [j], score + -torch.log10(probs[j]))
            candidates.append(candidate)
        sequences = sorted(candidates, key=lambda t: t[1])[:k]

        for pred in sequences:
            predicted_index = pred[0][0]
            queue.append(indexed_tokens + [predicted_index])
        i += 1

    for pred in sequences:
        results.append(tokenizer.decode(indexed_tokens + [pred[0][0]]))

    return results

