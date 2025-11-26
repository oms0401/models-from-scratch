"""
A full implementation of Byte Pair Encoding (BPE) Tokenizer.
Based on Andrej Karpathy's 'minbpe' and GPT-4 architecture.
"""

import regex as re


def get_stats(ids):
    """
    Given a list of integers, find the frequency of every adjacent pair.
    Input: [1, 2, 1, 2, 3]
    Output: {(1, 2): 2, (2, 3): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Replace all occurrences of 'pair' with the new token 'idx'.
    Input: ids=[1, 2, 1, 2, 3], pair=(1, 2), idx=256
    Output: [256, 256, 3]
    """
    newids = []
    i = 0
    while i < len(ids):

        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2 
        else:
            newids.append(ids[i])
    return newids



class BasicTokenizer:
    def __init__(self):
        self.merges = {} 
        self.vocab = {}  

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

          
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)

            self.merges[pair] = idx
            
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
            
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")



GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self):
        super().__init__()
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_chunks = self.pattern.findall(text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                chunk_stats = get_stats(chunk_ids)
                for pair, count in chunk_stats.items():
                    stats[pair] = stats.get(pair, 0) + count

            if not stats: break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            
            self.merges[pair] = idx
            
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx}")
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        text_chunks = self.pattern.findall(text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = merge(chunk_ids, pair, idx)
            ids.extend(chunk_ids)
            
        return ids


if __name__ == "__main__":
    
    text = "Hello world! This is a test of the Basic Tokenizer. Hello world again!"
    tokenizer = RegexTokenizer() 
    print("Training Tokenizer...")
    tokenizer.train(text, vocab_size=300, verbose=True)
    sample = "Hello world!"
    encoded = tokenizer.encode(sample)
    print(f"\nString: '{sample}'")
    print(f"Encoded IDs: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    assert sample == decoded
    print("\nSuccess! Decoded string matches original.")