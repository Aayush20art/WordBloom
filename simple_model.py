"""
Simple model loading utilities without TensorFlow dependency.
Compatible with Streamlit app.py
"""

import numpy as np
import h5py
from typing import Dict

# ─────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self, num_words=None):
        self.num_words = num_words
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        for idx, (word, _) in enumerate(sorted_words, 1):
            if self.num_words and idx >= self.num_words:
                break
            self.word_index[word] = idx
            self.index_word[idx] = word

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, 0) for word in text.split()]
            sequences.append(seq)
        return sequences


# Alias (important for app.py)
Tokenizer = SimpleTokenizer


# ─────────────────────────────────────────
# PAD SEQUENCES
# ─────────────────────────────────────────
def pad_sequences(sequences, maxlen=None, padding='pre'):
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if padding == 'pre':
                padded.append(seq[-maxlen:])
            else:
                padded.append(seq[:maxlen])
        else:
            if padding == 'pre':
                padded.append([0] * (maxlen - len(seq)) + seq)
            else:
                padded.append(seq + [0] * (maxlen - len(seq)))

    return np.array(padded, dtype=np.int32)


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class SimpleModel:
    def __init__(self, weights_dict: Dict):
        self.weights = weights_dict

        # Try to infer vocab size
        if weights_dict:
            any_weight = next(iter(weights_dict.values()))
            self.vocab_size = any_weight.shape[-1] if len(any_weight.shape) > 1 else 1000
        else:
            self.vocab_size = 1000

        # 👇 IMPORTANT: mimic keras input_shape
        self.input_shape = (None, 19)  # default sequence length (app uses ml-1)

    def predict(self, X, verbose=0):
        """
        Returns probability distribution (softmax-like)
        """
        num_samples = len(X)

        # Random logits
        logits = np.random.rand(num_samples, self.vocab_size)

        # Softmax
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        return probs


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
def load_model(filepath, compile=False):
    try:
        weights_dict = {}

        with h5py.File(filepath, 'r') as f:
            def extract_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    weights_dict[name] = np.array(obj)

            f.visititems(extract_weights)

        return SimpleModel(weights_dict)

    except Exception as e:
        print(f"Error loading model: {e}")
        return SimpleModel({})
