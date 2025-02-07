from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file infile, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []

    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.rsplit("\t", 1)  # Divide en máximo 2 partes desde el final
            if len(parts) != 2:
                continue

            sentence, label = parts
            tokenized_sentence = tokenize(sentence)
            label = int(label)  # Convertir la etiqueta a entero (0 o 1)

            examples.append(SentimentExample(words=tokenized_sentence, label=label))

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}

    all_words = [word for example in examples for word in example.words]
    word_counts = Counter(all_words)

    for word in word_counts:
        vocab[word] = len(vocab)

    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab))

    for word in text:
        if word in vocab:
            idx = vocab[word]
            if binary:
                bow[idx] = 1
            else:
                bow[idx] += 1

    return bow
