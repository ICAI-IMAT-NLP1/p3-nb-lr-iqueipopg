import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1]
        self.conditional_probabilities = self.estimate_conditional_probabilities(
            features, labels, delta
        )
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        label_counts = Counter(labels.tolist())

        total_samples = labels.shape[0]

        class_priors = {
            label: torch.tensor(count / total_samples, dtype=torch.float32)
            for label, count in label_counts.items()
        }

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        num_classes = len(torch.unique(labels))
        vocab_size = features.shape[1]

        class_word_counts = {
            c: torch.zeros(vocab_size, dtype=torch.float32) for c in range(num_classes)
        }
        total_words_per_class = {c: 0 for c in range(num_classes)}

        for i in range(features.shape[0]):
            label = labels[i].item()
            class_word_counts[label] += features[i]
            total_words_per_class[label] += features[i].sum().item()

        # Aplicar suavizado de Laplace para calcular las probabilidades condicionales P(w|c)
        class_conditional_probs = {
            c: (class_word_counts[c] + delta)
            / (total_words_per_class[c] + delta * vocab_size)
            for c in range(num_classes)
        }

        return class_conditional_probs

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "El modelo debe ser entrenado antes de estimar las probabilidades posteriores de clase."
            )
        log_posteriors = torch.zeros(len(self.class_priors), dtype=torch.float32)

        for class_label in self.class_priors.keys():
            # Comenzamos con el log de la probabilidad a priori para la clase
            log_posterior = torch.log(self.class_priors[class_label])

            # Sumamos el logaritmo de las probabilidades condicionales para cada palabra
            for i, word_count in enumerate(feature):
                if word_count > 0:
                    log_posterior += torch.log(
                        self.conditional_probabilities[class_label][i]
                    )

            log_posteriors[int(class_label)] = log_posterior

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and obtain the class of maximum likelihood
        log_posteriors = self.estimate_class_posteriors(feature)

        pred = torch.argmax(log_posteriors).item()

        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        probs: torch.Tensor = torch.softmax(
            self.estimate_class_posteriors(feature), dim=0
        )
        return probs
