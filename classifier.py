"""
classifier.py

HOG + SVM symbol classifier.
Handles feature extraction, training, evaluation, and model persistence.
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt


class SymbolClassifier:
    """
    Classifies crochet symbol images using HOG features and an SVM.

    Usage — training:
        clf = SymbolClassifier()
        clf.train(images, labels)
        clf.save("model.joblib")

    Usage — inference:
        clf = SymbolClassifier.load("model.joblib")
        label = clf.predict(image)          # single image (H×W uint8)
        labels = clf.predict_batch(images)  # list of images
    """

    IMG_SIZE = (64, 64)

    # HOG parameters — tuned for small symbol crops
    HOG_PARAMS = dict(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=None,
    )

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.label_encoder = LabelEncoder()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize image to standard size."""
        import cv2 as cv
        return cv.resize(image, self.IMG_SIZE)

    def _extract_hog(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG feature vector from a single (already-resized) image."""
        features = hog(image, **self.HOG_PARAMS)
        return features

    def _extract_hog_with_viz(self, image: np.ndarray):
        """Extract HOG features and return the visualisation image as well."""
        from skimage import exposure
        features, hog_image = hog(image, visualize=True, **self.HOG_PARAMS)
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        return features, hog_image

    def _featurise(self, images: list[np.ndarray]) -> np.ndarray:
        """Preprocess and extract HOG features for a list of images."""
        return np.array([self._extract_hog(self._preprocess(img)) for img in images])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, images: list[np.ndarray], labels: list[str]) -> None:
        """
        Train the classifier on a list of grayscale images and string labels.

        Args:
            images: list of uint8 grayscale numpy arrays (any size — resized internally)
            labels: corresponding class name strings
        """
        X = self._featurise(images)
        y = self.label_encoder.fit_transform(labels)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                probability=True,
            )),
        ])
        self.pipeline.fit(X, y)
        print(f"Trained on {len(images)} samples | classes: {list(self.label_encoder.classes_)}")

    def cross_validate(
        self,
        images: list[np.ndarray],
        labels: list[str],
        n_splits: int = 5,
    ) -> np.ndarray:
        """
        Run stratified k-fold cross-validation and print results.
        Returns per-fold accuracy scores.
        """
        if self.pipeline is None:
            raise RuntimeError("Call train() before cross_validate()")

        X = self._featurise(images)
        y = self.label_encoder.transform(labels)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="accuracy")
        print(f"{n_splits}-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"Per-fold: {np.round(scores, 3)}")
        return scores

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> str:
        """Predict class label for a single grayscale image."""
        self._check_trained()
        features = self._extract_hog(self._preprocess(image)).reshape(1, -1)
        idx = self.pipeline.predict(features)[0]
        return self.label_encoder.classes_[idx]

    def predict_proba(self, image: np.ndarray) -> dict[str, float]:
        """Return class probabilities for a single image."""
        self._check_trained()
        features = self._extract_hog(self._preprocess(image)).reshape(1, -1)
        probs = self.pipeline.predict_proba(features)[0]
        return dict(zip(self.label_encoder.classes_, probs))

    def predict_batch(self, images: list[np.ndarray]) -> list[str]:
        """Predict class labels for a list of images."""
        self._check_trained()
        X = self._featurise(images)
        indices = self.pipeline.predict(X)
        return [self.label_encoder.classes_[i] for i in indices]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        images: list[np.ndarray],
        labels: list[str],
        plot: bool = True,
    ) -> dict:
        """
        Evaluate on a labeled set and print classification report + confusion matrix.

        Returns:
            dict with 'report' (str) and 'accuracy' (float)
        """
        self._check_trained()
        X = self._featurise(images)
        y_true = self.label_encoder.transform(labels)
        y_pred = self.pipeline.predict(X)

        report = classification_report(y_true, y_pred, target_names=self.label_encoder.classes_)
        print(report)

        if plot:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred,
                display_labels=self.label_encoder.classes_,
                ax=ax,
                colorbar=False,
            )
            plt.title("Confusion matrix")
            plt.tight_layout()
            plt.show()

        accuracy = (y_true == y_pred).mean()
        return {"report": report, "accuracy": accuracy}

    def show_misclassified(
        self,
        images: list[np.ndarray],
        labels: list[str],
        max_shown: int = 10,
    ) -> None:
        """Display images that were predicted incorrectly."""
        self._check_trained()
        import cv2 as cv

        X = self._featurise(images)
        y_true = self.label_encoder.transform(labels)
        y_pred = self.pipeline.predict(X)

        wrong = np.where(y_pred != y_true)[0]
        if len(wrong) == 0:
            print("No misclassifications!")
            return

        wrong = wrong[:max_shown]
        fig, axes = plt.subplots(1, len(wrong), figsize=(3 * len(wrong), 3))
        if len(wrong) == 1:
            axes = [axes]
        for ax, idx in zip(axes, wrong):
            ax.imshow(images[idx], cmap="gray")
            true_cls = self.label_encoder.classes_[y_true[idx]]
            pred_cls = self.label_encoder.classes_[y_pred[idx]]
            ax.set_title(f"true: {true_cls}\npred: {pred_cls}", fontsize=9)
            ax.axis("off")
        plt.suptitle(f"{len(wrong)} misclassified sample(s)")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save trained pipeline and label encoder to a .joblib file."""
        self._check_trained()
        joblib.dump({"pipeline": self.pipeline, "label_encoder": self.label_encoder}, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SymbolClassifier":
        """Load a previously saved classifier."""
        data = joblib.load(path)
        obj = cls()
        obj.pipeline = data["pipeline"]
        obj.label_encoder = data["label_encoder"]
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
