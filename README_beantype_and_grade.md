Coffee Bean Type Identification and Defect Grading Module

This module implements an image-based deep learning solution for automated coffee bean analysis. It focuses on identifying coffee bean type and evaluating bean quality from single images captured under real-world conditions.

Two convolutional neural network models were developed using the PyTorch framework with a MobileNetV3-Large architecture and an input resolution of 512×512 pixels.

The first model performs coffee bean type classification and predicts whether a given bean belongs to the Arabica or Robusta category. This model achieved a validation accuracy of approximately 78.7%. Due to visual similarity between bean types, confidence-based thresholds are applied during inference to avoid unreliable predictions.

The second model performs coffee bean defect classification and categorizes beans as Good, Broken, or Severe Defect. This model achieved a validation accuracy of approximately 95.0%. Confusion matrix analysis indicates strong performance across all classes, with minor misclassification occurring only in visually ambiguous cases.

During inference, both models are executed on the same input image and their outputs are combined to produce a single result consisting of bean type, quality category, and confidence scores. If confidence thresholds are not satisfied, the system outputs an UNKNOWN label to prevent incorrect classification.

This module is designed to be lightweight, modular, and suitable for future extension to conveyor-belt inspection, multi-bean detection, and additional defect categories.
