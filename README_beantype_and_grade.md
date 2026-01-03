<<<<<<< HEAD
☕ Ceyloncoffee
AI-Powered Coffee Bean Type Identification and Quality Grading System

This project implements a deep learning–based system for automated coffee bean analysis using single images. The system is designed to support Sri Lankan coffee production and export processes by identifying bean type and evaluating bean quality through image-based classification.

The system consists of two independent models developed using the PyTorch framework with a MobileNetV3-Large architecture and 512×512 input resolution.

Model 1 performs coffee bean type classification and identifies whether a given bean belongs to the Arabica or Robusta category. This model achieved a validation accuracy of approximately 78.7%. Due to visual similarity between bean types, confidence-based thresholds are applied during inference to avoid unreliable predictions.

Model 2 performs coffee bean quality assessment and classifies beans into Good, Broken, or Severe Defect categories. This model achieved a higher validation accuracy of approximately 95.0%. Confusion matrix analysis shows strong performance across all defect classes, with minor misclassification occurring only in visually ambiguous cases.

A validation accuracy comparison graph is included to clearly illustrate the performance difference between the two models.

During inference, the outputs of both models are combined to produce a single result per image, including bean type, quality category, and confidence scores. If confidence thresholds are not satisfied, the system outputs an UNKNOWN result to prevent incorrect classification.

The modular design of the system allows future extension to multi-bean detection, conveyor-belt inspection, and additional defect categories.
