# Tweet Disaster Detection

## Introduction
This repository houses the **Disaster Detection System**, a state-of-the-art NLP solution designed to identify disaster-related tweets. With the proliferation of social media, quick and accurate detection of disaster events from user-generated content becomes paramount for timely interventions and response.

## Description
Given the vast amount of tweets generated every second, our system aims to efficiently distinguish between tweets that indicate a real disaster and those that don't. Our solution leverages sophisticated algorithms and methodologies to achieve high precision and accuracy.

## Models Utilized

1. **BERT (Bidirectional Encoder Representations from Transformers)**: 
   - A cutting-edge deep learning model tailored for understanding context within text. BERT's bidirectional training approach and vast pre-training on large text corpora enable it to capture nuanced linguistic patterns, making it apt for our task.
2. **Naive Bayes (NB)**: 
   - A probabilistic algorithm renowned for its simplicity and efficiency in text classification tasks. It serves as a benchmark model to validate the superiority of deep learning models like BERT for the task.

## Results

| Model | Precision | Recall | Accuracy | F1-Score |
|-------|:---------:|:------:|:--------:|:-------:|
| **BERT**  | ![Green](https://via.placeholder.com/15/008000?text=+) `86%` | ![Green](https://via.placeholder.com/15/008000?text=+) `84%` | ![Green](https://via.placeholder.com/15/008000?text=+) `85%` | ![Green](https://via.placeholder.com/15/008000?text=+) `86%` |
| **NB**    | ![Red](https://via.placeholder.com/15/f03c15?text=+) `82%` | ![Red](https://via.placeholder.com/15/f03c15?text=+) `70%` | ![Red](https://via.placeholder.com/15/f03c15?text=+) `56%` | ![Red](https://via.placeholder.com/15/f03c15?text=+) `75%` |

## Conclusion
The comparative results clearly highlight the superior performance of the BERT model over traditional machine learning models like Naive Bayes. The application of such advanced algorithms, paired with meticulous feature engineering and quality datasets, ensures our system remains on the cutting edge of disaster prediction capabilities.
