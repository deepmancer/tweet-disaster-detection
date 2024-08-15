# 🌩️ Tweet Disaster Detection

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter%20Notebook-orange.svg)
![NLP](https://img.shields.io/badge/NLP-BERT%20%26%20Naive%20Bayes-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-red.svg)
![Huggingface](https://img.shields.io/badge/Huggingface-transformers-yellow.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange.svg)

## 📘 Introduction
This repository hosts the **Tweet Disaster Detection** system, an advanced NLP solution designed to identify disaster-related tweets in real-time. With the explosion of social media usage, rapidly detecting potential disaster events through user-generated content is crucial for timely interventions and responses.

## 🌟 Features

- **Real-Time Disaster Detection**: The system is capable of analyzing tweets in real-time to detect potential disasters as they unfold.
- **State-of-the-Art NLP Models**: Utilizes cutting-edge NLP models, including a fine-tuned BERT model, for high accuracy in disaster tweet classification.
- **Customizable and Extendable**: Built with flexibility in mind, the codebase allows for easy extension and customization to suit specific needs or datasets.
- **Visualization Tools**: Includes tools for visualizing model performance, such as learning curves and confusion matrices, providing insights into model behavior.

## 💼 Libraries and Tools Used

The project leverages several powerful libraries and tools, including:

- **[PyTorch](https://pytorch.org/)**: The primary deep learning library used for implementing and fine-tuning the BERT model. PyTorch’s dynamic computation graph and easy-to-use API make it ideal for such tasks.
  
- **[Huggingface Transformers](https://huggingface.co/transformers/)**: Provides pre-trained BERT models and utilities for tokenization, model fine-tuning, and other NLP tasks, allowing us to quickly implement state-of-the-art models.

- **[scikit-learn](https://scikit-learn.org/)**: Used for traditional machine learning tasks, including implementing the Naive Bayes model and performance evaluation metrics.

- **[Matplotlib](https://matplotlib.org/)**: Utilized for plotting learning curves, confusion matrices, and other visualizations that help in analyzing model performance.

- **[Pandas](https://pandas.pydata.org/)**: Facilitates data manipulation and analysis, making it easier to preprocess the tweet data and prepare it for model training.

## 💡 Project Overview
In the vast sea of tweets generated every second, our system stands out by efficiently distinguishing between tweets that indicate real disasters and those that don't. Leveraging cutting-edge machine learning algorithms and deep learning models, our approach ensures high precision and accuracy in disaster detection.

### 🧠 Model Fine-Tuning and Training

Our primary model is a fine-tuned version of **BERT** (Bidirectional Encoder Representations from Transformers), a state-of-the-art transformer model originally developed by Google. BERT's ability to understand context and disambiguate meaning in text makes it particularly suited for this task.

#### Model Fine-Tuning Process:

1. **Preprocessing**:
   - Tweets are tokenized using BERT's tokenizer, converting the text into a format that BERT can process (token IDs, attention masks, and segment IDs).
   
2. **Model Architecture**:
   - The BERT model is fine-tuned with an additional dense layer to classify tweets as either disaster-related or not. The architecture captures the complex semantics of tweets, ensuring robust classification performance.

   ```python
   input_word_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_word_ids')
   input_mask = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='input_mask')
   segment_ids = Input(shape=(self.max_seq_length,), dtype=tf.int32, name='segment_ids')

   pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])
   clf_output = sequence_output[:, 0, :]
   out = Dense(1, activation='sigmoid')(clf_output)
   model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
   ```

3. **Training Strategy**:
   - The model is trained using the SGD optimizer with a learning rate of `0.0001` and momentum of `0.8`, ensuring convergence and stability during the fine-tuning process. Multiple epochs are run, and key metrics like accuracy, precision, recall, and F1-score are tracked to monitor performance.

### 🚀 Results

| Model  | Precision | Recall | Accuracy | F1-Score |
|--------|:---------:|:------:|:--------:|:--------:|
| **BERT** | ![Green](https://via.placeholder.com/15/008000?text=+) `86%` | ![Green](https://via.placeholder.com/15/008000?text=+) `84%` | ![Green](https://via.placeholder.com/15/008000?text=+) `85%` | ![Green](https://via.placeholder.com/15/008000?text=+) `86%` |
| **Naive Bayes** | ![Red](https://via.placeholder.com/15/f03c15?text=+) `82%` | ![Red](https://via.placeholder.com/15/f03c15?text=+) `70%` | ![Red](https://via.placeholder.com/15/f03c15?text=+) `56%` | ![Red](https://via.placeholder.com/15/f03c15?text=+) `75%` |

### 📊 Visualizations and Performance Metrics

Throughout the training process, several visualizations were generated:

- **Learning Curves**: These illustrate the model's accuracy, precision, recall, and F1-score across epochs, offering insights into its learning behavior.
- **Confusion Matrix**: A detailed confusion matrix for the BERT model highlights its performance in correctly classifying disaster and non-disaster tweets.

### 🌍 Use Cases

Our model has several real-world applications that can make a significant impact:

- **Preventing Accidents**: By identifying tweets that signal real disasters, our system can alert first responders and relevant authorities, potentially preventing accidents or minimizing damage.
  
- **Early Warning Systems**: The model can provide early warnings of disasters, giving people time to prepare or evacuate to safety.

- **Accurate Disaster Reporting**: By filtering out false or irrelevant tweets, our system can improve the accuracy of disaster reporting, ensuring that people receive trustworthy information during crises.

## 🎯 Conclusion

The **Tweet Disaster Detection** system demonstrates the powerful application of modern NLP techniques in critical real-world scenarios. With its high accuracy and precision, especially using the fine-tuned BERT model, this project shows great potential in contributing to disaster management and response strategies globally.

We are committed to further refining this system and exploring its applications across different domains to make the world a safer place.

## 🛠️ How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/deepmancer/tweet-disaster-detection.git
   cd tweet-disaster-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   - Open `Advanced_Data_Science_Capstone.ipynb` to explore the code and see the results.

4. **Predict Disaster Tweets**:
   - Use the trained models to predict new tweets by following the instructions in the notebook.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author
- **Your Name** - [@deepmancer](https://github.com/deepmancer)
