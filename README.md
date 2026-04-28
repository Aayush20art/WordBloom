# 🧠 WordBloom — Next Word Predictor

> **An industry-level LSTM-powered next-word prediction engine built with TensorFlow & Streamlit**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-WordBloom-00e5ff?style=for-the-badge)](https://wordbloom-awjzjr59o2yctvhuyzdeio.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

**WordBloom** is a deep learning web application that predicts the next most probable word(s) given any input sequence. Trained on a curated quotes dataset using a Long Short-Term Memory (LSTM) neural network, the model learns linguistic patterns and context to generate meaningful word predictions and complete sentences.

The app features a fully animated, dark-themed UI with infinite CSS animations, real-time predictions, and a text generation engine — all wrapped in a production-grade Streamlit interface.

---

## 🚀 Live Demo

👉 **[https://wordbloom-awjzjr59o2yctvhuyzdeio.streamlit.app/](https://wordbloom-awjzjr59o2yctvhuyzdeio.streamlit.app/)**

No installation required — runs directly in your browser.

---

## ✨ Features

### 🔮 Prediction Engine
- Predicts **Top-K next words** with probability scores
- Animated progress bars showing relative confidence per word
- Staggered card animations for each prediction result
- Rank badges (🥇🥈🥉) for top candidates

### ✍️ Text Generation
- Auto-generates **multi-word continuations** from a seed phrase
- Weighted random sampling from top-3 predictions for variety
- Animated progress bar during generation
- Displays word count, character count, and new words generated

### 📊 Dataset Explorer
- Preview the training dataset in an interactive table
- **Keyword search** across all columns in real-time
- Dataset stats: total rows, columns, unique tokens

### 🎨 UI / Animations
- **Infinite floating particles** — 22 particles with random speed, color, size
- **Triple spinning rings** around the logo — each rotates at a different speed
- **Audio-wave loader** — animated bars during model load and prediction
- **Shimmer sweep** on every prediction card (infinite)
- **Breathing glow** on the generated text box (infinite pulse)
- **Animated gradient divider** — color flows left to right (infinite)
- Slide-in animations, hover lift effects, and cursor blink on generated text

---

## 🛠️ Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Language     | Python 3.9+                       |
| Deep Learning| TensorFlow / Keras (LSTM)         |
| Frontend     | Streamlit + Custom CSS            |
| NLP          | Keras Tokenizer, Pad Sequences    |
| Data         | Pandas, NumPy                     |
| Animations   | Pure CSS Keyframes (no JS needed) |
| Fonts        | Syne, Space Mono (Google Fonts)   |
| Deployment   | Streamlit Community Cloud         |

---

## 📁 Project Structure

```
WordBloom/
│
├── next_word_predictor.py      # Main Streamlit application
├── lstm_model.h5               # Trained LSTM model (Keras)
├── qoute_dataset.csv           # Training dataset (quotes)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/wordbloom.git
cd wordbloom
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Required Files

Make sure these two files are in the **same directory** as `next_word_predictor.py`:

```
├── qoute_dataset.csv       ← Your training dataset (must have a "quote" column)
└── lstm_model.h5           ← Your trained Keras LSTM model
```

### 5. Run the App

```bash
streamlit run next_word_predictor.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Usage

### Predict Tab
1. Type any partial sentence in the input box (e.g., `the secret of life is`)
2. Click **⚡ Predict**
3. Top-K next words appear as animated cards with confidence scores

### Generate Tab
1. Enter a starting phrase (e.g., `in the beginning there was`)
2. Click **✨ Generate**
3. The model auto-completes the sentence word by word

### Sidebar Controls
| Control           | Description                                      |
|-------------------|--------------------------------------------------|
| Dataset path      | Path to your `.csv` file                        |
| Model path        | Path to your `.h5` Keras model                  |
| Top K slider      | Number of next-word candidates to return (1–10) |
| Generate words    | Number of words to generate (5–50)              |

---

## 🧬 Model Architecture

```
Input Sequence (padded)
        │
        ▼
  Embedding Layer
        │
        ▼
  LSTM Layer(s)
        │
        ▼
  Dense Layer (ReLU)
        │
        ▼
  Dense Output (Softmax)
        │
        ▼
  Vocabulary Probability Distribution
```

- **Model type:** Sequential LSTM
- **Input:** Tokenized & padded word sequences
- **Output:** Softmax probability over entire vocabulary
- **Prediction:** `argmax` / `top-K` sampling from output probabilities
- **Generation:** Weighted random sampling from top-3 predictions to ensure variety

---

## 📂 Dataset

The model is trained on a **quotes dataset** (`qoute_dataset.csv`) containing a `quote` column with text data.

**Preprocessing pipeline:**
1. Lowercase all text
2. Remove non-alphabetic characters via regex
3. Tokenize using Keras `Tokenizer`
4. Build N-gram sequences from each sentence
5. Pad sequences to uniform length
6. One-hot encode target word labels

---

## 🔭 Future Improvements

- [ ] Add **Beam Search** decoding for higher-quality generation
- [ ] Support **GPT-2 / Transformer** model as an alternative backend
- [ ] Add **temperature control** slider for generation creativity
- [ ] Export generated text as `.txt` or `.pdf`
- [ ] Upload custom dataset directly in the UI
- [ ] Add **token-level attention visualization**
- [ ] REST API endpoint for programmatic access

---

## 👨‍💻 Author

**Aayush**
B.Tech Information Technology | Chandigarh Engineering College, Landran

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](www.linkedin.com/in/aayush-sharma-b108a93b0)
[![Live App](https://img.shields.io/badge/Live_App-WordBloom-00e5ff?style=flat-square)](https://wordbloom-awjzjr59o2yctvhuyzdeio.streamlit.app/)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ using TensorFlow · Streamlit · Python</sub>
</div>
