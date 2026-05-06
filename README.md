<<<<<<< HEAD
<<<<<<< HEAD
---
title: Garbage Classification-identifier
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).
=======
=======
>>>>>>> 6d3295f4c0eb5c0cac8bd3cd939185207e0bf822
# ♻️ Smart Waste Classifier: AI-Powered Sustainability

**Smart Waste Classifier** is a sustainability-focused artificial intelligence application designed to automate waste sorting. By leveraging state-of-the-art Computer Vision, the system accurately classifies waste into 12 distinct categories, providing immediate disposal guidance to support environmental sustainability and circular economy initiatives.

---

## 🌟 Key Features

- **Advanced Computer Vision**: Utilizes a pre-trained **ResNet-18** deep learning architecture for high-accuracy image classification.
- **Interactive UI**: A professional, responsive dashboard built with **Streamlit** and enhanced with custom **Glassmorphism** CSS styling.
- **Confidence Visualization**: Real-time visualization of model confidence scores via interactive progress bars and distribution charts.
- **Smart Disposal Guide**: Context-aware disposal recommendations (Recyclable, Compostable, Hazardous, or Landfill) for every classification.
- **Modular Design**: Engineered with a decoupled architecture, separating core inference logic from the frontend presentation layer.

---

## 🛠️ Tech Stack

- **Core Logic**: Python 3.x
- **Deep Learning**: PyTorch, Torchvision
- **Frontend**: Streamlit
- **Data Handling**: Pandas, NumPy
- **Image Processing**: PIL (Pillow)
- **Styling**: Custom CSS (Glassmorphism)

---

## 🏗️ Project Architecture

The project is built on **Modular Software Engineering principles**. The application logic is decoupled into distinct modules:
- `app.py`: Handles the Streamlit UI, state management, and custom styling.
- `model/`: Contains the pre-trained weights and model state.
- `Inference Pipeline`: Integrated within the application to handle real-time image transformation and ResNet-18 forward passes.

This decoupling ensures that the inference engine can be scaled or updated independently of the user interface.

---

## 📸 Screenshots

### 🏠 Dashboard Overview
> *[Placeholder: Insert Screenshot of the main Upload screen here]*

### 🧪 Analysis & Results
> *[Placeholder: Insert Screenshot of the classification result and confidence breakdown here]*

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-waste-classifier.git
cd smart-waste-classifier
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```
*Note: If `streamlit` is not in your requirements file, install it via `pip install streamlit`.*

### 3. Launch the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure
```text
├── app.py              # Main Streamlit Application
├── model/              # Model weights (best.pth)
├── model_predict.py    # Inference utility functions
├── main.py             # Model training script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 👨‍💻 Author & Project Status

Developed by **Ishan Singh Tomar** as part of a **Final-Year CSIT Portfolio Project**.

This project is currently maintained as a professional demonstration of integrating Deep Learning with modern Web Frameworks to solve real-world environmental challenges.

---
*CSIT Final Year | Sustainability through Technology*
<<<<<<< HEAD
>>>>>>> 6d3295f (feat: initial release of Smart Waste Classifier with Streamlit UI)
=======
>>>>>>> 6d3295f4c0eb5c0cac8bd3cd939185207e0bf822
