# 🖐️ Sign Language Recognition System

Sign language recognition is a crucial application of computer vision and deep learning that enables individuals with speech and hearing impairments to communicate effectively. This project aims to develop a system that captures hand gestures, translates them into text, and converts the recognized text into speech using Natural Language processing. 

Built using **TensorFlow, Keras, and OpenCV**, this project focuses on accessibility applications for people with hearing and speech impairments.  

ASL(American Sign Language) is used train and test the dataset

---

## 🚀 Features
- Recognizes sign language hand gestures using **CNN-based deep learning**  
- Converts recognized gestures into **text**  
- Translates text into **speech** for accessibility  
- Achieved **95.37% validation accuracy** and **86.5% training accuracy** with low loss values  

---

## 📂 Project Structure
SignLanguageRecognition/
│── data/ # Dataset folder (not uploaded to GitHub)
│── models/ # Pre-trained models (link provided below)
│── src/ # Source code (Python scripts)
│── requirements.txt # Dependencies
│── README.md # Project documentation

## 📊 Results
- **Training Accuracy:** 86.5%  
- **Validation Accuracy:** 95.37%  
- **Training Loss:** 0.3977  
- **Validation Loss:** 0.1682  
- The model generalizes well with high accuracy and low loss on unseen data.  

---

## 📥 Dataset
The dataset I used was made by me which  is not uploaded to GitHub due to size limits. 
Download a similar dataset from: [asl_image_recognition_deep_learning (https://www.kaggle.com/code/cardocodes/asl-image-recognition-deep-learning/input)))  

After downloading, place the dataset inside the `data/` folder:  
SignLanguageRecognition/data/

---

## 🧠 Pre-trained Model
The trained Keras model (`keras_model10.h5`) can be downloaded from:  
[Google Drive Link](https://drive.google.com/file/d/140POw0YebtU95O9gahoEklJ3wnTQTf32/view?usp=sharing)  

Save it inside the `models/` folder:  
SignLanguageRecognition/models/

css
Copy code

If you want to train from scratch:  
```bash
python train.py
Installation
Clone the repository
git clone https://github.com/YourUsername/SignLanguageRecognition.git
cd SignLanguageRecognition
Install dependencies
pip install -r requirements.txt
Run the project
python main.py
