# salary-prediction-
# 💼 Salary Prediction using TabNet Classifier

This project aims to predict whether a person earns more than or less than ₹50K per year based on various features such as age, education, occupation, and working hours.  
The main goal is to build a smart and accurate classifier using **TabNet**, a deep learning model specialized for tabular data.

---

## 📂 Dataset

You can use any income classification dataset with relevant features.  
In this project, I used the **Adult Income Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult), which includes:
- Age
- Education
- Occupation
- Hours-per-week
- Gender, Race
- Capital gain/loss
- Workclass, Relationship, etc.

---

## ⚙️ Technologies Used

- **Python 3**
- **Google Colab** (used for implementation)
- **Libraries**:
  - `pandas` – for data handling
  - `scikit-learn` – for preprocessing and evaluation
  - `pytorch-tabnet` – TabNetClassifier model
  - `matplotlib` – for plotting accuracy graphs
  - `numpy` – for numerical operations

---

## 🧠 ML Model: TabNetClassifier

- TabNet is a deep learning model that performs well on structured (tabular) data.
- It uses attention to select useful features during training.
- Compared with:
  - **Logistic Regression**
  - **Random Forest**
- Achieved over **85% accuracy** on classification task.

---

## 📈 Results

- ✅ Accuracy (TabNet): ~85.37%
- ✅ Confusion Matrix: Included in output
- ✅ Accuracy Comparison Graph: `accuracy_plot.png`

---

## 🚀 How to Run the Project

### 🔧 Step 1: Clone or download the repo

```bash
git clone https://github.com/YuvaDurga-tech/salary-prediction-tabnet.git
cd salary-prediction-tabnet
```

### 📁 Step 2: Place the dataset

Place the dataset file named `adult 3.csv` in the root folder.  
(You can rename the UCI Adult dataset to this if needed)

### 💻 Step 3: Install required libraries

If using **Google Colab**, run:

```python
!pip install pytorch-tabnet pandas scikit-learn matplotlib
```

If using **local Jupyter Notebook or terminal**, run:

```bash
pip install pytorch-tabnet pandas scikit-learn matplotlib
```

### 🏃 Step 4: Run the script

```bash
python tabnet_income_classifier.py
```

You’ll get:
- Accuracy printed in terminal
- Confusion matrix of predictions
- Bar graph of accuracy saved as `accuracy_plot.png`

---

## 📝 Author

**Yuva Durga**  
AI Capstone Project – CSE-CyberSecurity Department

---

## 📚 References

- UCI Adult Dataset: [https://archive.ics.uci.edu/ml/datasets/adult](https://archive.ics.uci.edu/ml/datasets/adult)  
- PyTorch TabNet GitHub  
- scikit-learn documentation  
- Kaggle notebooks on income prediction
