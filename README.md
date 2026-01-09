# Loan Approval Data Engineering Project

This repository contains a data engineering project for **loan approval prediction** — including data preprocessing, exploratory analysis, a trained model, a web app interface, and automated tests.

---

## 🧠 Project Overview

A simple end-to-end pipeline that:

1. Loads and preprocesses loan applicant data
2. Performs exploratory data analysis
3. Uses a trained ML model to predict loan approval
4. Deploys an interactive web app for inference
5. Includes tests to ensure components function correctly

---

## 📂 Repository Structure

```

├── EDA_Train_pickle.py         # EDA and preprocessing
├── app.py                      # Web app to predict loan approval
├── applicant_info.json         # Example applicant personal info
├── financial_info.json         # Example applicant financial info
├── loan_info.json              # Example loan details
├── loan_approval_model.pkl     # Pre-trained model pickle
├── requirements.txt            # Python dependencies
├── test_load_data.py           # Unit tests for data loading
└── README.md                   # Project README

````

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/parshva106/Loan_Approval_Data_Engineering_Project.git
cd Loan_Approval_Data_Engineering_Project
````

### 2. Set up a Python environment

Create & activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Running Exploratory Analysis

The `EDA_Train_pickle.py` script can be used to:

* Explore training data
* Preprocess features
* Save a pickled model (if training is implemented)

Run:

```bash
python EDA_Train_pickle.py
```

---

## 📊 Launching the Web App

Use `app.py` to start a simple interface that loads JSON input files and outputs loan approval predictions.

```bash
python app.py
```

Once running, the app will:

* Read sample JSON files
* Load the trained model (`loan_approval_model.pkl`)
* Display results in the console or UI as configured

---

## 📄 Sample Input JSON

The repo includes sample files to test inference:

* `applicant_info.json` — applicant demographic data
* `financial_info.json` — applicant financial details
* `loan_info.json` — loan specific details

You can edit these to create custom inputs.

---

## 🧪 Testing

A basic unit test file is included to validate data loading and parsing:

```bash
pytest test_load_data.py
```

Make sure `pytest` is installed (via `requirements.txt`).

---

## 🧰 Dependencies

All required Python packages are listed in `requirements.txt`. Some expected dependencies are:

```
pandas
scikit-learn
flask (or FastAPI / streamlit depending on your app)
pytest
```

Install them via:

```bash
pip install -r requirements.txt
```

---

## 👍 Contributing

Contributions are welcome! Feel free to open an issue or submit pull requests.

---

## 📜 License

This project does not have a license specified.

---

## 🤝 Contact

Created by **parshva106**.
Feel free to reach out for questions or collaboration!



