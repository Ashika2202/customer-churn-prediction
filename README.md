# Customer Churn Prediction Using Machine Learning

## Project Overview

Customer churn is one of the most important challenges faced by businesses. This project uses Machine Learning techniques to predict whether a customer is likely to leave a company based on factors such as age, monthly charges, and tenure.

The objective of this project is to help organizations identify customers at risk of churning and take proactive measures to improve customer retention and business growth.

---

## Problem Statement

Retaining existing customers is often more cost-effective than acquiring new ones. Businesses need predictive analytics solutions to identify customers who are likely to discontinue their services.

This project analyzes customer data and builds a Machine Learning model to predict customer churn.

---

## Objectives

- Analyze customer-related data.
- Perform data preprocessing and cleaning.
- Explore customer behavior patterns.
- Build a churn prediction model.
- Evaluate model performance.
- Generate insights for customer retention strategies.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Algorithm

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── churn_data.csv
│
├── output/
│   ├── confusion_matrix.png
│   └── prediction_results.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset Features

| Feature | Description |
|----------|-------------|
| Age | Customer age |
| MonthlyCharges | Monthly subscription charges |
| Tenure | Number of months as a customer |
| Churn | Target variable (0 = No Churn, 1 = Churn) |

---

## Exploratory Data Analysis (EDA)

The project includes:

- Dataset inspection
- Statistical summary
- Missing value analysis
- Customer behavior analysis
- Data visualization

---

## Model Training Process

1. Load customer dataset
2. Perform data preprocessing
3. Split data into training and testing sets
4. Train Random Forest Classifier
5. Predict customer churn
6. Evaluate model performance
7. Visualize results using a confusion matrix

---

## Evaluation Metrics

The model performance is measured using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
```

Navigate to the project directory:

```bash
cd customer-churn-prediction
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the following command:

```bash
python app.py
```

---

## Output

The project generates:

### 1. Prediction Results

```text
output/prediction_results.csv
```

Contains actual and predicted churn values.

### 2. Confusion Matrix Visualization

```text
output/confusion_matrix.png
```

Displays model prediction performance visually.

---

## Sample Business Insights

- Customers with higher monthly charges may have a greater risk of churn.
- Customer tenure can significantly influence retention.
- Predictive analytics can help businesses target at-risk customers with personalized offers.

---

## Future Enhancements

- Streamlit Interactive Dashboard
- Power BI Integration
- Hyperparameter Optimization
- Feature Engineering
- Deployment on Cloud Platforms
- Real-Time Customer Churn Monitoring

---

## Learning Outcomes

Through this project, the following skills were developed:

- Data Analysis
- Data Visualization
- Machine Learning Classification
- Model Evaluation
- Business Analytics
- Customer Retention Analysis

---

## Author

Ashika N

Data Analytics Internship Project

CodeTech IT Solutions

---

## License

This project is created for educational and internship purposes.
