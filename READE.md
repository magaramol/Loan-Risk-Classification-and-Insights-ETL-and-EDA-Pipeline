
# Loan Risk Classification and Insights: ETL and EDA Pipeline

## Overview
This project demonstrates an end-to-end ETL (Extract, Transform, Load) and Exploratory Data Analysis (EDA) workflow on loan data. The primary goal is to classify loan risk levels, segment tenure groups, recommend communication channels, and perform feature engineering for enhanced insights into borrower behavior.

## Table of Contents
1. [Technologies Used](#technologies-used)
2. [Dataset](#dataset)
3. [ETL Process](#etl-process)
4. [Feature Engineering](#feature-engineering)
5. [Data Analysis and Visualization](#data-analysis-and-visualization)
6. [Database Integration](#database-integration)
7. [Instructions](#instructions)
8. [Code](#code)

## Technologies Used
- **Python**: pandas, matplotlib, seaborn, mysql.connector, SQLAlchemy
- **MySQL**: for database storage
- **Jupyter Notebook**: for development

## Dataset
The dataset (`dpd_zero.csv`) contains borrower information, including:
- Bounce history over time
- Tenure in months
- Amount pending, disbursed amounts, interest rates
- City of the borrower

The dataset was preprocessed by:
- Removing spaces and converting column names to lowercase.

## ETL Process
1. **Extraction**: The data is extracted from a CSV file.
2. **Transformation**:
    - Cleaned and formatted the dataset.
    - Risk classification based on bounce history (last 6 months).
    - Tenure classification to identify early, mid, and late tenure stages.
    - Ticket size segmentation based on cumulative loan amounts.
    - Feature engineering for additional insights (e.g., Loan-to-Value ratio, Debt-to-Income ratio).
3. **Loading**: Data is stored into a MySQL database using SQLAlchemy.

## Feature Engineering
1. **Risk Classification**: Classify borrowers based on their bounce history over the last 6 months.
2. **Tenure Classification**: Categorize borrowers as 'early', 'mid', or 'late' based on tenure.
3. **Ticket Size Segmentation**: Group borrowers into low, medium, or high-ticket segments.
4. **Communication Channel Recommendation**: Suggest a channel (WhatsApp Bot, Voice Bot, or Human Call) based on risk, tenure, and interest rate.
5. Additional features include:
    - Total bounces, last bounce month, months since last payment
    - Loan-to-value ratio, debt-to-income ratio, urban vs. rural classification, and more.

## Data Analysis and Visualization
The project uses Seaborn and Matplotlib to visualize:
1. **Risk Label Distribution**: Displays the distribution of borrowers across risk levels.
2. **Ticket Size Distribution**: Shows the segmentation of borrowers into different ticket sizes.
3. **Tenure Label Distribution**: Visualizes borrowers by their tenure stages.

## Database Integration
Data is loaded into a MySQL database, with schema creation and data storage using `mysql.connector` and SQLAlchemy. The database integration follows these steps:
1. Create the database.
2. Load the transformed dataset into MySQL as a table.

## Instructions
1. Clone the repository and install the necessary packages:
   ```bash
   pip install mysql-connector-python pymysql pandas matplotlib seaborn sqlalchemy
   ```
2. Replace the placeholder values for the database URL, username, and password in the script.
3. Run the Python script to:
   - Extract and preprocess the dataset.
   - Perform feature engineering and risk classification.
   - Generate visualizations.
   - Load the data into a MySQL database.

## Code

```python
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

# Read dataset
df = pd.read_csv('/content/dpd_zero.csv')

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.lower()

# Risk classification function
def classify_risk(bounce_history):
    last_six_months = bounce_history[-6:]
    bounce_count = last_six_months.count('B') + last_six_months.count('L')

    if 'F' in last_six_months:
        return 'unknown_risk'
    elif bounce_count == 0:
        return 'low_risk'
    elif bounce_count <= 2 and last_six_months[-1] not in ['B', 'L']:
        return 'medium_risk'
    else:
        return 'high_risk'

# Apply risk classification
df['risk_label'] = df['bounce_string'].apply(classify_risk)

# Tenure classification function
def classify_tenure(tenure_months):
    if tenure_months == 3:
        return 'early_tenure'
    elif tenure_months >= 12:
        return 'late_tenure'
    else:
        return 'mid_tenure'

# Apply tenure classification
df['tenure_label'] = df['tenure'].apply(classify_tenure)

# Sort by 'Amount Pending' and segment ticket size
df = df.sort_values(by='amount_pending')
df['cumulative_sum'] = df['amount_pending'].cumsum()
total_sum = df['amount_pending'].sum()

df['ticket_size'] = pd.cut(df['cumulative_sum'], 
                           bins=3, 
                           labels=['low_ticket_size', 'medium_ticket_size', 'high_ticket_size'])

# Channel recommendation function
def recommend_channel(row):
    threshold_rate = 10
    if row['risk_label'] == 'low_risk' and row['bounce_string'][-1] not in ['B', 'L']:
        return 'whatsapp_bot'
    elif (row['risk_label'] in ['medium_risk', 'low_risk'] and
          row['interest_rate'] < threshold_rate and 
          row['tenure_label'] in ['low_emis', 'medium_emis']):
        return 'voice_bot'
    else:
        return 'human_call'

# Apply channel recommendation
df['spend_channel'] = df.apply(recommend_channel, axis=1)

# Visualize data
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='risk_label')
plt.title('Distribution of Borrowers by Risk Label')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='ticket_size')
plt.title('Distribution of Borrowers by Ticket Size')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='tenure_label')
plt.title('Distribution of Borrowers by Tenure')
plt.show()

# Additional feature engineering
df['total_bounces'] = df['bounce_string'].apply(lambda x: x.count('B') + x.count('L'))

def last_bounce_month(bounce_string):
    bounce_positions = [i for i, char in enumerate(bounce_string) if char in ['B', 'L']]
    return len(bounce_string) - 1 if not bounce_positions else bounce_positions[-1]

df['last_bounce_month'] = df['bounce_string'].apply(last_bounce_month)
df['months_since_last_payment'] = df['bounce_string'].apply(lambda x: len(x) - last_bounce_month(x) - 1)

market_value = 200000
df['ltv_ratio'] = df['disbursed_amount'] / market_value

income_data = [50000, 60000, 70000, 80000, 90000]
if len(income_data) == len(df):
    df['income'] = income_data
    df['dti_ratio'] = df['disbursed_amount'] / df['income']
else:
    df['dti_ratio'] = None

urban_cities = ['CityA', 'CityB', 'CityC']
df['urban_rural'] = df['city'].apply(lambda x: 'urban' if x in urban_cities else 'rural')

df['payment_term_type'] = df['interest_rate'].apply(lambda x: 'fixed' if x < 10 else 'variable')

previous_loan_counts = [1, 2, 0, 3, 1]
if len(previous_loan_counts) == len(df):
    df['previous_loan_count'] = previous_loan_counts
else:
    df['previous_loan_count'] = None

df['time_on_book'] = df['bounce_string'].apply(len)
df['risk_tenure'] = df['risk_label'] + '_' + df['tenure_label']
df['bounce_rate'] = df['total_bounces'] / df['time_on_book']

# Load data into MySQL
conn = mysql.connector.connect(host='DB_url',user='userName',password='password')
mycursor = conn.cursor()
mycursor.execute('CREATE DATABASE db_name')
conn.commit()

engine = create_engine("mysql+pymysql://userName:password@db_url/db_name")
df.to_sql('dpd', con=engine)
```
