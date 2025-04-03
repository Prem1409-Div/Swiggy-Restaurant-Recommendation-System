# Swiggy Restaurant Recommendation System

## Overview
This project is a restaurant recommendation system built using **Streamlit** and **Python**. It utilizes **data preprocessing**, **one-hot encoding**, and **clustering (K-Means, Cosine Similarity)** to suggest restaurants based on user preferences such as city, cuisine, rating, and cost.

## Features
- Data Cleaning and Preprocessing
- One-Hot Encoding of Categorical Features
- Clustering-Based Recommendation Engine
- Interactive Web App using Streamlit

## Tech Stack
- **Python** (Data Processing & ML Algorithms)
- **Pandas & NumPy** (Data Manipulation)
- **Scikit-Learn** (Clustering & Similarity)
- **Streamlit** (Web App UI)
- **GitHub** (Version Control)

## Project Structure
```
Swiggy-Recommendation-System/
│── data_preprocessing.py  # Data cleaning & encoding
│── recommendation.py       # Clustering & similarity computation
│── app.py                  # Streamlit app for recommendations
│── restaurant_data.csv      # Raw dataset file
│── cleaned_data.csv         # Cleaned dataset
│── encoded_data.csv         # One-hot encoded dataset
│── encoder.pkl              # Saved encoder model
│── clustered_data.csv       # Clustered data
│── README.md                # Project documentation
```

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Swiggy-Recommendation-System.git
cd Swiggy-Recommendation-System
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Data Preprocessing
```bash
python data_preprocessing.py
```

### 4. Train the Recommendation Model
```bash
python recommendation.py
```

### 5. Run the Streamlit App
```bash
streamlit run app.py
```

## Usage
- Select **City** and **Cuisine**
- Adjust **Rating** and **Cost** filters
- Click **Get Recommendations**
- View restaurant suggestions

## Contributing
Feel free to fork this repository and create a pull request with improvements!

## License
This project is licensed under the MIT License.

