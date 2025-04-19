import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Load raw dataset
def load_data(path='swiggy_data.csv'):
    return pd.read_csv(path)

# Clean and preprocess
def preprocess_data(df):
    # Drop unwanted columns
    df = df.drop(columns=['lic_no', 'menu'], errors='ignore')

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df = df.dropna(subset=['name', 'city', 'cuisine', 'rating', 'cost'])  # essential fields

    # Fill address if missing (optional)
    df['address'] = df['address'].fillna('Unknown')

    # Reset index after cleaning
    df.reset_index(drop=True, inplace=True)

    return df

# One-hot encode categorical & scale numerical features
def transform_data(df):
    categorical_cols = ['city', 'cuisine']
    numerical_cols = ['rating', 'rating_count', 'cost']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ], remainder='drop')

    # Fit and transform
    X = preprocessor.fit_transform(df)

    # Save transformer
    joblib.dump(preprocessor, 'preprocessor.joblib')

    return X, df

# Complete flow
def main():
    df = load_data()
    df_clean = preprocess_data(df)
    X, df_final = transform_data(df_clean)
    df_final.to_csv('preprocessed_swiggy_data.csv', index=False)
    print("✅ Preprocessing completed and saved.")

if __name__ == "__main__":
    main()
