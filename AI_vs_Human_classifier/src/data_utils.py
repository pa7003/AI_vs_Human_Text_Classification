import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    # Basic cleaning (customize as needed)
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str).str.strip()
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])
    return df, le

def split_data(df, test_size=0.2, random_state=42):
    return train_test_split(df['text'], df['label_enc'], test_size=test_size, random_state=random_state)
