import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, max_features: int = 3000, ngram_range: Tuple[int, int] = (1, 1)):
        nltk.download('stopwords', quiet=True)
        self.stopwords = list(set(stopwords.words('romanian') + stopwords.words('english')))
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words=self.stopwords, ngram_range=ngram_range)
    
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'\$NE\$', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def prepare_data(self, df: pd.DataFrame, column: str = 'sample') -> pd.DataFrame:
        df['cleaned_sample'] = df[column].apply(self.clean_text)
        return df

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        
    def train_dialect_model(self, X_train, y_train):
        model = LogisticRegression(C=10, max_iter=1000, solver='liblinear', random_state=self.random_state)
        model.fit(X_train, y_train)
        return model
    
    def train_theme_model(self, X_train, y_train, num_classes: int) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes, n_estimators=100, max_depth=10, learning_rate=0.1, gamma=0, min_child_weight=1, random_state=self.random_state)
        model.fit(X_train, y_train)
        return model

def main():
    preprocessor = TextPreprocessor()
    trainer = ModelTrainer()
    
    train_data = pd.read_csv('train_data.csv')
    train_data = preprocessor.prepare_data(train_data)
    
    X = preprocessor.tfidf.fit_transform(train_data['cleaned_sample'])
    
    X_train_dialect, X_test_dialect, y_train_dialect, y_test_dialect = train_test_split(X, train_data['dialect'], test_size=0.2, random_state=42)
    dialect_model = trainer.train_dialect_model(X_train_dialect, y_train_dialect)
    
    y_pred_dialect = dialect_model.predict(X_test_dialect)
    print(classification_report(y_test_dialect, y_pred_dialect))
    
    X_train_theme, X_test_theme, y_train_theme, y_test_theme = train_test_split(X, train_data['category'], test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train_theme_resampled, y_train_theme_resampled = smote.fit_resample(X_train_theme, y_train_theme)
    y_train_theme_encoded = trainer.label_encoder.fit_transform(y_train_theme_resampled)
    y_test_theme_encoded = trainer.label_encoder.transform(y_test_theme)
    
    theme_model = trainer.train_theme_model(X_train_theme_resampled, y_train_theme_encoded, len(trainer.label_encoder.classes_))
    
    y_pred_theme_encoded = theme_model.predict(X_test_theme)
    y_pred_theme = trainer.label_encoder.inverse_transform(y_pred_theme_encoded)
    print(classification_report(y_test_theme, y_pred_theme, zero_division=0))
    
    test_data = pd.read_csv('test_data.csv')
    test_data = preprocessor.prepare_data(test_data)
    X_test = preprocessor.tfidf.transform(test_data['cleaned_sample'])
    
    test_data['predicted_dialect'] = dialect_model.predict(X_test)
    test_data['predicted_theme'] = trainer.label_encoder.inverse_transform(theme_model.predict(X_test))
    
    test_data[['datapointID', 'predicted_dialect', 'predicted_theme']].to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()