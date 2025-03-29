import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

romanian_stopwords = stopwords.words('romanian')

data = pd.read_csv('train_data.csv')
data['cleaned_sample'] = data['sample'].str.replace(r'\$NE\$', '', regex=True).str.lower()
tfidf = TfidfVectorizer(max_features=10000, stop_words=romanian_stopwords, ngram_range=(1, 3))
X = tfidf.fit_transform(data['cleaned_sample'])

X_train_dialect, X_test_dialect, y_train_dialect, y_test_dialect = train_test_split(
    X, data['dialect'], test_size=0.2, random_state=42
)

dialect_model = LogisticRegression()
dialect_model.fit(X_train_dialect, y_train_dialect)
y_pred_dialect = dialect_model.predict(X_test_dialect)
print("Dialect Classification Report:")
print(classification_report(y_test_dialect, y_pred_dialect))

X_train_theme, X_test_theme, y_train_theme, y_test_theme = train_test_split(
    X, data['category'], test_size=0.2, random_state=42
)

class_weights = compute_class_weight('balanced', classes=data['category'].unique(), y=data['category'])
class_weight_dict = dict(zip(data['category'].unique(), class_weights))

theme_model = RandomForestClassifier(class_weight=class_weight_dict, random_state=42)
theme_model.fit(X_train_theme, y_train_theme)
y_pred_theme = theme_model.predict(X_test_theme)
print("Theme Classification Report:")
print(classification_report(y_test_theme, y_pred_theme, zero_division=0))

test_data = pd.read_csv('test_data.csv')
test_data['cleaned_sample'] = test_data['sample'].str.replace(r'\$NE\$', '', regex=True).str.lower()
X_test = tfidf.transform(test_data['cleaned_sample'])
test_data['predicted_dialect'] = dialect_model.predict(X_test)
test_data['predicted_theme'] = theme_model.predict(X_test)
output_filepath = 'predictions.csv'
test_data[['datapointID', 'predicted_dialect', 'predicted_theme']].to_csv(output_filepath, index=False)
print(f"Predictions saved to {output_filepath}")