import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
# filepath: c:\Users\legio\Documents\GitHub\nlp_project\train_data.csv
data = pd.read_csv('train_data.csv')

# Preprocess the data
data['cleaned_sample'] = data['sample'].str.replace(r'\$NE\$', '', regex=True).str.lower()

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['cleaned_sample'])

# Split data for dialect classification
X_train_dialect, X_test_dialect, y_train_dialect, y_test_dialect = train_test_split(
    X, data['dialect'], test_size=0.2, random_state=42
)

# Train a classifier for dialect
dialect_model = LogisticRegression()
dialect_model.fit(X_train_dialect, y_train_dialect)

# Evaluate dialect classifier
y_pred_dialect = dialect_model.predict(X_test_dialect)
print("Dialect Classification Report:")
print(classification_report(y_test_dialect, y_pred_dialect))

# Split data for theme classification
X_train_theme, X_test_theme, y_train_theme, y_test_theme = train_test_split(
    X, data['category'], test_size=0.2, random_state=42
)

# Train a classifier for theme
theme_model = LogisticRegression()
theme_model.fit(X_train_theme, y_train_theme)

# Evaluate theme classifier
y_pred_theme = theme_model.predict(X_test_theme)
print("Theme Classification Report:")
print(classification_report(y_test_theme, y_pred_theme))