# import pandas as pd
# import re
# import nltk
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# romanian_stopwords = stopwords.words('romanian')
# data = pd.read_csv('train_data.csv')
# data['cleaned_sample'] = data['sample'].str.replace(r'\$NE\$', '', regex=True)
# data['cleaned_sample'] = data['cleaned_sample'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
# tfidf = TfidfVectorizer(
#     max_features=10000,
#     stop_words='english',
#     ngram_range=(1, 3)
# )
# X = tfidf.fit_transform(data['cleaned_sample'])
# X_train_dialect, X_test_dialect, y_train_dialect, y_test_dialect = train_test_split(
#     X, data['dialect'], test_size=0.2, random_state=42
# )
# dialect_model = LogisticRegression(C=10, max_iter=1000, solver='liblinear')
# dialect_model.fit(X_train_dialect, y_train_dialect)
# y_pred_dialect = dialect_model.predict(X_test_dialect)
# print("Dialect Classification Report:")
# print(classification_report(y_test_dialect, y_pred_dialect))

# # Theme classification
# X_train_theme, X_test_theme, y_train_theme, y_test_theme = train_test_split(
#     X, data['category'], test_size=0.2, random_state=42
# )
# class_weights = compute_class_weight('balanced', classes=data['category'].unique(), y=data['category'])
# class_weight_dict = dict(zip(data['category'].unique(), class_weights))

# # Apply SMOTE for class balancing
# smote = SMOTE(random_state=42)
# X_train_theme_resampled, y_train_theme_resampled = smote.fit_resample(X_train_theme, y_train_theme)

# # Encode the labels to start from 0
# label_encoder = LabelEncoder()
# y_train_theme_resampled_encoded = label_encoder.fit_transform(y_train_theme_resampled)
# y_test_theme_encoded = label_encoder.transform(y_test_theme)

# # FIXED: Updated parameter grid with XGBoost-specific parameters
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'gamma': [0, 0.1, 0.2],  # Replaces min_samples_split
#     'min_child_weight': [1, 3, 5]  # Replaces min_samples_leaf
# }

# # Set up GridSearchCV with XGBoost
# grid_search = GridSearchCV(
#     xgb.XGBClassifier(
#         objective='multi:softmax',
#         num_class=len(label_encoder.classes_),
#         learning_rate=0.1,
#         random_state=42
#     ),
#     param_grid,
#     cv=5
# )

# # Use the encoded labels for training
# grid_search.fit(X_train_theme_resampled, y_train_theme_resampled_encoded)

# theme_model = grid_search.best_estimator_

# # Predict and decode the labels back to the original format
# y_pred_theme_encoded = theme_model.predict(X_test_theme)
# y_pred_theme = label_encoder.inverse_transform(y_pred_theme_encoded)

# print("Theme Classification Report:")
# print(classification_report(y_test_theme, y_pred_theme, zero_division=0))

# # Process test data
# test_data = pd.read_csv('test_data.csv')
# test_data['cleaned_sample'] = test_data['sample'].str.replace(r'\$NE\$', '', regex=True)
# test_data['cleaned_sample'] = test_data['cleaned_sample'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
# X_test = tfidf.transform(test_data['cleaned_sample'])

# test_data['predicted_dialect'] = dialect_model.predict(X_test)
# test_data['predicted_theme'] = label_encoder.inverse_transform(theme_model.predict(X_test))

# output_filepath = 'predictions.csv'
# test_data[['datapointID', 'predicted_dialect', 'predicted_theme']].to_csv(output_filepath, index=False)
# print(f"Predictions saved to {output_filepath}")