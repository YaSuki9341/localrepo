from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pylint.lint

# Data Collection
code_samples = [
    {'code': 'def add(a, b): return a + b', 'label': 'good'},
    {'code': 'def subtract(a, b): return a - b', 'label': 'good'},
    {'code': 'x = 5; y = x + 3', 'label': 'improvement'},
    {'code': 'print("Hello, World!")', 'label': 'good'},
    {'code': 'for i in range(10): print(i)', 'label': 'bug'},
]

# Data Preprocessing
X = [sample['code'] for sample in code_samples]
y = [sample['label'] for sample in code_samples]

# Feature Engineering
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Model Training
model = RandomForestClassifier()
model.fit(X_vectorized, y)

# Code Review Assistant
def code_review(code):
    # Code Analysis
    pylint_output = pylint.lint.Run([code], do_exit=False)
    pylint_score = pylint_output.linter.stats['global_note']

    # Predictions
    code_vectorized = vectorizer.transform([code])
    prediction = model.predict(code_vectorized)[0]

    # Generate Suggestions
    suggestions = {'pylint_score': pylint_score, 'prediction': prediction}

    if prediction == 'improvement':
        suggestions['recommendation'] = 'Consider refactoring for better readability.'

    elif prediction == 'bug':
        suggestions['recommendation'] = 'Potential bug detected. Please review.'

    return suggestions

# Example usage:
code_to_review = 'def multiply(a, b): return a * b'
review_result = code_review(code_to_review)
print(review_result)
