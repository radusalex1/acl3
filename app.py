import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load sample data
data = {
    'text': [
        "Absolutely love it, exceeded my expectations.",
        "Terrible, broke after one use.",
        "It works fine, nothing extraordinary.",
        "Best product ever, worth every penny!",
        "Completely useless, don't waste your money.",
        "Great value for the price, highly satisfied.",
        "Not bad, but could be better.",
        "Awful experience, returning it immediately.",
        "Perfect for my needs, couldn't be happier.",
        "Regret this purchase, very disappointing.",
        "Impressive design and functionality, love it.",
        "Disaster! Do not recommend to anyone.",
        "Average quality, nothing to write home about.",
        "Superb performance, highly recommend!",
        "Horrible quality, fell apart quickly.",
        "Decent product, but overpriced.",
        "Exceptional product, exceeded my expectations.",
        "Would not recommend, not worth it.",
        "Solid choice, met my requirements.",
        "Avoid at all costs, waste of time.",
        "Fantastic! Will be buying again soon.",
        "Disappointed, does not match the description.",
        "Pretty good, but there are better options.",
        "Outstanding, exactly what I was looking for.",
        "Extremely poor quality, very let down.",
        "Met my expectations, decent purchase.",
        "Amazing features, totally worth it.",
        "Terrible fit, uncomfortable to use.",
        "Good product, but slow delivery.",
        "Wonderful experience, 5 stars from me!",
        "Worst item I've ever bought.",
        "So-so, not sure I'd recommend.",
        "Thrilled with this purchase, highly satisfied.",
        "Disgusting! Never buying from this brand again.",
        "Reasonable quality, good for the price.",
        "Incredible product, changed my life!",
        "Wouldn't buy again, not durable.",
        "Exactly as advertised, very happy.",
        "Waste of money, doesn't work as intended.",
        "Pretty good for the price, no complaints.",
        "Life-changing, can't recommend it enough!",
        "Disaster! Absolutely awful product.",
        "Functional, but lacks durability.",
        "Beyond satisfied, this is fantastic!",
        "Extremely disappointed, not as promised."
    ],
    'sentiment': [
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'positive', 'neutral', 'negative', 'positive', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'neutral', 'negative',
        'positive', 'negative', 'neutral', 'positive', 'negative',
        'neutral', 'positive', 'negative', 'neutral', 'positive',
        'negative', 'neutral', 'positive', 'negative', 'neutral',
        'positive', 'negative', 'positive', 'negative', 'neutral',
        'positive', 'negative', 'positive', 'negative',"negative"
    ]
}

df = pd.DataFrame(data)

# Preprocess text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train a Naive Bayes classifier
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Function to predict sentiment of new text
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    return model.predict(vectorized_text)[0]

# Test the function
new_text = "This is the best service I have ever used!"
print(f"Sentiment for '{new_text}':", predict_sentiment(new_text))
