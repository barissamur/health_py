from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv(r".\medquad.csv")
# TF-IDF Vektörleştirici
vectorizer = TfidfVectorizer()
# TF-IDF matrisini yeniden oluşturma
tfidf_matrix = vectorizer.fit_transform(df["question"])


def find_closest_question(input_question):
    input_tfidf = vectorizer.transform([input_question])
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix).flatten()
    closest = cosine_similarities.argsort()[-1]
    return df.iloc[closest]


# Örnek bir soru ile modeli test etme
test_question = " Blood  ?"
closest_question = find_closest_question(test_question)
print("Soru:", closest_question["question"])
print("Cevap:", closest_question["answer"])
