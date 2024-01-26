import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random
import json


# Eğitim veri setini yükleme
train_df = pd.read_csv("Training.csv")

# Test veri setini yükleme
test_df = pd.read_csv("Testing.csv")


# 'Unnamed' gibi ekstra sütunları kaldırma
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

# Eğitim ve test setlerinden özellik ve hedefleri ayırma
X_train = train_df.drop("prognosis", axis=1)
y_train = train_df["prognosis"]
X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

# Modeli başlatma
model = DecisionTreeClassifier()

# Modeli eğitme
model.fit(X_train, y_train)

# Test veri setinde tahmin yapma
y_pred = model.predict(X_test)

# Modelin doğruluğunu değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print("Modelin doğruluğu: {:.2f}%".format(accuracy * 100))


# Çeviri dosyalarını yükleme
with open("belirtiler_ceviri.json", "r", encoding="utf-8") as f:
    belirtiler_ceviri = json.load(f)

with open("hastaliklar_ceviri.json", "r", encoding="utf-8") as f:
    hastalik_ceviri = json.load(f)


def translate_symptom(symptom):
    return belirtiler_ceviri.get(symptom, symptom)


def translate_disease(disease):
    return hastalik_ceviri.get(disease, disease)


def find_related_diseases(symptom):
    # Belirti ile ilişkili hastalıkları bulma
    related_diseases = train_df[train_df[symptom] == 1]["prognosis"].unique()
    return related_diseases


def get_related_symptoms(diseases, asked_symptoms):
    # Hastalıklarla ilişkili diğer belirtileri bulma
    related_symptoms = set()
    for disease in diseases:
        symptoms = train_df[train_df["prognosis"] == disease].iloc[0]
        for symptom in symptoms[symptoms == 1].index:
            if symptom not in asked_symptoms:
                related_symptoms.add(symptom)
    return list(related_symptoms)


def ask_symptoms(symptoms, asked_symptoms, threshold=10):
    total_score = 0

    while total_score < threshold:
        if not symptoms:  # Eğer sorulacak belirti kalmadıysa, yeni bir hastalık seç
            random_symptom = random.choice(list(set(X_train.columns) - asked_symptoms))
            related_diseases = find_related_diseases(random_symptom)
            symptoms = get_related_symptoms(related_diseases, asked_symptoms)

        symptom = symptoms.pop(0)
        asked_symptoms.add(symptom)  # Sorulan belirtiyi kaydet

        translated_symptom = translate_symptom(symptom)
        response = (
            input(f"{translated_symptom} belirtisine sahip misiniz? (Evet/Hayır): ")
            .strip()
            .lower()
        )

        if response == "evet":
            total_score += 1
            print(f"Şu anki puanınız: {total_score}")

    return True


# Rastgele bir belirti seçme ve sorma
# Rastgele bir belirti seçme ve sorma
random_symptom = random.choice(X_train.columns)
asked_symptoms = {random_symptom}  # Sorulan belirtileri takip etmek için bir set

related_diseases = find_related_diseases(random_symptom)
related_symptoms = get_related_symptoms(related_diseases, asked_symptoms)

translated_symptom = translate_symptom(random_symptom)
print(f"Sorulacak belirti: {translated_symptom}")

response = (
    input(f"{translated_symptom} belirtisine sahip misiniz? (Evet/Hayır): ")
    .strip()
    .lower()
)

# Kullanıcının cevabına göre ilerleme

if ask_symptoms(related_symptoms, asked_symptoms):
    user_data = {symptom: 0 for symptom in X_train.columns}
    for symptom in asked_symptoms:
        user_data[symptom] = 1
    user_data_df = pd.DataFrame([user_data])
    predicted_disease = model.predict(user_data_df)[0]
    translated_disease = translate_disease(predicted_disease)
    print(f"Modelin tahmin ettiği hastalık: {translated_disease}")
else:
    print("Yeterli puan toplanamadı. Lütfen daha fazla belirti sağlayın.")
