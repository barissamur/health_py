import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random
import json
from flask import Flask, request, render_template, session, redirect, url_for

app = Flask(__name__)
app.secret_key = "sizin_gizli_anahtarınız"  # Bir gizli anahtar belirleyin

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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        response = request.form["response"]
        symptom = session.get("current_symptom")
        symptoms = session.get("symptoms", [])
        asked_symptoms = set(session.get("asked_symptoms", []))

        if response.lower() == "evet":
            session["total_score"] = session.get("total_score", 0) + 1

        asked_symptoms.add(symptom)
        session["asked_symptoms"] = list(asked_symptoms)

        if session["total_score"] >= 10:
            return redirect(url_for("predict"))

        # Belirtileri ve ilişkili hastalıkları güncelleme
        if not symptoms:
            random_symptom = random.choice(list(set(X_train.columns) - asked_symptoms))
            related_diseases = find_related_diseases(random_symptom)
            symptoms = get_related_symptoms(related_diseases, asked_symptoms)

        session["current_symptom"] = symptoms.pop(0)
        session["symptoms"] = symptoms
        # Puan güncellemesi ve şablona aktarılması
        return render_template(
            "index.html",
            symptom=translate_symptom(session["current_symptom"]),
            total_score=session["total_score"],
        )

    # İlk belirtiyi seçme ve oturumu başlatma
    random_symptom = random.choice(list(X_train.columns))
    asked_symptoms = {random_symptom}
    related_diseases = find_related_diseases(random_symptom)
    symptoms = get_related_symptoms(related_diseases, asked_symptoms)

    session["total_score"] = 0
    session["asked_symptoms"] = list(asked_symptoms)
    session["current_symptom"] = symptoms.pop(0)
    session["symptoms"] = symptoms

    return render_template(
        "index.html",
        symptom=translate_symptom(session["current_symptom"]),
        total_score=session.get("total_score", 0),
    )


@app.route("/predict")
def predict():
    # Hastalık tahmini yapma
    user_data = {symptom: 0 for symptom in X_train.columns}
    for symptom in session.get("asked_symptoms", []):
        user_data[symptom] = 1
    user_data_df = pd.DataFrame([user_data])
    predicted_disease = model.predict(user_data_df)[0]
    translated_disease = translate_disease(predicted_disease)
    return render_template("prediction.html", disease=translated_disease)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
