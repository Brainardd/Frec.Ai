import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump, load
import nltk
from nltk.corpus import words

app = Flask(__name__)

nltk.download('words')
english_vocab = set(words.words())

def check_input_language(input_text):
    words_in_input = input_text.split()
    english_words = [word for word in words_in_input if word.lower() in english_vocab]
    if len(english_words) / len(words_in_input) < 0.25:
        return False
    return True

def train_models(df):
    X = df[['Cleaned_Description']]
    y = df['Career']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=1000)
    preprocessor = ColumnTransformer(transformers=[('tfidf', tfidf, 'Cleaned_Description')], remainder='passthrough')

    dtc = DecisionTreeClassifier(max_depth=3)
    svc = SVC(C=1.0, probability=True)
    rfc = RandomForestClassifier(n_estimators=100)

    dtc_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', dtc)])
    svc_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', svc)])
    rfc_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', rfc)])

    dtc_params = {'classifier__max_depth': [5, 10, 15], 'classifier__min_samples_split': [2, 5, 10],
                  'classifier__min_samples_leaf': [1, 2, 4], 'classifier__max_features': [1, 2, 3, 4, None]}
    svc_params = {'classifier__C': [1, 10, 100], 'classifier__gamma': ['scale', 'auto']}
    rfc_params = {'classifier__n_estimators': [100, 200, 300], 'classifier__max_depth': [None, 10, 20],
                  'classifier__min_samples_split': [2, 5, 10]}

    ensemble = VotingClassifier(estimators=[
        ('decision_tree', dtc_pipeline),
        ('support_vector_machine', svc_pipeline),
        ('random_forest', rfc_pipeline)], voting='soft')

    print("Starting Decision Tree Grid Search...")
    dtc_grid = GridSearchCV(dtc_pipeline, dtc_params, cv=2)
    dtc_grid.fit(X_train, y_train)
    print("Decision Tree Grid Search Completed...")

    print("Starting SVC Grid Search...")
    svc_grid = GridSearchCV(svc_pipeline, svc_params, cv=2)
    svc_grid.fit(X_train, y_train)
    print("SVC Grid Search Completed...")

    print("Starting Random Forest Grid Search...")
    rfc_grid = GridSearchCV(rfc_pipeline, rfc_params, cv=2)
    rfc_grid.fit(X_train, y_train)
    print("Random Forest Grid Search Completed...")

    dtc_pipeline = dtc_grid.best_estimator_
    svc_pipeline = svc_grid.best_estimator_
    rfc_pipeline = rfc_grid.best_estimator_

    print("Fitting Ensemble Model...")
    ensemble.fit(X_train, y_train)
    print("Ensemble Model Fit Completed...")

    dtc_accuracy = accuracy_score(y_test, dtc_pipeline.predict(X_test))
    svc_accuracy = accuracy_score(y_test, svc_pipeline.predict(X_test))
    rfc_accuracy = accuracy_score(y_test, rfc_pipeline.predict(X_test))
    ensemble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))
    
    print(f"Decision Tree Accuracy: {dtc_accuracy}")
    print(f"SVC Accuracy: {svc_accuracy}")
    print(f"Random Forest Accuracy: {rfc_accuracy}")
    print(f"Ensemble Model Accuracy: {ensemble_accuracy}")

    print("Training and saving models...")
    print(dtc_pipeline)
    print(svc_pipeline)
    print(rfc_pipeline)
    print(ensemble)

    # Saving models
    dump(dtc_pipeline, 'dtc_pipeline.joblib')
    dump(svc_pipeline, 'svc_pipeline.joblib')
    dump(rfc_pipeline, 'rfc_pipeline.joblib')
    dump(ensemble, 'ensemble.joblib')

    print("Models saved successfully!")

    return dtc_pipeline, svc_pipeline, rfc_pipeline, ensemble

try:
    dtc_pipeline = load('dtc_pipeline.joblib')
    svc_pipeline = load('svc_pipeline.joblib')
    rfc_pipeline = load('rfc_pipeline.joblib')
    ensemble = load('ensemble.joblib')
except FileNotFoundError:
    df = pd.read_csv('C:/Users/brain/Desktop/NexGen/NexGen/clean_file.csv')
    dtc_pipeline, svc_pipeline, rfc_pipeline, ensemble = train_models(df)

@app.route('/Main.html')
def main():
    return render_template('Main.html')

@app.route('/Career.html')
def career():
    return render_template('Career.html')

@app.route('/Questionaire.html', methods=['GET', 'POST'])
def questionaire():
    if request.method == 'POST':
        
        
        interests = request.form['Interests']
        skills = request.form['Skills']
        hobbies = request.form['Hobbies']

        user_input = interests + skills + hobbies

        if not check_input_language(user_input):
            return "Please enter text in English."

        user_data = pd.DataFrame({'Cleaned_Description': [user_input]})

        dtc_prediction = dtc_pipeline.predict(user_data)[0]
        svc_prediction = svc_pipeline.predict(user_data)[0]
        rfc_prediction = rfc_pipeline.predict(user_data)[0]
        ensemble_prediction = ensemble.predict(user_data)[0]
        
        default_label = 'Unknown'
        dtc_prediction_str = str(dtc_prediction)
        svc_prediction_str = str(svc_prediction)
        rfc_prediction_str = str(rfc_prediction)

        dtc_prediction = dtc_prediction_str if dtc_prediction_str in dtc_pipeline.named_steps['classifier'].classes_ else default_label
        svc_prediction = svc_prediction_str if svc_prediction_str in svc_pipeline.named_steps['classifier'].classes_ else default_label
        rfc_prediction = rfc_prediction_str if rfc_prediction_str in rfc_pipeline.named_steps['classifier'].classes_ else default_label

        ensemble_prediction_str = str(ensemble_prediction)
        ensemble_prediction = ensemble_prediction_str if ensemble_prediction_str in ensemble.classes_ else default_label

        
        return jsonify({
        'dtc_prediction': dtc_prediction,
        'svc_prediction': svc_prediction,
        'rfc_prediction': rfc_prediction,
        'ensemble_prediction': ensemble_prediction
    })
        
    return render_template('Questionaire.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
