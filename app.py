from flask import Flask, render_template, request
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import LinearSVC
import pandas as pd
import re
import nltk
import csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import math

# Unduh stopwords dan punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

Dataset_covid = pd.DataFrame()


def cleansing(data):
    # Case Folding
    data = data.lower()
    # Hapus tab, baris baru, dan irisan belakang
    data = data.strip()
    # Hapus non ASCII (emotikon)
    data = re.sub(r'[^\x00-\x7F]+', '', data)
    # Hapus mention, link, hashtag
    data = re.sub(r'@\S+|https?://\S+|#\S+', '', data)
    # Hapus URL
    data = re.sub(r'http\S+', '', data)
    # Menghilangkan nomor
    data = re.sub(r'\d+', '', data)
    # Hilangkan tanda baca
    data = re.sub(r'[^\w\s]', '', data)
    # Hapus spasi di awal & akhir
    data = data.strip()
    # Hapus myltiple whitespac menjadi spasi tunggal
    data = re.sub(r'\s+', ' ', data)
    # Hapus satu karakter
    data = re.sub(r'\b\w{1}\b', '', data)

    return data


def tokenizing(data):
    return nltk.word_tokenize(data)


def normalisasi(data):
    kata_baku = {}
    with open('normalisasi.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            kata_baku[row[0]] = row[1]
    kalimat_baru = []
    for kata in data:
        if kata in kata_baku:
            kalimat_baru.append(kata_baku[kata])
        else:
            kalimat_baru.append(kata)
    return kalimat_baru


def stopword(data):
    stopwords = nltk.corpus.stopwords.words('indonesian')
    return [kata for kata in data if kata not in stopwords]


def stemming(data):
    stemmer = nltk.stem.PorterStemmer()
    return [stemmer.stem(kata) for kata in data]


def preprocessing(data):
    # Cleansing
    data = cleansing(data)
    # Tokenizing
    data = tokenizing(data)
    # Normalisasi kata baku
    data = normalisasi(data)
    # Stopword
    data = stopword(data)
    # Stemming
    data = stemming(data)

    return data


@app.route('/')
def index():
    # Menampilkan halaman index.html dengan button mulai
    return render_template('indexEx.html')


def dataset():
    # Membaca file Excel
    df = Dataset_covid
    Z = df['Id'].tolist()
    X = df['Text'].tolist()
    y = df['Label'].tolist()

    # Preprocess the data
    X_preprocessed = [preprocessing(data) for data in X]

    # Flatten the list of tokens into a list of strings
    X_preprocessed = [' '.join(tokens) for tokens in X_preprocessed]
    df_preprocessed = pd.DataFrame(
        {'Id': Z, 'Text': X_preprocessed, 'Label': y})

    return df_preprocessed


@app.route('/result', methods=['POST'])
def result():
    file = request.files['file']
    uploadData = pd.read_excel(file)
    global Dataset_covid
    Dataset_covid = uploadData
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']

    train_X, test_X, train_y, test_y = train_test_split(
        df_preprocessed['Text'], df_preprocessed['Label'], train_size=0.8)

    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_X)
    X_test_tfidf = tfidf_vectorizer.transform(test_X)

    # Train a SVM
    svm_no_smote = LinearSVC(C=3)
    svm_no_smote.fit(X_train_tfidf, train_y)

    # Make predictions on the test set using the SVM model without SMOTE
    y_pred_no_smote = svm_no_smote.predict(X_test_tfidf)
    acc_svm = accuracy_score(test_y, y_pred_no_smote)
    acc_svm_1 = round(acc_svm, 4)
    recall_svm = recall_score(test_y, y_pred_no_smote, pos_label='negative')
    recall_svm_1 = round(recall_svm, 4)
    precision_svm = precision_score(
        test_y, y_pred_no_smote, pos_label='negative')
    precision_svm_1 = round(precision_svm, 4)
    f1_svm = f1_score(
        test_y, y_pred_no_smote, pos_label='negative')
    f1_svm_1 = round(f1_svm, 4)

    kf = KFold(n_splits=5, shuffle=True)
    cv_scores_SVM = cross_val_score(
        svm_no_smote, X_train_tfidf, train_y, cv=kf)
    K_Fold_SVM = round(cv_scores_SVM.mean(), 4)

    # ------------- SVM  SMOTE ---------------------------
    tfidf_vecto = TfidfVectorizer()
    X = tfidf_vecto.fit_transform(X_preprocessed)

    smote = SMOTE()
    X_resample, y_resample = smote.fit_resample(X, df_preprocessed['Label'])

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        X_resample, y_resample, train_size=0.8, random_state=42)

    # Modeling
    modelSVM_SMOTE = SVC(kernel='linear', probability=True)
    modelSVM_SMOTE.fit(x_train, y_train)
    predictedSVM_SMOTE = modelSVM_SMOTE.predict(x_test)
    acc_svm_smote = accuracy_score(y_test, predictedSVM_SMOTE)
    acc_svm_smote_1 = round(acc_svm_smote, 4)
    recall_svm_smote = recall_score(
        y_test, predictedSVM_SMOTE, pos_label='negative')
    recall_svm_smote_1 = round(recall_svm_smote, 4)
    precision_svm_smote = recall_score(
        y_test, predictedSVM_SMOTE, pos_label='negative')
    precision_svm_smote_1 = round(precision_svm_smote, 4)
    f1_svm_smote = f1_score(
        y_test, predictedSVM_SMOTE, pos_label='negative')
    f1_svm_smote_1 = round(f1_svm_smote, 4)
    # K-FOLD SVM+SMOTE
    kf = KFold(n_splits=5, shuffle=True)
    cv_scores_SMOTE = cross_val_score(
        modelSVM_SMOTE, x_train, y_train, cv=kf)
    K_Fold_SMOTE = round(cv_scores_SMOTE.mean(), 4)
    return render_template('resultEx.html', acc_svm=acc_svm_1, recall_svm=recall_svm_1, recall_svm_smote=recall_svm_smote_1, precision_svm=precision_svm_1, precision_svm_smote=precision_svm_smote_1, f1_svm=f1_svm_1, acc_svm_smote=acc_svm_smote_1, f1_svm_smote=f1_svm_smote_1, cv_scores_SVM=cv_scores_SVM, K_Fold_SVM=K_Fold_SVM, cv_scores_SMOTE=cv_scores_SMOTE, K_Fold_SMOTE=K_Fold_SMOTE)


@app.route('/SVM')
def SVM():
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']

    train_X, test_X, train_y, test_y = train_test_split(
        df_preprocessed['Text'], df_preprocessed['Label'], train_size=0.8)

    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_X)
    X_test_tfidf = tfidf_vectorizer.transform(test_X)

    # Train a SVM
    svm_no_smote = LinearSVC(C=3)
    svm_no_smote.fit(X_train_tfidf, train_y)

    # Make predictions on the test set using the SVM model without SMOTE
    y_pred_no_smote = svm_no_smote.predict(X_test_tfidf)

    # SVM_DF = pd.DataFrame({'Text': [X_preprocessed[i] for i in range(
    #     len(test_y))], 'Label': test_y, 'SVM': y_pred_no_smote})
    SVM_DF = pd.DataFrame({'Text': [X_preprocessed[i] for i in range(
        len(test_y))], 'Label': [df_preprocessed['Label'][i] for i in range(
            len(test_y))], 'SVM': y_pred_no_smote})

  # set the number of items per page
    per_page = 20
    num_pages = math.ceil(SVM_DF.shape[0]/per_page)

    # get the page number from the request's query parameters
    page_number = int(request.args.get('page', 1))
    if page_number > num_pages:
        page_number = num_pages
    elif page_number < 1:
        page_number = 1

    # get the data that should be shown on the current page
    SVM_TABEL = SVM_DF.iloc[per_page*(page_number-1):per_page*page_number]
    return render_template('SVMEx.html', data=SVM_TABEL, num_pages=num_pages, page_number=page_number)


@app.route('/SVM_SMOTE')
def SVM_SMOTE():
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']

    tfidf_vecto = TfidfVectorizer()
    X = tfidf_vecto.fit_transform(X_preprocessed)

    smote = SMOTE()
    X_resample, y_resample = smote.fit_resample(X, df_preprocessed['Label'])

    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        X_resample, y_resample, train_size=0.8, random_state=42)

    # Modeling
    modelSVM_SMOTE = SVC(kernel='linear', probability=True)
    modelSVM_SMOTE.fit(x_train, y_train)
    predictedSVM_SMOTE = modelSVM_SMOTE.predict(x_test)

    # SVM_SMOTE_DF = pd.DataFrame({'Text': [X_preprocessed[i] for i in range(
    #     len(y_test))], 'Label': y_test, 'SVM_SMOTE': predictedSVM_SMOTE})
    SVM_SMOTE_DF = pd.DataFrame({'Text': [X_preprocessed[i] for i in range(
        len(y_test))], 'Label': [df_preprocessed['Label'][i] for i in range(
            len(y_test))], 'SVM_SMOTE': predictedSVM_SMOTE})

    # set the number of items per page
    per_page = 20
    num_pages = math.ceil(SVM_SMOTE_DF.shape[0]/per_page)

    # get the page number from the request's query parameters
    page_number = int(request.args.get('page', 1))
    if page_number > num_pages:
        page_number = num_pages
    elif page_number < 1:
        page_number = 1
    SVM_SMOTE_TABEL = SVM_SMOTE_DF.iloc[per_page *
                                        (page_number-1):per_page*page_number]
    return render_template('SVM_SMOTE_Ex.html', data=SVM_SMOTE_TABEL, num_pages=num_pages, page_number=page_number)


if __name__ == '__main__':
    app.run()
