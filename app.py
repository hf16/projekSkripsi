from flask import Flask, render_template, request
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
from sklearn.svm import LinearSVC
import pandas as pd
import re
import csv
import math
import time
import nltk
nltk.download('stopwords')


app = Flask(__name__)


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


def TFIDF(prm_tfidf):
    tfidf_vectorizer = TfidfVectorizer()
    uji_tfidf = tfidf_vectorizer.fit_transform(prm_tfidf)
    return uji_tfidf


def SVM(prm_x, prm_y):
    svm = LinearSVC()
    uji_svm = svm.fit(prm_x, prm_y)
    return uji_svm


def oversamplingSMOTE():
    uji_smote = SMOTE()
    return uji_smote


@app.route('/')
def index():
    return render_template('index.html')


dataset_input = pd.DataFrame()


def dataset():
    # Membaca file dataset
    df = dataset_input
    Z = df['Id'].tolist()
    X = df['Text'].tolist()
    y = df['Label'].tolist()
    X_preprocessed = [preprocessing(data) for data in X]
    X_preprocessed = [' '.join(tokens) for tokens in X_preprocessed]
    df_preprocessed = pd.DataFrame(
        {'Id': Z, 'Text': X_preprocessed, 'Label': y, 'Tweet': X})
    return df_preprocessed


@app.route('/Dashboard', methods=['POST'])
def predictResult_svm_smote():
    start_time_SVM = time.time()
    start_time_SVM_SMOTE = time.time()
    file = request.files['file']
    uploadData = pd.read_csv(file)
    global dataset_input
    dataset_input = uploadData
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']
    uji_X = df_preprocessed['Text']
    uji_Y = df_preprocessed['Label']
    X_uji_tfidf = TFIDF(uji_X)
    svm_no_smote = SVM(X_uji_tfidf, uji_Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    precision = make_scorer(precision_score, pos_label='positive')
    recall = make_scorer(recall_score, pos_label='positive')
    f1 = make_scorer(f1_score, pos_label='positive')
    cv_scores_accuracy = cross_val_score(
        svm_no_smote, X_uji_tfidf, uji_Y, cv=kf, scoring='accuracy')
    acc_svm_1 = "%.2f" % (float(cv_scores_accuracy.mean()) * 100)
    cv_scores_precision = cross_val_score(
        svm_no_smote, X_uji_tfidf, uji_Y, cv=kf, scoring=precision)
    prec_svm = "%.2f" % (float(cv_scores_precision.mean()) * 100)
    cv_scores_recall = cross_val_score(
        svm_no_smote, X_uji_tfidf, uji_Y, cv=kf, scoring=recall)
    recall_svm = "%.2f" % (float(cv_scores_recall.mean()) * 100)
    cv_scores_f1 = cross_val_score(
        svm_no_smote, X_uji_tfidf, uji_Y, cv=kf, scoring=f1)
    f1_svm = "%.2f" % (float(cv_scores_f1.mean()) * 100)
    end_time_SVM = time.time()
    waktu_komputasi_SVM = "%.2f" % (end_time_SVM - start_time_SVM)
    # Model SVM dengan Kombinasi SMOTE
    X_tfidf_smote = TFIDF(X_preprocessed)
    smote = oversamplingSMOTE()
    X_resample, y_resample = smote.fit_resample(
        X_tfidf_smote, df_preprocessed['Label'])
    X_uji = X_resample
    Y_uji = y_resample
    modelSVM_SMOTE = SVM(X_uji, Y_uji)
    cv_scores_accuracy_smote = cross_val_score(
        modelSVM_SMOTE, X_uji, Y_uji, cv=kf, scoring='accuracy')
    acc_svm_smote_1 = "%.2f" % (float(cv_scores_accuracy_smote.mean()) * 100)
    cv_scores_precision_smote = cross_val_score(
        modelSVM_SMOTE, X_uji, Y_uji, cv=kf, scoring=precision)
    prec_smote = "%.2f" % (
        float(cv_scores_precision_smote.mean()) * 100)
    cv_scores_recall_smote = cross_val_score(
        modelSVM_SMOTE, X_uji, Y_uji, cv=kf, scoring=recall)
    recall_smote = "%.2f" % (float(cv_scores_recall_smote.mean()) * 100)
    cv_scores_f1_smote = cross_val_score(
        modelSVM_SMOTE, X_uji, Y_uji, cv=kf, scoring=f1)
    f1_smote = "%.2f" % (float(cv_scores_f1_smote.mean()) * 100)
    positive_count = sum(uji_Y == 'positive')
    negative_count = sum(uji_Y == 'negative')
    jumlah_data = len(uji_Y)
    end_time_SVM_SMOTE = time.time()
    waktu_komputasi_SVM_SMOTE = "%.2f" % (
        end_time_SVM_SMOTE - start_time_SVM_SMOTE)
    return render_template('dashboardResult.html', acc_svm=acc_svm_1, acc_svm_smote=acc_svm_smote_1,  waktu_komputasi_SVM=waktu_komputasi_SVM, waktu_komputasi_SVM_SMOTE=waktu_komputasi_SVM_SMOTE,
                           jumlah_data=jumlah_data, positive_count=positive_count, negative_count=negative_count, prec_svm=prec_svm, recall_svm=recall_svm, f1_svm=f1_svm, prec_smote=prec_smote, recall_smote=recall_smote, f1_smote=f1_smote)


@ app.route('/tabel_svm')
def predict_tabel_svm():
    df_preprocessed = dataset()
    uji_X = df_preprocessed['Text']
    uji_Y = df_preprocessed['Label']
    X_uji_tfidf = TFIDF(uji_X)
    svm_no_smote = SVM(X_uji_tfidf, uji_Y)
    kf = KFold(n_splits=5, shuffle=True)
    cv_predictions = cross_val_predict(
        svm_no_smote, X_uji_tfidf, uji_Y, cv=kf)
    cv_scores_accuracy = cross_val_score(
        svm_no_smote, X_uji_tfidf, uji_Y, cv=kf, scoring='accuracy')
    acc_svm_1 = "%.2f" % (float(cv_scores_accuracy.mean()) * 100)
    SVM_DF = pd.DataFrame({'Text': [df_preprocessed['Tweet'][i] for i in range(
        len(uji_X))], 'SVM': cv_predictions})
    per_page = 5
    num_pages = math.ceil(SVM_DF.shape[0]/per_page)
    page_number = int(request.args.get('page', 1))
    if page_number > num_pages:
        page_number = num_pages
    elif page_number < 1:
        page_number = 1
    SVM_TABEL = SVM_DF.iloc[per_page*(page_number-1):per_page*page_number]

    return render_template('tabel_svm.html', data=SVM_TABEL, num_pages=num_pages, page_number=page_number, acc_svm_1=acc_svm_1)


@ app.route('/tabel_svm_smote')
def predict_tabel_svm_smote():
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']
    cv_label = df_preprocessed['Label']
    X_uji_tfidf_smote = TFIDF(X_preprocessed)
    smote = oversamplingSMOTE()
    X_resample, y_resample = smote.fit_resample(
        X_uji_tfidf_smote, df_preprocessed['Label'])
    modelSVM_SMOTE = SVM(X_resample, y_resample)
    kf = KFold(n_splits=5, shuffle=True)
    cv_predictions = cross_val_predict(
        modelSVM_SMOTE, X_uji_tfidf_smote, cv_label, cv=kf)
    cv_scores_accuracy_smote = cross_val_score(
        modelSVM_SMOTE, X_uji_tfidf_smote, cv_label, cv=kf, scoring='accuracy')
    cv_predictions = cv_label
    acc_svm_smote_1 = "%.2f" % (float(cv_scores_accuracy_smote.mean()) * 100)
    SVM_SMOTE_DF = pd.DataFrame({'Text': [df_preprocessed['Tweet'][i] for i in range(
        len(cv_predictions))], 'SVM_SMOTE': cv_predictions})
    per_page = 5
    num_pages = math.ceil(SVM_SMOTE_DF.shape[0]/per_page)
    page_number = int(request.args.get('page', 1))
    if page_number > num_pages:
        page_number = num_pages
    elif page_number < 1:
        page_number = 1
    SVM_SMOTE_TABEL = SVM_SMOTE_DF.iloc[per_page *
                                        (page_number-1):per_page*page_number]
    return render_template('tabel_svm_smote.html', data=SVM_SMOTE_TABEL, num_pages=num_pages, page_number=page_number, acc_svm_smote_1=acc_svm_smote_1)


if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
