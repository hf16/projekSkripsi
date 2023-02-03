from flask import Flask, render_template, request
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import LinearSVC
import pandas as pd
import re
import csv
import math
import time
import nltk
nltk.download('stopwords')


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
    return render_template('index.html')


def dataset():
    # Membaca file dataset
    df = Dataset_covid
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
    uploadData = pd.read_excel(file)
    global Dataset_covid
    Dataset_covid = uploadData
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']
    train_X = df_preprocessed['Text']
    train_y = df_preprocessed['Label']
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_X)
    # X_test_tfidf = tfidf_vectorizer.transform(test_X)
    svm_no_smote = LinearSVC()
    svm_no_smote.fit(X_train_tfidf, train_y)
    # y_pred_no_smote = svm_no_smote.predict(X_test_tfidf)
    kf = KFold(n_splits=5, shuffle=True)
    precision = make_scorer(precision_score, pos_label='positive')
    recall = make_scorer(recall_score, pos_label='positive')
    f1 = make_scorer(f1_score, pos_label='positive')
    cv_scores_accuracy = cross_val_score(
        svm_no_smote, X_train_tfidf, train_y, cv=kf, scoring='accuracy')
    acc_svm_1 = "%.2f" % (float(cv_scores_accuracy.mean()) * 100)
    cv_scores_precision = cross_val_score(
        svm_no_smote, X_train_tfidf, train_y, cv=kf, scoring=precision)
    precision_svm_1 = cv_scores_precision.mean()
    cv_scores_recall = cross_val_score(
        svm_no_smote, X_train_tfidf, train_y, cv=kf, scoring=recall)
    recall_svm_1 = cv_scores_recall.mean()
    cv_scores_f1 = cross_val_score(
        svm_no_smote, X_train_tfidf, train_y, cv=kf, scoring=f1)
    f1_svm_1 = cv_scores_f1.mean()
    end_time_SVM = time.time()
    waktu_komputasi_SVM = "%.2f" % (end_time_SVM - start_time_SVM)
    # Model SVM dengan Kombinasi SMOTE
    tfidf_vecto = TfidfVectorizer()
    X = tfidf_vecto.fit_transform(X_preprocessed)
    smote = SMOTE()
    X_resample, y_resample = smote.fit_resample(X, df_preprocessed['Label'])
    x_train = X_resample
    y_train = y_resample
    modelSVM_SMOTE = SVC(kernel='linear', probability=True)
    modelSVM_SMOTE.fit(x_train, y_train)
    # predictedSVM_SMOTE = modelSVM_SMOTE.predict(x_test)
    cv_scores_accuracy_smote = cross_val_score(
        modelSVM_SMOTE, x_train, y_train, cv=kf, scoring='accuracy')
    cv_scores_precision_smote = cross_val_score(
        modelSVM_SMOTE, x_train, y_train, cv=kf, scoring=precision)
    cv_scores_recall_smote = cross_val_score(
        modelSVM_SMOTE, x_train, y_train, cv=kf, scoring=recall)
    cv_scores_f1_smote = cross_val_score(
        modelSVM_SMOTE, x_train, y_train, cv=kf, scoring=f1)
    acc_svm_smote_1 = "%.2f" % (float(cv_scores_accuracy_smote.mean()) * 100)
    precision_svm_smote_1 = cv_scores_precision_smote.mean()
    recall_svm_smote_1 = cv_scores_recall_smote.mean()
    f1_svm_smote_1 = cv_scores_f1_smote.mean()
    positive_count = sum(train_y == 'positive')
    negative_count = sum(train_y == 'negative')
    jumlah_data = len(train_y)
    end_time_SVM_SMOTE = time.time()
    waktu_komputasi_SVM_SMOTE = "%.2f" % (
        end_time_SVM_SMOTE - start_time_SVM_SMOTE)
    return render_template('dashboardResult.html', cv_scores_accuracy=cv_scores_accuracy, cv_scores_precision=cv_scores_precision, cv_scores_recall=cv_scores_recall, cv_scores_f1=cv_scores_f1,
                           cv_scores_accuracy_smote=cv_scores_accuracy_smote, cv_scores_precision_smote=cv_scores_precision_smote, cv_scores_recall_smote=cv_scores_recall_smote, cv_scores_f1_smote=cv_scores_f1_smote,
                           acc_svm=acc_svm_1, recall_svm=recall_svm_1, recall_svm_smote=recall_svm_smote_1, precision_svm=precision_svm_1, precision_svm_smote=precision_svm_smote_1, f1_svm=f1_svm_1, acc_svm_smote=acc_svm_smote_1, f1_svm_smote=f1_svm_smote_1, waktu_komputasi_SVM=waktu_komputasi_SVM, waktu_komputasi_SVM_SMOTE=waktu_komputasi_SVM_SMOTE,
                           jumlah_data=jumlah_data, positive_count=positive_count, negative_count=negative_count)


@ app.route('/tabel_svm')
def tabel_svm():
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Tweet']
    train_X, test_X, train_y, test_y = train_test_split(
        df_preprocessed['Text'], df_preprocessed['Label'])
    # Apply TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_X)
    X_test_tfidf = tfidf_vectorizer.transform(test_X)
    # Model SVM
    svm_no_smote = LinearSVC()
    svm_no_smote.fit(X_train_tfidf, train_y)
    # Membuat Hasil Prediksi SVM Tanpa SMOTE
    y_pred_no_smote = svm_no_smote.predict(X_test_tfidf)
    SVM_DF = pd.DataFrame(
        {'Text': test_X, 'SVM': y_pred_no_smote})
    index_to_text_map = {i: X_preprocessed[i]
                         for i in range(len(X_preprocessed))}
    SVM_DF.loc[:, 'Text'] = SVM_DF.index.map(
        index_to_text_map)
  # Set Peganation
    per_page = 20
    num_pages = math.ceil(SVM_DF.shape[0]/per_page)
    page_number = int(request.args.get('page', 1))
    if page_number > num_pages:
        page_number = num_pages
    elif page_number < 1:
        page_number = 1
    SVM_TABEL = SVM_DF.iloc[per_page*(page_number-1):per_page*page_number]
    return render_template('tabel_svm.html', data=SVM_TABEL, num_pages=num_pages, page_number=page_number)


@ app.route('/tabel_svm_smote')
def tabel_svm_smote():
    df_preprocessed = dataset()
    X_preprocessed = df_preprocessed['Text']
    tfidf_vecto = TfidfVectorizer()
    X = tfidf_vecto.fit_transform(X_preprocessed)
    smote = SMOTE()
    X_resample, y_resample = smote.fit_resample(X, df_preprocessed['Label'])
    # Split Data
    x_train, x_test, y_train, y_test = train_test_split(
        X_resample, y_resample, random_state=42)
    # Modeling SVM
    modelSVM_SMOTE = SVC(kernel='linear', probability=True)
    modelSVM_SMOTE.fit(x_train, y_train)
    # predictedSVM_SMOTE = modelSVM_SMOTE.predict(x_test)
    SVM_SMOTE_DF = pd.DataFrame({'Text': [df_preprocessed['Tweet'][i] for i in range(
        len(y_test))], 'SVM_SMOTE': [df_preprocessed['Label'][i] for i in range(
            len(y_test))]})
    # Set Peganation
    per_page = 20
    num_pages = math.ceil(SVM_SMOTE_DF.shape[0]/per_page)
    page_number = int(request.args.get('page', 1))
    if page_number > num_pages:
        page_number = num_pages
    elif page_number < 1:
        page_number = 1
    SVM_SMOTE_TABEL = SVM_SMOTE_DF.iloc[per_page *
                                        (page_number-1):per_page*page_number]
    return render_template('tabel_svm_smote.html', data=SVM_SMOTE_TABEL, num_pages=num_pages, page_number=page_number)


# if __name__ == '__main__':
#     app.run()
if __name__ == '__main__':
    app.run(debug=True)
