from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.contrib import messages
from .forms import UploadFileForm, PredictForm
from .functions import handle_uploaded_file

import pandas as pd
import numpy as np
# from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import json
# global data
def index(request):
    if request.method == 'POST':  
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
          handle_uploaded_file(request.FILES['file'])
          message = messages.success(request, 'File uploaded successfully')

          data = pd.read_csv('blr/uploads/'+request.FILES['file'].name)

          # output data survey
          json_data_survey = data.reset_index().to_json(orient='records')
          result_data_survey = []
          result_data_survey = json.loads(json_data_survey)

          # data cleaning
          data_cleaning = data.drop(data.columns[[0,2,3]], axis=1)          
          # output data cleaning
          json_data_cleaning = data_cleaning.reset_index().to_json(orient='records')
          result_data_cleaning = []
          result_data_cleaning = json.loads(json_data_cleaning)

          # data transformation
          data_transformation = data_cleaning
          sum_p = ['P1','P2','P3','P4','P5','P6']
          sum_t = ['T1','T2','T3','T4','T5','T6']
          sum_i = ['I1','I2','I3','I4','I5','I6']
          sum_sd = ['SD1','SD2','SD3','SD4','SD5','SD6']
          data_transformation['people'] = data_transformation[sum_p].sum(axis=1, skipna=False)
          data_transformation['technology'] = data_transformation[sum_t].sum(axis=1, skipna=False)
          data_transformation['innovation'] = data_transformation[sum_i].sum(axis=1, skipna=False)
          data_transformation['self_development'] = data_transformation[sum_sd].sum(axis=1, skipna=False)
          data_transformation = data_transformation.drop(data_transformation.loc[:, 'P1':'SD6'].columns, axis=1)
          # output data transformation
          json_data_transformation = data_transformation.reset_index().to_json(orient='records')
          result_data_transformation = []
          result_data_transformation = json.loads(json_data_transformation)
          # data.to_json('blr/json/data_survey.json', orient='records')
          # with open('blr/json/data_survey.json', 'r') as json_file:
          #     json_data_survey = json.load(json_file)

          # data scaling
          data_scaling = data_transformation.iloc[:, 1:5]
          # MinMaxScaler
          scaler = MinMaxScaler()
          scaled_features = scaler.fit_transform(data_scaling)
          features = pd.DataFrame(scaled_features, columns=data_scaling.columns)

          # save data_transformation, data_scaling & features to json
          data_transformation.to_json('blr/json/data_transformation.json', orient='records')
          data_scaling.to_json('blr/json/data_scaling.json', orient='records')
          features.to_json('blr/json/features.json', orient='records')

           # output data scaling
          json_data_scaling = features.reset_index().to_json(orient='records')
          result_data_scaling = []
          result_data_scaling = json.loads(json_data_scaling)
          template = 'views/preprocessing.html'
          context = {
            'message': message,
            'result_data_survey': result_data_survey,
            'result_data_cleaning': result_data_cleaning,
            'result_data_transformation': result_data_transformation,
            'result_data_scaling': result_data_scaling,
          }
          return render(request, template, context)
        return HttpResponseRedirect('processing', context)
    else:
        form = UploadFileForm()
        template = 'views/index.html'
        context = {
          'form': form
        }
        return render(request, template, context)

def processing(request):
    dataset = pd.read_json('blr/json/data_transformation.json', orient='records')
    df = pd.read_json('blr/json/features.json', orient='records')

    # Intance KMeans
    kmeans = KMeans(
        init="random",
        n_clusters=2,
        max_iter=300,
        random_state=42
    )
    # Fit KMeans
    kmeans.fit(df)
    #  The lowest SSE value
    lowest_sse = kmeans.inertia_
    # The number of clusters
    n_clusters = kmeans.n_clusters
    # The cluster centroids
    # centroids = kmeans.cluster_centers_
    # The cluster labels
    cluster = kmeans.labels_
    #  plot detail kmeans
    json_detail_kmeans = {
      'detail': [
        {
          'lowest_sse': lowest_sse,
          'n_clusters': n_clusters,
        },
      ]
    }

    #  Menampung cluster pada column class
    dataset['class'] = cluster
    # Labeling
    def labeling(df):
        if df['class'] == 0:
            return 'Siap'
        elif df['class'] == 1:
            return 'Belum Siap'
    
    dataset['class_new']=dataset.apply(labeling,axis=1)
    result_kmeans = dataset.loc[:, ['Sekolah','people','technology','innovation','self_development','class_new']]

    # output kmeans
    json_kmeans = result_kmeans.reset_index().to_json(orient='records')
    result_json_kmeans = []
    result_json_kmeans = json.loads(json_kmeans)

    # SSE
    # evaluasi kmeans
    # sse = []
    # index = range(1, 1)
    # for i, nilai_sse in enumerate(sse):
    #     print(i, nilai_sse)
    #     kmeans = KMeans(n_clusters=i, random_state=42)
    #     kmeans.fit(df)
    #     sse_ = kmeans.inertia_
    #     sse.append(sse_)

    # plot sse
    # dataset['sse']=sse

    # output SSE
    # result_sse = dataset.loc[:, ['Sekolah','class_new','sse']]
    # json_sse = result_sse.reset_index().to_json(orient='records')
    # result_json_sse = []
    # result_json_sse = json.loads(json_sse)

    #  SVM
    svm_X = df[['people','technology','innovation','self_development']]
    svm_y = dataset[['class_new']]
    svm_y = svm_y.replace(['Siap', 'Belum Siap'], [1, -1])

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(svm_X, svm_y, test_size = 0.2)

    # Instance SVM
    classifier = svm.SVC(kernel = 'linear')

    # Fit SVM
    classifier.fit(X_train, y_train.values.ravel())
    
    # Predict SVM
    y_predict = classifier.predict(X_test)
    
    # Predict Features
    features_predict = classifier.predict(df)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    accuracy = accuracy * 100
    precision = precision_score(y_test, y_predict)
    precision = precision * 100
    recall = recall_score(y_test, y_predict)
    recall = recall * 100

    #output X_train
    json_X_train = X_train.reset_index().to_json(orient='records')
    result_json_X_train = []
    result_json_X_train = json.loads(json_X_train)

    #output y_train
    json_y_train = y_train.reset_index().to_json(orient='records')
    result_json_y_train = []
    result_json_y_train = json.loads(json_y_train)

    # output X_test
    json_X_test = X_test.reset_index().to_json(orient='records')
    result_json_X_test = []
    result_json_X_test = json.loads(json_X_test)

    # output y_test
    json_y_test = y_test.reset_index().to_json(orient='records')
    result_json_y_test = []
    result_json_y_test = json.loads(json_y_test)

    # output y_predict
    y_predict_df = pd.DataFrame(y_predict)
    y_predict_df = y_predict_df.replace([1, -1], ['1 = Siap', ' -1 = Belum Siap'])
    json_y_predict = y_predict_df.reset_index().to_json(orient='records')
    result_json_y_predict = []
    result_json_y_predict = json.loads(json_y_predict)
    
    # output features_predict
    # def parsing_result(request):
    #   features_predict_df = pd.DataFrame(features_predict)
    #   features_predict_df = features_predict_df.replace([1, -1], ['1 = Siap', ' -1 = Belum Siap'])
    #   json_features_predict = features_predict_df.reset_index().to_json(orient='records')
    #   result_json_features_predict = []
    #   result_json_features_predict = json.loads(json_features_predict)
      
    #   template = 'view/result.html'
    #   context = {
    #     'result_json_features_predict': result_json_features_predict,
    #   }
    #   return render(request, template, context)
    
    # output svm
    # print(y_predict)
    result_svm = dataset.loc[:, ['Sekolah','class_new']]
    json_svm = result_svm.reset_index().to_json(orient='records')
    result_json_svm = {}
    result_json_svm = json.loads(json_svm)

    # output confusion matrix
    cm_df = pd.DataFrame(cm)
    json_cm = cm_df.to_json(orient='records')
    result_json_cm = [[]]
    result_json_cm = json.loads(json_cm)

    template = 'views/processing.html'
    context = {
      'dataset': dataset,
      'df': df,
      'result_json_kmeans': result_json_kmeans,
      'json_detail_kmeans': json_detail_kmeans,
      # 'result_json_sse': result_json_sse,
      'result_json_X_train': result_json_X_train,
      'result_json_y_train': result_json_y_train,
      'result_json_X_test': result_json_X_test,
      'result_json_y_test': result_json_y_test,
      'result_json_y_predict': result_json_y_predict,
      'result_json_svm': result_json_svm,
      'result_json_cm': result_json_cm,
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
    }
    return render(request, template, context)

def predict(request):
    if request.method == 'POST':
      form = UploadFileForm(request.POST)
      if form.is_valid():
        # name = form.cleaned_data['name']
        # people = form.cleaned_data['people']
        # technology = form.cleaned_data['technology']
        # innovation = form.cleaned_data['innovation']
        # self_development = form.cleaned_data['self_development']
        # data_predict = {
        #   'name': name,
        #   'people': people,
        #   'technology': technology,
        #   'innovation': innovation,
        #   'self_development': self_development,
        # }
        # df_predict = pd.DataFrame(data_predict, index=[0])
        
        scaled_df_predict = MinMaxScaler()

        # json_data_predict = df_predict.reset_index().to_json(orient='records')
        # result_json_data_predict = []
        # result_json_data_predict = json.loads(json_data_predict)
        # print(df_predict)
        template = 'views/result.html'
        context = {
          # 'result_json_data_predict': result_json_data_predict,
        }
      messages.success(request, 'Form Submitted Successfully!')
      return render(request, template, context)

    else:
      form = UploadFileForm()
      template = 'views/predict.html'
      context = {
        'form': form,
      }
      return render(request, template, context)

# def result(request):
#     template = 'views/result.html'
#     return render(request, template)