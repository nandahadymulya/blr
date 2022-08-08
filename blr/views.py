from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render
from django.contrib import messages
from .forms import UploadFileForm
from .functions import handle_uploaded_file

import pandas as pd
import numpy as np

import json
global data
def index(request):
    if request.method == 'POST':  
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
          handle_uploaded_file(request.FILES['file'])
          data = pd.read_csv('blr/uploads/'+request.FILES['file'].name)
          message = messages.success(request, 'File uploaded successfully')

          # output data survey
          data.to_json('blr/json/data_survey.json', orient='records')
          with open('blr/json/data_survey.json', 'r') as json_file:
              json_data_survey = json.load(json_file)
          template = 'views/datasurvey.html'
          context = {
            'message': message,
            'json_data_survey': json_data_survey,
          }
          return render(request, template, context)
        return HttpResponseRedirect('/')
    else:
        form = UploadFileForm()
        template = 'views/index.html'
        context = {
          'form': form
        }
        return render(request, template, context)

def datacleaning(request):
    data = request.session.get('data')
    # data cleaning
    data = data.drop(data.columns[[0,2,3]], axis=1)
    # output data cleaning
    data.to_json('blr/json/data_cleaning.json', orient='records')
    with open('blr/json/data_cleaning.json', 'r') as json_file:
        json_data_cleaning = json.load(json_file)
        template = 'views/datacleaning.html'
        context = {
          'json_data_cleaning': json_data_cleaning,
        }
    return render(request, template, context)

def datatransformation(request):
    # # data transformation
    # data_transformation = data.drop(data.columns[[0,2,3]], axis=1)
    # sum_p = ['P1','P2','P3','P4','P5','P6']
    # sum_t = ['T1','T2','T3','T4','T5','T6']
    # sum_i = ['I1','I2','I3','I4','I5','I6']
    # sum_sd = ['SD1','SD2','SD3','SD4','SD5','SD6']
    # data_transformation['people'] = data_transformation[sum_p].sum(axis=1, skipna=False)
    # data_transformation['technology'] = data_transformation[sum_t].sum(axis=1, skipna=False)
    # data_transformation['innovation'] = data_transformation[sum_i].sum(axis=1, skipna=False)
    # data_transformation['self_development'] = data_transformation[sum_sd].sum(axis=1, skipna=False)
    return render(request, 'views/datatransformation.html')

def about(request):
    return render(request, 'views/about.html')