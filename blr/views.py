import http
from django.http import HttpResponse
from django.shortcuts import render
from django.contrib import messages
from .forms import UploadFileForm
from .functions import handle_uploaded_file

def index(request):
    if request.method == 'POST':  
      form = UploadFileForm(request.POST, request.FILES)
      if form.is_valid():
        handle_uploaded_file(request.FILES['file'])
        messages.success(request, 'File uploaded successfully')
        return HttpResponse(status=http.HTTPStatus.OK)
    else:
      form = UploadFileForm() 
      return render(request, 'views/index.html',{'form':form})

def about(request):
    return render(request, 'views/about.html')