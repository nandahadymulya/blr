from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    # path('datasurvey/', views.datasurvey, name='datasurvey'),
    path('datacleaning/', views.datacleaning, name='datacleaning'),
    path('datatransformation/', views.datatransformation, name='datatransformation'),
    path('about/', views.about, name='about'),
    # path('varealibity/', views.varealibity, name='varealibity'),
]
