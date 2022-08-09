from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    # path('preprocessing/', views.preprocessing, name='preprocessing'),
    path('processing/', views.processing, name='processing'),
    path('about/', views.about, name='about'),
]
