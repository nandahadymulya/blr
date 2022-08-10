from django import forms  

class UploadFileForm(forms.Form):  
    file = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control','id': 'form','required': 'true',}),label='',)

class PredictForm(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control','id': 'form','required': 'true',}),label='School',)
    people = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control','id': 'form','required': 'true',}),label='People',)
    technology = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control','id': 'form','required': 'true',}),label='Technology',)
    innovation = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control','id': 'form','required': 'true',}),label='Innovation',)
    self_development = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control','id': 'form','required': 'true',}),label='Self Development',)