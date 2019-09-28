from django.shortcuts import render

# Create your views here.
from django.views.generic import TemplateView

class trainView(TemplateView):
    template_name = 'test.html'
