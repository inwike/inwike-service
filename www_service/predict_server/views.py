from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import json
import predict_server.ml as ml


# Create your views here.
from django.views.generic import TemplateView, View

class trainView(View):
    template_name = 'test.html'
    test_data = [{'Должность': 1, 'КоличествоДолжностей': 29,
                  'списокКомпетенций': {'п1': 0.25, 'п2': 33, 'п3': 0, 'п4': 0, 'п5': 0, 'п6': 0, 'п7': 2, 'п8': 0,
                                        'п9': 0, 'п10': 2.5, 'п11': 0, 'п12': 5,
                                        'п13': 0, 'п14': 0, 'п15': 0, 'п16': 17, 'п17': 10, 'п18': 0, 'п19': 30,
                                        'п20': 0, 'п21': 0, 'п22': 0, 'п23': 0, 'п24': 10, 'п25': 0, 'п26': 0, 'п27': 0,
                                        'п28': 10, 'п29': 0}},
                 {'Должность': 2, 'КоличествоДолжностей': 29,
                  'списокКомпетенций': {'п1': 0, 'п2': 10, 'п3': 10, 'п4': 2, 'п5': 0, 'п6': 0, 'п7': 29, 'п8': 0,
                                        'п9': 0, 'п10': 0, 'п11': 0, 'п12': 10, 'п13': 0,
                                        'п14': 0, 'п15': 3, 'п16': 4, 'п17': 4, 'п18': 30, 'п19': 10, 'п20': 0,
                                        'п21': 0, 'п22': 30, 'п23': 18, 'п24': 2, 'п25': 0, 'п26': 0, 'п27': 2,
                                        'п28': 0, 'п29': 0}}]

    def train_net(self, comp_list, prof_list, comp_max, prof_max):
        print('MAX VALUES >',comp_max, prof_max)
        scope_net = ml.ScopeNet(comp_max, prof_max)
        (x_train, y_train) = ml.prepare_data(comp_list, prof_list, prof_max, 100)
        ml.train_network(scope_net, x_train, y_train, 100, 1)
        ml.save_network(scope_net)

    def post(self, request, *args, **kwargs):
        json_str = ((self.request.body).decode('utf-8'))
        # json_obj = json.loads(json_str)
        json_obj = self.test_data
        prof_list = []
        comp_list = []
        for e in json_obj:
            prof_id = e['Должность'] - 1
            prof_max = e['КоличествоДолжностей']
            competences = e['списокКомпетенций']
            prof_list.append(prof_id)
            comp_list.append([float(competences[key]) for key in competences])
        self.train_net(comp_list, prof_list, len(comp_list[0]), prof_max)
        response = {'status': 'ok'}
        return JsonResponse(response)

class testView(TemplateView):
    template_name = 'test.html'
