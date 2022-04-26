from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
import pandas as pd 
import os
import joblib
import json
import numpy as np

def home(request):
    print("Rendering Home...")
    if request.method == "POST":
        uploaded_file = request.FILES['csvFile']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name,uploaded_file)
        path = os.path.join(settings.MEDIA_ROOT, name)
        with open(path, newline='') as csvfile:
            data = pd.read_csv(csvfile, sep=',') 
        df = pd.DataFrame(data) 
        cls = joblib.load('KNN Model.sav')
        pred = cls.predict(df)
        ch = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","Amount","Results"]
        pred = pd.Series(pred)
        df['Results'] = pred.values
        json_records = df.reset_index().to_json(orient ='records')
        data = []
        data = json.loads(json_records)
        print(data)
        context = {'d': data, 'column_headers' : ch}
        return  render(request,"home.html",context)
    return  render(request,"home.html")

    



    
