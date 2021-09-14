from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


# Create your views here.
def home(request):
 return render(request, 'index.html')

@csrf_exempt
def result(request):
  #df = pd.read_csv('iris.csv')
  dataset = pd.read_csv('iris.csv')
 
  y = dataset.species #target coluna alvo
  X = dataset.drop('species', axis=1) #todas ascolunas menos target

#train-test-split
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.5)

#train model
  model = LogisticRegression()
  model.fit(X_train, y_train)

#captura dados de entrada
  val1=request.POST['SepalLengthCm']
  val2=request.POST['SepalWidthCm']
  val3=request.POST['PetalLengthCm']
  val4=request.POST['PetalWidthCm']

#predition
  pred = model.predict([[val1,val2,val3,val4]])
  print(pred)
  
  context = {
 'result': pred
  }
  return render(request, 'result.html',context)