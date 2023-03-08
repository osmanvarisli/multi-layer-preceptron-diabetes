# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 00:27:45 2022

@author: Osman VARIŞLI

Multi Layer Preceptron
"""

import numpy as np
import csv
import copy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
import sys


with open("dataset/diabetes.csv", 'r') as x:
    diabet_data = list(csv.reader(x, delimiter=","))
diabet_data = np.array(diabet_data)
diabet_data = np.delete(diabet_data, 0, axis=0) #başlıkları siliyoruz

diabet_data = np.delete(diabet_data, 6, axis=1) #Insulin siliyoruz
diabet_data = np.delete(diabet_data, 5, axis=1) #BMI siliyoruz
diabet_data = np.delete(diabet_data, 4, axis=1) #DiabetesPedigreeFunction siliyoruz

diabet_data = diabet_data.astype(np.float)

Xa=diabet_data[:,0:8] # illk 9 sutun verileri içermekte
X=preprocessing.scale(Xa)


y=diabet_data[:,-1] #son sutun sonuçları göstermekte

y[y==1] = -1 #tanh fonksiyonuna uygun olmalı için veriler 1 ile -1 arasında olmalıdır.
y[y==0] = 1

#verileri test ve eğtim için %33 %67 oranında ayrılmıştır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

satir, sutun = X_train.shape #verilerin boyutlarını çek

katmanlar=[sutun,4,2,1] #giriş katmanı 5. iki gizli katman. sonuç katmanı 1 olmalı

sinif = np.unique(y_train) #sonuç değerlerini gurupla.
sinif_sayisi = len(sinif)#sonuçta çıkacak değerleri sayısnı göster. 

epok_sayisi=100
bias=1
n=0.06 #öğrenme katsayısı
egitim_dur=0.06

def isnan(value):#değeri boş olanverileri bul 
  try:
      import math
      return math.isnan(float(value))
  except:
      return False


def ileri_yayilim(y,w,bias):
    y_trans=[]
    V1=[]
    y1=[]
    y1.append(y[0])

    for katman in range(len(y)-1):
        y_=np.transpose([np.append(y[katman],bias)])

        V=np.dot(w[katman],y_) #matris çarpımı ile V değerini bulduk

        if isnan(w[katman][0][0]): 
            sys.exit(0)
        y[katman+1]=np.transpose(np.tanh(V))[0] #tanh aktivasyon kodu.
        #print(katman,'  ',y[katman+1])
        
        y_trans.append(y_)
        V1.append(V)
        y1.append(y[katman+1]) 

    return y_trans,V1,y1

def hata_hesapla(x,y): #hata hesapla
    e_list=[]
    toplam=0
    for i in range(len(x)):
        e=(y[i]+x[i])
        e_list.append(e)
        toplam=toplam+e*e
    E=0.5*toplam

    return E,e_list

def geri_yayilim(e_list,V,w,y_trans):
    
    w_=copy.deepcopy(w)
    for e in e_list:
        for i in range(len(w)) :
            t=len(w)-1-i

            #print(w[t])
            if i==0: #birinci adım da ise 
                delta=e*(-1)*aktivasyon_fon_turev(V[t])
                y_trans[t]=np.transpose(y_trans[t])
                w_yeni=w[t]+n*np.dot(delta,y_trans[t])           

            else: #diğer adımlarda
                w_[t+1]=w[t+1][:,:-1] # bias sil
                
                delta=np.dot(
                        np.transpose(w_[t+1]),delta
                        )*aktivasyon_fon_turev(V[t])

                y_trans[t]=np.transpose(y_trans[t])

                w_yeni=w[t]+n*np.dot(delta,y_trans[t])

            if i==3 :sys.exit(0)
            
            w[t]=w_yeni[0]+0.9*(w_yeni[0]-w[t])
       
    return w     
 
def htan(x): #aktivasyon fonk....
  return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def aktivasyon_fon_turev(x):#aktivasyon fonk türevi.....
    #return 1-x**2
    return (1-np.tanh(x)*np.tanh(x)) #tanha turev

def katmanlar_olustur(katmanlar): #katmanlar değeşikeninde belirtildi.... [sutun,4,2,1]
    y=[np.zeros(katmanlar[0])]
    w_t=[]
    for i in range(len(katmanlar)-1):
        #print(katmanlar[i])
        y.append(np.zeros(katmanlar[i+1]))
        w_t.append(np.random.rand(katmanlar[i+1],katmanlar[i]+1))
    
    return y,w_t


def tahminler(x,w,bias): #sonuçların tahimn edilmesi için fonk...
    y,w_sanal=katmanlar_olustur(katmanlar)
    y[0]=x
    y_trans,V,y=ileri_yayilim(y,w,bias)

    if y[len(y)-1][0]>0: ss=-1
    else: ss=1
    
    
    return ss
 
y,w=katmanlar_olustur(katmanlar)

Ortalam_hata=0
for epok_say in range(epok_sayisi):
    print (epok_say, '. epok')
    i=0
    for x in X_train:
        
        sonuc=[y_train[i]]
        #print(sonuc)
        
        i=i+1
        y,w___=katmanlar_olustur(katmanlar)
        y[0]=x
        #print('ileri yayılıma giriyor')
        #print(y)

        y_trans,V,y=ileri_yayilim(y,w,bias)

        E,e_list=hata_hesapla(y[len(y)-1],sonuc)
        Ortalam_hata=Ortalam_hata+E   
        
        w=geri_yayilim(e_list,V,w,y_trans)
    Ortalam_hata=Ortalam_hata/sutun  
    #print(Ortalam_hata)
    if Ortalam_hata<egitim_dur:
        break 


from sklearn.metrics import accuracy_score #bu işlem sonuç doğruluğunu hesaplamak için 

tahmin=[]
sonuc_t=[]
for x, y_sonuc in zip(X_test, y_test):
    t=tahminler(x,w,bias)
    #print(t,int(y_sonuc))
    tahmin.append(t)
    sonuc_t.append(int(y_sonuc))


print('-------------------------------')

print
print('MLP modeli için başarı oranı : ', accuracy_score(sonuc_t, tahmin))
print('-------------------------------')




    
    


