# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import sys
sys.setrecursionlimit(10000)

def insertionSort(alist):
   for index in range(1,len(alist)):

     currentvalue = alist[index]
     position = index

     while position>0 and alist[position-1]>currentvalue:
         alist[position]=alist[position-1]
         position = position-1

     alist[position]=currentvalue

def selectionSort(alist):
   for fillslot in range(len(alist)-1,0,-1):
       positionOfMax=0
       for location in range(1,fillslot+1):
           if alist[location]>alist[positionOfMax]:
               positionOfMax = location

       temp = alist[fillslot]
       alist[fillslot] = alist[positionOfMax]
       alist[positionOfMax] = temp
def quickSort(alist):
   quickSortHelper(alist,0,len(alist)-1)

def quickSortHelper(alist,first,last):
   if first<last:

       splitpoint = partition(alist,first,last)

       quickSortHelper(alist,first,splitpoint-1)
       quickSortHelper(alist,splitpoint+1,last)


def partition(alist,first,last):
   pivotvalue = alist[first]

   leftmark = first+1
   rightmark = last

   done = False
   while not done:

       while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
           leftmark = leftmark + 1

       while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
           rightmark = rightmark -1

       if rightmark < leftmark:
           done = True
       else:
           temp = alist[leftmark]
           alist[leftmark] = alist[rightmark]
           alist[rightmark] = temp

   temp = alist[first]
   alist[first] = alist[rightmark]
   alist[rightmark] = temp


   return rightmark
   
def bubbleSort(alist):
    for passnum in range(len(alist)-1,0,-1):
        for i in range(passnum):
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp





eleman_sayisi=[1000,2000,3000,4000,5000]

def CreateShuffledArray(length):
    a = (np.arange(length))
    shuffle(a)
    return a

    
    
y_equals_n = eleman_sayisi
y_power_two = []
for i in eleman_sayisi:
    y_power_two.append((i * i))
y_nlogn = []
for i in eleman_sayisi:
    y_nlogn.append(i * (math.log(i)))
    

plt.figure(1)
plt.title('y=n')
plt.xlabel('eleman sayısı')
plt.ylabel('karşılık')
plt.plot(eleman_sayisi,y_equals_n)
plt.axis([0,max(eleman_sayisi),0,max(y_equals_n)])
plt.show()

plt.figure(2)
plt.title('y=n^2')
plt.xlabel('eleman sayısı')
plt.ylabel('karşılık')
plt.plot(eleman_sayisi,y_power_two)
plt.axis([0,max(eleman_sayisi),0,int(max(y_power_two))])
plt.show()

plt.figure(3)
plt.title('y=nlogn')
plt.xlabel('eleman sayısı')
plt.ylabel('karşılık')
plt.plot(eleman_sayisi,y_nlogn)
plt.axis([0,max(eleman_sayisi),0,max(y_nlogn)])
plt.show()

bubble_times = []
selection_times = []
insertion_times = []
quick_times = []
for i in eleman_sayisi:
    a1 = CreateShuffledArray(i)
    
    starttime = time.time()
    bubbleSort(a1)
    endtime = time.time()
    bubble_times.append(endtime-starttime)
    
    a1 = CreateShuffledArray(i)
    starttime = time.time()
    selectionSort(a1)
    endtime = time.time()
    selection_times.append(endtime-starttime)
    
    a1 = CreateShuffledArray(i)
    starttime = time.time()
    insertionSort(a1)
    endtime = time.time()
    insertion_times.append(endtime-starttime)
    
    a1 = CreateShuffledArray(i)
    starttime = time.time()
    quickSort(a1)
    endtime = time.time()
    quick_times.append(endtime-starttime)
    
    
e = eleman_sayisi
plt.figure(4)
plt.title('bubble sort')
plt.xlabel('eleman sayısı')
plt.ylabel('zaman')
plt.plot(eleman_sayisi,bubble_times)
plt.axis([0,max(eleman_sayisi),0,max(bubble_times)])
plt.show()

plt.figure(5)
plt.title('insertion sort')
plt.xlabel('eleman sayısı')
plt.ylabel('zaman')
plt.plot(eleman_sayisi,insertion_times)
plt.axis([0,max(eleman_sayisi),0,max(insertion_times)])
plt.show()

plt.figure(6)
plt.title('selection sort')
plt.xlabel('eleman sayısı')
plt.ylabel('zaman')
plt.plot(eleman_sayisi,selection_times)
plt.axis([0,max(eleman_sayisi),0,max(selection_times)])
plt.show()

plt.figure(7)
plt.title('quick sort')
plt.xlabel('eleman sayısı')
plt.ylabel('zaman')
plt.plot(eleman_sayisi,quick_times)
plt.axis([0,max(eleman_sayisi),0,max(quick_times)])
plt.show()

plt.figure(8)
plt.title('y=n, y=n^2, y=nlogn')
plt.xlabel('eleman sayısı')
plt.ylabel('karşılık')
plt.plot(e,y_equals_n, e,y_power_two, e, y_nlogn)
plt.annotate('y=n', xy=(e[len(e)-1], y_equals_n[len(y_equals_n)-1]), xytext=(e[len(e)-1], y_equals_n[len(y_equals_n)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('y=n^2', xy=(e[len(e)-1], y_power_two[len(y_power_two)-1]), xytext=(e[len(e)-1], y_power_two[len(y_power_two)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('y=nlogn', xy=(e[len(e)-1], y_nlogn[len(y_nlogn)-1]), xytext=(e[len(e)-1], y_nlogn[len(y_nlogn)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
merged_y_n = y_equals_n + y_nlogn + y_power_two
plt.axis([0,max(eleman_sayisi),0,max(merged_y_n)])
plt.show()
print("y_equals_n:")
print(y_equals_n)
print("y_power_two:")
print(y_power_two)
print("y_nlogn:")
print(y_nlogn)

plt.figure(9)
plt.title('bubble, insertion, selection, quick')
plt.xlabel('eleman sayısı')
plt.ylabel('zaman')
plt.plot(e,bubble_times, e,insertion_times, e, selection_times,e,quick_times)
plt.annotate('bubble:' + str(bubble_times[len(bubble_times)-1]), xy=(e[len(e)-1], bubble_times[len(bubble_times)-1]), xytext=(e[len(e)-1], bubble_times[len(bubble_times)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('insertion:' + str(insertion_times[len(insertion_times)-1]), xy=(e[len(e)-1], insertion_times[len(insertion_times)-1]), xytext=(e[len(e)-1], insertion_times[len(insertion_times)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('selection:' + str(selection_times[len(selection_times)-1]), xy=(e[len(e)-1], selection_times[len(selection_times)-1]), xytext=(e[len(e)-1], selection_times[len(selection_times)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate('quick:' + str(quick_times[len(quick_times)-1]), xy=(e[len(e)-1], quick_times[len(quick_times)-1]), xytext=(e[len(e)-1], quick_times[len(quick_times)-1]),
            #arrowprops=dict(facecolor='black', shrink=0.05),
            )
merged_times = bubble_times + insertion_times + selection_times + quick_times
plt.axis([0,max(eleman_sayisi),0,max(merged_times)])
plt.show()
print("bubble:")
print(bubble_times)
print("insertion:")
print(insertion_times)
print("selection:")
print(selection_times)
print("quick:")
print(quick_times)
