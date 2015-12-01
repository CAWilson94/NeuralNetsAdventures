'''
Created on 30 Nov 2015

@author: Charlotte Alexandra Wilson

Simple file to strip down components of csv for input to machine learning model
i.e. the input has 64x50 rows and columns, but that is not what we need as the 
input to our model, we just need one giant row.
'''

import csv 
cr = csv.reader(open("BC27.L3D.csv","rb"),delimiter = ',',lineterminator = '\n')
#No header line here, so no need to skip anything
