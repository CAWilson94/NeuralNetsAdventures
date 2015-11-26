'''
Created on 26 Nov 2015
@author: Charlotte Alexandra Wilson
 
Each given file represents one seconds worth of data
Rows: high voltage cycle number (0 - 49) ; power frequency = 50HZ, therefore 1s of data
Columns: different wave position on HV waveform 
There are 64 columns thus, 360 degrees/64 = 5.625 degrees per column

So, if a PD occurred during the second cycle at 10 degrees, it would go in the second row and second column.
Amplitude of the PD pulse is represented by the colour intensity of the pixel(in bmp) or the value of the cell(csv)

Input to neural nets: One of the csv files
There are 64x50 values in the file thus, number of inputs to network = 64x50 = 3200
The number of outputs will be 6 for the 6 different defect types.

'''

