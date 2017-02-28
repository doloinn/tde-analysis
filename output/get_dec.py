# David Lynn
# Helper function to format numbers for plot

# Standard imports
import numpy as np
import decimal
import random

def get_dec(arr, sig=3):
    '''Format numbers as having the same number of places as the largert
    number to three significant digits
    '''
    ma = max(arr)
    sig3str = '%.3g' % ma
    decstr = str(float(sig3str))[::-1]
    if decstr[0] == '0' and len(decstr) > 4:
        dec = 0
    elif len(decstr) <= 4:
        dec = 6 - len(decstr)        
    else:
        dec = decstr.find('.')
    nicearr = ['%.3g' % round(float('%.3g' % i), dec) for i in arr]
    return nicearr
