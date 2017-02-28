import numpy as np
import decimal
import random



def get_dec(arr, sig=3):
    ma = max(arr)
    sig3str = '%.3g' % ma
    decstr = str(float(sig3str))[::-1]
    if decstr[0] == '0' and len(decstr) > 4:
        dec = 0
    elif len(decstr) <= 4:
        # print('here')
        dec = 6 - len(decstr)        
    else:
        dec = decstr.find('.')
    # print(ma, decstr, dec)
    nicearr = ['%.3g' % round(float('%.3g' % i), dec) for i in arr]
    # print(nicearr)
    return nicearr

# print(get_dec([3453.43, 3453, 1.242]))
# print(get_dec([1.235, 1.5, 1.242]))
# print(get_dec(np.random.uniform(0, 0.1, 10)))