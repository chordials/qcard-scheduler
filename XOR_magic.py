def rsr_variable_length(number, l, n):          # 11100111 rsr 6 len 8
    bottom_n = int('1'*n, 2) & number           # 00100111
    top_n = int(('1'*(l-n))+('0'*n)) & number   # 11000000
    top_n >>= n                                 # 00000011
    top_n &= int('1'*(l-n))
    bottom_n <<= l-n                            # 10011100
    return bottom_n | top_n                     # 10011111