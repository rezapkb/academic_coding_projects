#!/usr/bin/env python
# coding: utf-8

# # Tests Running : Module

# In[ ]:



from __future__ import print_function

import argparse
import sys


def read_bits_from_file(filename,bigendian):
    bitlist = list()
    if filename == None:
        f = sys.stdin
    else:
        f = open(filename, "rb")
    while True:
        bytes = f.read(16384)
        if bytes:
            for bytech in bytes:
                if sys.version_info > (3,0):
                    byte = bytech
                else:
                    byte = ord(bytech) 
                for i in range(8):
                    if bigendian:
                        bit = (byte & 0x80) >> 7
                        byte = byte << 1
                    else:
                        bit = (byte >> i) & 1
                    bitlist.append(bit)    
        else:
            break
    f.close()
    return bitlist

import argparse
import sys
parser = argparse.ArgumentParser(description='Test data for distinguishability form random, using NIST SP800-22Rev1a algorithms.')
parser.add_argument('filename', type=str, nargs='?', help='Filename of binary file to test')
parser.add_argument('--be', action='store_false',help='Treat data as big endian bits within bytes. Defaults to little endian')
parser.add_argument('-t', '--testname', default=None,help='Select the test to run. Defaults to running all tests. Use --list_tests to see the list')
parser.add_argument('--list_tests', action='store_true',help='Display the list of tests')

args = parser.parse_args()

bigendian = args.be
filename = args.filename

# X 3.1  Frequency (Monobits) Test
# X 3.2  Frequency Test within a Block
# X 3.3  Runs Test
# X 3.4  Test for the Longest Run of Ones in a Block
# X 3.5  Binary Matrix Rank Test
# X 3.6  Discrete Fourier Transform (Specral) Test
# X 3.7  Non-Overlapping Template Matching Test
# X 3.8  Overlapping Template Matching Test
# X 3.9  Maurers Universal Statistical Test
# X 3.10 Linear Complexity Test
# X 3.11 Serial Test
# X 3.12 Approximate Entropy Test
# X 3.13 Cumulative Sums Test
# X 3.14 Random Excursions Test
# X 3.15 Random Excursions Variant Test 


testlist = [
        'monobit_test',
        'frequency_within_block_test',
        'runs_test',
        'longest_run_ones_in_a_block_test',
        'binary_matrix_rank_test',
        'dft_test',
        'non_overlapping_template_matching_test',
        'overlapping_template_matching_test',
        'maurers_universal_test',
        'linear_complexity_test',
        'serial_test',
        'approximate_entropy_test',
        'cumulative_sums_test',
        'random_excursion_test',
        'random_excursion_variant_test']

print("Tests of Distinguishability from Random")
if args.list_tests:
    for i,testname in zip(range(len(testlist)),testlist):
        print(str(i+1).ljust(4)+": "+testname)
    exit()

bits = read_bits_from_file(filename,bigendian)    
gotresult=False
if args.testname:
    if args.testname in testlist:    
        m = __import__ ("sp800_22_"+args.testname)
        func = getattr(m,args.testname)
        print("TEST: %s" % args.testname)
        success,p,plist = func(bits)
        gotresult = True
        if success:
            print("PASS")
        else:
            print("FAIL")
 
        if p:
            print("P="+str(p))

        if plist:
            for pval in plist:
                print("P="+str(pval))
    else:
        print("Test name (%s) not known" % args.ttestname)
        exit()
else:
    results = list()
    
    for testname in testlist:
        print("TEST: %s" % testname)
        m = __import__ ("sp800_22_"+testname)
        func = getattr(m,testname)
        
        (success,p,plist) = func(bits)

        summary_name = testname
        if success:
            print("  PASS")
            summary_result = "PASS"
        else:
            print("  FAIL")
            summary_result = "FAIL"
        
        if p != None:
            print("  P="+str(p))
            summary_p = str(p)
            
        if plist != None:
            for pval in plist:
                print("P="+str(pval))
            summary_p = str(min(plist))
        
        results.append((summary_name,summary_p, summary_result))
        
    print()
    print("SUMMARY")
    print("-------")
    
    for result in results:
        (summary_name,summary_p, summary_result) = result


# # Appendix 1: Throughout this document we provided the underlying codes of tests packed into a module to use more conveniently the RNG TESTS

# # 1_ Approximate Entropy Test

# In[ ]:


from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

def bits_to_int(bits):
    theint = 0
    for i in range(len(bits)):
        theint = (theint << 1) + bits[i]
    return theint
        
def approximate_entropy_test(bits):
    n = len(bits)
    
    m = int(math.floor(math.log(n,2)))-6
    if m < 2:
        m = 2
    if m >3 :
        m = 3
        
    print("  n         = ",n)
    print("  m         = ",m)
    
    Cmi = list()
    phi_m = list()
    for iterm in range(m,m+2):
        # Step 1 
        padded_bits=bits+bits[0:iterm-1]
    
        # Step 2
        counts = list()
        for i in range(2**iterm):
            #print "  Pattern #%d of %d" % (i+1,2**iterm)
            count = 0
            for j in range(n):
                if bits_to_int(padded_bits[j:j+iterm]) == i:
                    count += 1
            counts.append(count)
            print("  Pattern %d of %d, count = %d" % (i+1,2**iterm, count))
    
        # step 3
        Ci = list()
        for i in range(2**iterm):
            Ci.append(float(counts[i])/float(n))
        
        Cmi.append(Ci)
    
        # Step 4
        sum = 0.0
        for i in range(2**iterm):
            if (Ci[i] > 0.0):
                sum += Ci[i]*math.log((Ci[i]/10.0))
        phi_m.append(sum)
        print("  phi(%d)    = %f" % (m,sum))
        
    # Step 5 - let the loop steps 1-4 complete
    
    # Step 6
    appen_m = phi_m[0] - phi_m[1]
    print("  AppEn(%d)  = %f" % (m,appen_m))
    chisq = 2*n*(math.log(2) - appen_m)
    print("  ChiSquare = ",chisq)
    # Step 7
    p = gammaincc(2**(m-1),(chisq/2.0))
    
    success = (p >= 0.01)
    return (success, p, None)

if __name__ == "__main__":
    bits = [1,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,
            1,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,
            0,0,1,0,0,0,0,1,0,1,1,0,1,0,0,0,
            1,1,0,0,0,0,1,0,0,0,1,1,0,1,0,0,
            1,1,0,0,0,1,0,0,1,1,0,0,0,1,1,0,
            0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,1,
            1,0,0,0]
    success, p, _ = approximate_entropy_test(bits)
    
    print("success =",success)
    print("p = ",p)
    


# # Binary Matrix Rank Test

# In[ ]:



from __future__ import print_function

import math
import copy
import gf2matrix

def binary_matrix_rank_test(bits,M=32,Q=32):
    n = len(bits)
    N = int(math.floor(n/(M*Q))) #Number of blocks
    print("  Number of blocks %d" % N)
    print("  Data bits used: %d" % (N*M*Q))
    print("  Data bits discarded: %d" % (n-(N*M*Q))) 
    
    if N < 38:
        print("  Number of blocks must be greater than 37")
        p = 0.0
        return False,p,None
        
    # Compute the reference probabilities for FM, FMM and remainder 
    r = M
    product = 1.0
    for i in range(r):
        upper1 = (1.0 - (2.0**(i-Q)))
        upper2 = (1.0 - (2.0**(i-M)))
        lower = 1-(2.0**(i-r))
        product = product * ((upper1*upper2)/lower)
    FR_prob = product * (2.0**((r*(Q+M-r)) - (M*Q)))
    
    r = M-1
    product = 1.0
    for i in range(r):
        upper1 = (1.0 - (2.0**(i-Q)))
        upper2 = (1.0 - (2.0**(i-M)))
        lower = 1-(2.0**(i-r))
        product = product * ((upper1*upper2)/lower)
    FRM1_prob = product * (2.0**((r*(Q+M-r)) - (M*Q)))
    
    LR_prob = 1.0 - (FR_prob + FRM1_prob)
    
    FM = 0      # Number of full rank matrices
    FMM = 0     # Number of rank -1 matrices
    remainder = 0
    for blknum in range(N):
        block = bits[blknum*(M*Q):(blknum+1)*(M*Q)]
        # Put in a matrix
        matrix = gf2matrix.matrix_from_bits(M,Q,block,blknum) 
        # Compute rank
        rank = gf2matrix.rank(M,Q,matrix,blknum)

        if rank == M: # count the result
            FM += 1
        elif rank == M-1:
            FMM += 1  
        else:
            remainder += 1

    chisq =  (((FM-(FR_prob*N))**2)/(FR_prob*N))
    chisq += (((FMM-(FRM1_prob*N))**2)/(FRM1_prob*N))
    chisq += (((remainder-(LR_prob*N))**2)/(LR_prob*N))
    p = math.e **(-chisq/2.0)
    success = (p >= 0.01)
    
    print("  Full Rank Count  = ",FM)
    print("  Full Rank -1 Count = ",FMM)
    print("  Remainder Count = ",remainder) 
    print("  Chi-Square = ",chisq)

    return (success, p, None)


# # 3_ Commulative Sum Test

# In[ ]:




from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *
#import scipy.stats

def normcdf(n):
    return 0.5 * math.erfc(-n * math.sqrt(0.5))

def p_value(n,z):
    sum_a = 0.0
    startk = int(math.floor((((float(-n)/z)+1.0)/4.0)))
    endk   = int(math.floor((((float(n)/z)-1.0)/4.0)))
    for k in range(startk,endk+1):
        c = (((4.0*k)+1.0)*z)/math.sqrt(n)
        #d = scipy.stats.norm.cdf(c)
        d = normcdf(c)
        c = (((4.0*k)-1.0)*z)/math.sqrt(n)
        #e = scipy.stats.norm.cdf(c)
        e = normcdf(c)
        sum_a = sum_a + d - e

    sum_b = 0.0
    startk = int(math.floor((((float(-n)/z)-3.0)/4.0)))
    endk   = int(math.floor((((float(n)/z)-1.0)/4.0)))
    for k in range(startk,endk+1):
        c = (((4.0*k)+3.0)*z)/math.sqrt(n)
        #d = scipy.stats.norm.cdf(c)
        d = normcdf(c)
        c = (((4.0*k)+1.0)*z)/math.sqrt(n)
        #e = scipy.stats.norm.cdf(c)
        e = normcdf(c)
        sum_b = sum_b + d - e 

    p = 1.0 - sum_a + sum_b
    return p
    
def cumulative_sums_test(bits):
    n = len(bits)
    # Step 1
    x = list()             # Convert to +1,-1
    for bit in bits:
        #if bit == 0:
        x.append((bit*2)-1)
        
    # Steps 2 and 3 Combined
    # Compute the partial sum and records the largest excursion.
    pos = 0
    forward_max = 0
    for e in x:
        pos = pos+e
        if abs(pos) > forward_max:
            forward_max = abs(pos)
    pos = 0
    backward_max = 0
    for e in reversed(x):
        pos = pos+e
        if abs(pos) > backward_max:
            backward_max = abs(pos)
     
    # Step 4
    p_forward  = p_value(n, forward_max)
    p_backward = p_value(n,backward_max)
    
    success = ((p_forward >= 0.01) and (p_backward >= 0.01))
    plist = [p_forward, p_backward]

    if success:
        print("PASS")
    else:    
        print("FAIL: Data not random")
    return (success, None, plist)

if __name__ == "__main__":
    bits = [1,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,0,1,
            1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,
            0,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,1,
            0,1,0,0,1,1,0,0,0,1,0,0,1,1,0,0,0,1,1,0,
            0,1,1,0,0,0,1,0,1,0,0,0,1,0,1,1,1,0,0,0]
    success, _, plist = cumulative_sums_test(bits)
    
    print("success =",success)
    print("plist = ",plist)


# # 4_ DFT Test

# In[ ]:




from __future__ import print_function

import math
import numpy
import sys

def dft_test(bits):
    n = len(bits)
    if (n % 2) == 1:        # Make it an even number
        bits = bits[:-1]

    ts = list()             # Convert to +1,-1
    for bit in bits:
        ts.append((bit*2)-1)

    ts_np = numpy.array(ts)
    fs = numpy.fft.fft(ts_np)  # Compute DFT
   
    if sys.version_info > (3,0):
        mags = abs(fs)[:n//2] # Compute magnitudes of first half of sequence
    else:
        mags = abs(fs)[:n/2] # Compute magnitudes of first half of sequence
    
    T = math.sqrt(math.log(1.0/0.05)*n) # Compute upper threshold
    N0 = 0.95*n/2.0
    print("  N0 = %f" % N0)

    N1 = 0.0   # Count the peaks above the upper theshold
    for mag in mags:
        if mag < T:
            N1 += 1.0
    print("  N1 = %f" % N1)
    d = (N1 - N0)/math.sqrt((n*0.95*0.05)/4) # Compute the P value
    p = math.erfc(abs(d)/math.sqrt(2))

    success = (p >= 0.01)
    return (success,p,None)


# # 5_ Frequency Within Block Test

# In[ ]:




from __future__ import print_function

import math
from fractions import Fraction
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

#ones_table = [bin(i)[2:].count('1') for i in range(256)]
def count_ones_zeroes(bits):
    ones = 0
    zeroes = 0
    for bit in bits:
        if (bit == 1):
            ones += 1
        else:
            zeroes += 1
    return (zeroes,ones)

def frequency_within_block_test(bits):
    # Compute number of blocks M = block size. N=num of blocks
    # N = floor(n/M)
    # miniumum block size 20 bits, most blocks 100
    n = len(bits)
    M = 20
    N = int(math.floor(n/M))
    if N > 99:
        N=99
        M = int(math.floor(n/N))
    
    if len(bits) < 100:
        print("Too little data for test. Supply at least 100 bits")
        return False,1.0,None
    
    print("  n = %d" % len(bits))
    print("  N = %d" % N)
    print("  M = %d" % M)
    
    num_of_blocks = N
    block_size = M #int(math.floor(len(bits)/num_of_blocks))
    #n = int(block_size * num_of_blocks)
    
    proportions = list()
    for i in range(num_of_blocks):
        block = bits[i*(block_size):((i+1)*(block_size))]
        zeroes,ones = count_ones_zeroes(block)
        proportions.append(Fraction(ones,block_size))

    chisq = 0.0
    for prop in proportions:
        chisq += 4.0*block_size*((prop - Fraction(1,2))**2)
    
    p = gammaincc((num_of_blocks/2.0),float(chisq)/2.0)
    success = (p >= 0.01)
    return (success,p,None)


# # 6_ Linear Complexity Test

# In[ ]:



 
from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

def berelekamp_massey(bits):
    n = len(bits)
    b = [0 for x in bits]  #initialize b and c arrays
    c = [0 for x in bits]
    b[0] = 1
    c[0] = 1
    
    L = 0
    m = -1
    N = 0
    while (N < n):
        #compute discrepancy
        d = bits[N]
        for i in range(1,L+1):
            d = d ^ (c[i] & bits[N-i])
        if (d != 0):  # If d is not zero, adjust poly
            t = c[:]
            for i in range(0,n-N+m):
                c[N-m+i] = c[N-m+i] ^ b[i] 
            if (L <= (N/2)):
                L = N + 1 - L
                m = N
                b = t 
        N = N +1
    # Return length of generator and the polynomial
    return L , c[0:L]
    
def linear_complexity_test(bits,patternlen=None):
    n = len(bits)
    # Step 1. Choose the block size
    if patternlen != None:
        M = patternlen  
    else: 
        if n < 1000000:
            print("Error. Need at least 10^6 bits")
            #exit()
            return False,0.0,None
        M = 512
    K = 6 
    N = int(math.floor(n/M))
    print("  M = ", M)
    print("  N = ", N)
    print("  K = ", K)    
    
    # Step 2 Compute the linear complexity of the blocks
    LC = list()
    for i in range(N):
        x = bits[(i*M):((i+1)*M)]
        LC.append(berelekamp_massey(x)[0])
    
    # Step 3 Compute mean
    a = float(M)/2.0
    b = ((((-1)**(M+1))+9.0))/36.0
    c = ((M/3.0) + (2.0/9.0))/(2**M)
    mu =  a+b-c
    
    T = list()
    for i in range(N):
        x = ((-1.0)**M) * (LC[i] - mu) + (2.0/9.0)
        T.append(x)
        
    # Step 4 Count the distribution over Ticket
    v = [0,0,0,0,0,0,0]
    for t in T:
        if t <= -2.5:
            v[0] += 1
        elif t <= -1.5:
            v[1] += 1
        elif t <= -0.5:
            v[2] += 1
        elif t <= 0.5:
            v[3] += 1
        elif t <= 1.5:
            v[4] += 1
        elif t <= 2.5:
            v[5] += 1            
        else:
            v[6] += 1

    # Step 5 Compute Chi Square Statistic
    pi = [0.010417,0.03125,0.125,0.5,0.25,0.0625,0.020833]
    chisq = 0.0
    for i in range(K+1):
        chisq += ((v[i] - (N*pi[i]))**2.0)/(N*pi[i])
    print("  chisq = ",chisq)
    # Step 6 Compute P Value
    P = gammaincc((K/2.0),(chisq/2.0))
    print("  P = ",P)
    success = (P >= 0.01)
    return (success, P, None)
    
if __name__ == "__main__":
    bits = [1,1,0,1,0,1,1,1,1,0,0,0,1]
    L,poly = berelekamp_massey(bits)

    bits = [1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,
            0,1,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,0,1,0,1,1,1,1,
            0,0,0,1,1,1,0,1,0,1,1,1,1,0,0,0,1]
    success,p,_ = linear_complexity_test(bits,patternlen=7)
    
    print("L =",L)
    print("p = ",p)
       


# # 7_ Longest Run in a Block Test

# In[ ]:




from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

import random

def probs(K,M,i):
    M8 =      [0.2148, 0.3672, 0.2305, 0.1875]
    M128 =    [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    M512 =    [0.1170, 0.2460, 0.2523, 0.1755, 0.1027, 0.1124]
    M1000 =   [0.1307, 0.2437, 0.2452, 0.1714, 0.1002, 0.1088]
    M10000 =  [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
    if (M == 8):        return M8[i]
    elif (M == 128):    return M128[i]
    elif (M == 512):    return M512[i]
    elif (M == 1000):   return M1000[i]
    else:               return M10000[i]

def longest_run_ones_in_a_block_test(bits):
    n = len(bits)

    if n < 128:
        return (False,1.0,None)
    elif n<6272:
        M = 8
    elif n<750000:
        M = 128
    else:
        M = 10000
            
    # compute new values for K & N
    if M==8:
        K=3
        N=16
    elif M==128:
        K=5
        N=49
    else:
        K=6
        N=75
        
    # Table of frequencies
    v = [0,0,0,0,0,0,0]

    for i in range(N): # over each block
        #find longest run
        block = bits[i*M:((i+1)*M)] # Block i
        
        run = 0
        longest = 0
        for j in range(M): # Count the bits.
            if block[j] == 1:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0

        if M == 8:
            if longest <= 1:    v[0] += 1
            elif longest == 2:  v[1] += 1
            elif longest == 3:  v[2] += 1
            else:               v[3] += 1
        elif M == 128:
            if longest <= 4:    v[0] += 1
            elif longest == 5:  v[1] += 1
            elif longest == 6:  v[2] += 1
            elif longest == 7:  v[3] += 1
            elif longest == 8:  v[4] += 1
            else:               v[5] += 1
        else:
            if longest <= 10:   v[0] += 1
            elif longest == 11: v[1] += 1
            elif longest == 12: v[2] += 1
            elif longest == 13: v[3] += 1
            elif longest == 14: v[4] += 1
            elif longest == 15: v[5] += 1
            else:               v[6] += 1
    
    # Compute Chi-Sq
    chi_sq = 0.0
    for i in range(K+1):
        p_i = probs(K,M,i)
        upper = (v[i] - N*p_i)**2
        lower = N*p_i
        chi_sq += upper/lower
    print("  n = "+str(n))
    print("  K = "+str(K))
    print("  M = "+str(M))
    print("  N = "+str(N))
    print("  chi_sq = "+str(chi_sq))
    p = gammaincc(K/2.0, chi_sq/2.0)
    
    success = (p >= 0.01)
    return (success,p,None)


# # 8_ Maurer Universal Test

# In[ ]:




from __future__ import print_function

import math

def pattern2int(pattern):
    l = len(pattern)
    n = 0
    for bit in (pattern):
        n = (n << 1) + bit
    return n          
         
def maurers_universal_test(bits,patternlen=None, initblocks=None):
    n = len(bits)

    # Step 1. Choose the block size
    if patternlen != None:
        L = patternlen  
    else: 
        ns = [904960,2068480,4654080,10342400,
              22753280,49643520,107560960,
              231669760,496435200,1059061760]
        L = 6
        if n < 387840:
            print("Error. Need at least 387840 bits. Got %d." % n)
            #exit()
            return False,0.0,None
        for threshold in ns:
            if n >= threshold:
                L += 1 

    # Step 2 Split the data into Q and K blocks
    nblocks = int(math.floor(n/L))
    if initblocks != None:
        Q = initblocks
    else:
        Q = 10*(2**L)
    K = nblocks - Q
    
    # Step 3 Construct Table
    nsymbols = (2**L)
    T=[0 for x in range(nsymbols)] # zero out the table
    for i in range(Q):             # Mark final position of
        pattern = bits[i*L:(i+1)*L] # each pattern
        idx = pattern2int(pattern)
        T[idx]=i+1      # +1 to number indexes 1..(2**L)+1
                        # instead of 0..2**L
    # Step 4 Iterate
    sum = 0.0
    for i in range(Q,nblocks):
        pattern = bits[i*L:(i+1)*L]
        j = pattern2int(pattern)
        dist = i+1-T[j]
        T[j] = i+1
        sum = sum + math.log(dist,2)
    print("  sum =", sum)
    
    # Step 5 Compute the test statistic
    fn = sum/K
    print("  fn =",fn)
       
    # Step 6 Compute the P Value
    # Tables from https://static.aminer.org/pdf/PDF/000/120/333/
    # a_universal_statistical_test_for_random_bit_generators.pdf
    ev_table =  [0,0.73264948,1.5374383,2.40160681,3.31122472,
                 4.25342659,5.2177052,6.1962507,7.1836656,
                 8.1764248,9.1723243,10.170032,11.168765,
                 12.168070,13.167693,14.167488,15.167379]
    var_table = [0,0.690,1.338,1.901,2.358,2.705,2.954,3.125,
                 3.238,3.311,3.356,3.384,3.401,3.410,3.416,
                 3.419,3.421]
                 
    # sigma = math.sqrt(var_table[L])
    mag = abs((fn - ev_table[L]) / ((0.7 - 0.8 / L + (4 + 32 / L) * (pow(K, -3 / L)) / 15) * (math.sqrt(var_table[L] / K)) * math.sqrt(2)))
    P = math.erfc(mag)

    success = (P >= 0.01)
    return (success, P, None)
    

if __name__ == "__main__":
    bits = [0,1,0,1,1,0,1,0,0,1,1,1,0,1,0,1,0,1,1,1]
    success, p, _ = maurers_universal_test(bits, patternlen=2, initblocks=4)
    
    print("success =",success)
    print("p       = ",p)
    


# # 9_ Monobit Test

# In[ ]:


from __future__ import print_function

import math

def count_ones_zeroes(bits):
    ones = 0
    zeroes = 0
    for bit in bits:
        if (bit == 1):
            ones += 1
        else:
            zeroes += 1
    return (zeroes,ones)

def monobit_test(bits):
    n = len(bits)
    
    zeroes,ones = count_ones_zeroes(bits)
    s = abs(ones-zeroes)
    print("  Ones count   = %d" % ones)
    print("  Zeroes count = %d" % zeroes)
    
    p = math.erfc(float(s)/(math.sqrt(float(n)) * math.sqrt(2.0)))
    
    success = (p >= 0.01)
    return (success,p,None)
    


# # 10_ NonOverlapping Template Matching Test

# In[ ]:


from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *
import random

def non_overlapping_template_matching_test(bits):
    # The templates provdided in SP800-22rev1a
    templates = [None for x in range(7)]
    templates[0] = [[0,1],[1,0]]
    templates[1] = [[0,0,1],[0,1,1],[1,0,0],[1,1,0]]
    templates[2] = [[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,0,0,0],[1,1,0,0],[1,1,1,0]]
    templates[3] = [[0,0,0,0,1],[0,0,0,1,1],[0,0,1,0,1],[0,1,0,1,1],[0,0,1,1,1],[0,1,1,1,1],
                    [1,1,1,0,0],[1,1,0,1,0],[1,0,1,0,0],[1,1,0,0,0],[1,0,0,0,0],[1,1,1,1,0]]
    templates[4] = [[0,0,0,0,0,1],[0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,1],[0,0,1,0,1,1],
                    [0,0,1,1,0,1],[0,0,1,1,1,1],[0,1,0,0,1,1],
                    [0,1,0,1,1,1],[0,1,1,1,1,1],[1,0,0,0,0,0],
                    [1,0,1,0,0,0],[1,0,1,1,0,0],[1,1,0,0,0,0],
                    [1,1,0,0,1,0],[1,1,0,1,0,0],[1,1,1,0,0,0],
                    [1,1,1,0,1,0],[1,1,1,1,0,0],[1,1,1,1,1,0]]
    templates[5] = [[0,0,0,0,0,0,1],[0,0,0,0,0,1,1],[0,0,0,0,1,0,1],[0,0,0,0,1,1,1],
                    [0,0,0,1,0,0,1],[0,0,0,1,0,1,1],[0,0,0,1,1,0,1],[0,0,0,1,1,1,1],
                    [0,0,1,0,0,1,1],[0,0,1,0,1,0,1],[0,0,1,0,1,1,1],[0,0,1,1,0,1,1],
                    [0,0,1,1,1,0,1],[0,0,1,1,1,1,1],[0,1,0,0,0,1,1],[0,1,0,0,1,1,1],
                    [0,1,0,1,0,1,1],[0,1,0,1,1,1,1],[0,1,1,0,1,1,1],[0,1,1,1,1,1,1],
                    [1,0,0,0,0,0,0],[1,0,0,1,0,0,0],[1,0,1,0,0,0,0],[1,0,1,0,1,0,0],
                    [1,0,1,1,0,0,0],[1,0,1,1,1,0,0],[1,1,0,0,0,0,0],[1,1,0,0,0,1,0],
                    [1,1,0,0,1,0,0],[1,1,0,1,0,0,0],[1,1,0,1,0,1,0],[1,1,0,1,1,0,0],
                    [1,1,1,0,0,0,0],[1,1,1,0,0,1,0],[1,1,1,0,1,0,0],[1,1,1,0,1,1,0],
                    [1,1,1,1,0,0,0],[1,1,1,1,0,1,0],[1,1,1,1,1,0,0],[1,1,1,1,1,1,0]]
    templates[6] = [[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,1,0,1],[0,0,0,0,0,1,1,1],
                    [0,0,0,0,1,0,0,1],[0,0,0,0,1,0,1,1],[0,0,0,0,1,1,0,1],[0,0,0,0,1,1,1,1],
                    [0,0,0,1,0,0,1,1],[0,0,0,1,0,1,0,1],[0,0,0,1,0,1,1,1],[0,0,0,1,1,0,0,1],
                    [0,0,0,1,1,0,1,1],[0,0,0,1,1,1,0,1],[0,0,0,1,1,1,1,1],[0,0,1,0,0,0,1,1],
                    [0,0,1,0,0,1,0,1],[0,0,1,0,0,1,1,1],[0,0,1,0,1,0,1,1],[0,0,1,0,1,1,0,1],
                    [0,0,1,0,1,1,1,1],[0,0,1,1,0,1,0,1],[0,0,1,1,0,1,1,1],[0,0,1,1,1,0,1,1],
                    [0,0,1,1,1,1,0,1],[0,0,1,1,1,1,1,1],[0,1,0,0,0,0,1,1],[0,1,0,0,0,1,1,1],
                    [0,1,0,0,1,0,1,1],[0,1,0,0,1,1,1,1],[0,1,0,1,0,0,1,1],[0,1,0,1,0,1,1,1],
                    [0,1,0,1,1,0,1,1],[0,1,0,1,1,1,1,1],[0,1,1,0,0,1,1,1],[0,1,1,0,1,1,1,1],
                    [0,1,1,1,1,1,1,1],[1,0,0,0,0,0,0,0],[1,0,0,1,0,0,0,0],[1,0,0,1,1,0,0,0],
                    [1,0,1,0,0,0,0,0],[1,0,1,0,0,1,0,0],[1,0,1,0,1,0,0,0],[1,0,1,0,1,1,0,0],
                    [1,0,1,1,0,0,0,0],[1,0,1,1,0,1,0,0],[1,0,1,1,1,0,0,0],[1,0,1,1,1,1,0,0],
                    [1,1,0,0,0,0,0,0],[1,1,0,0,0,0,1,0],[1,1,0,0,0,1,0,0],[1,1,0,0,1,0,0,0],
                    [1,1,0,0,1,0,1,0],[1,1,0,1,0,0,0,0],[1,1,0,1,0,0,1,0],[1,1,0,1,0,1,0,0],
                    [1,1,0,1,1,0,0,0],[1,1,0,1,1,0,1,0],[1,1,0,1,1,1,0,0],[1,1,1,0,0,0,0,0],
                    [1,1,1,0,0,0,1,0],[1,1,1,0,0,1,0,0],[1,1,1,0,0,1,1,0],[1,1,1,0,1,0,0,0],
                    [1,1,1,0,1,0,1,0],[1,1,1,0,1,1,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,1,0],
                    [1,1,1,1,0,1,0,0],[1,1,1,1,0,1,1,0],[1,1,1,1,1,0,0,0],[1,1,1,1,1,0,1,0],
                    [1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0]]
    
    n = len(bits)
    
    # Choose the template B
    r = random.SystemRandom()
    template_list = r.choice(templates)
    B = r.choice(template_list)
    
    m = len(B)
    
    N = 8
    M = int(math.floor(len(bits)/8))
    n = M*N
    
    blocks = list() # Split into N blocks of M bits
    for i in range(N):
        blocks.append(bits[i*M:(i+1)*M])

    W=list() # Count the number of matches of the template in each block Wj
    for block in blocks:
        position = 0
        count = 0
        while position < (M-m):
            if block[position:position+m] == B:
                position += m
                count += 1
            else:
                position += 1
        W.append(count)

    mu = float(M-m+1)/float(2**m) # Compute mu and sigma
    sigma = M * ((1.0/float(2**m))-(float((2*m)-1)/float(2**(2*m))))

    chisq = 0.0  # Compute Chi-Square
    for j in range(N):
        chisq += ((W[j] - mu)**2)/(sigma**2)

    p = gammaincc(N/2.0, chisq/2.0) # Compute P value

    success = ( p >= 0.01)
    return (success,p,None)


# # 11_ Overlapping Template Matching Test

# In[ ]:


from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

def lgamma(x):
    return math.log(gamma(x))
    
def Pr(u, eta):
    if ( u == 0 ):
        p = math.exp(-eta)
    else:
        sum = 0.0
        for l in range(1,u+1):
            sum += math.exp(-eta-u*math.log(2)+l*math.log(eta)-lgamma(l+1)+lgamma(u)-lgamma(l)-lgamma(u-l+1))
        p = sum
    return p

def overlapping_template_matching_test(bits,blen=6):
    n = len(bits)
    
    m = 10
    # Build the template B as a random list of bits
    B = [1 for x in range(m)]
    
    N = 968
    K = 5
    M = 1062
    if len(bits) < (M*N):
        print("Insufficient data. %d bit provided. 1,028,016 bits required" % len(bits))
        return False, 0.0, None
    
    blocks = list() # Split into N blocks of M bits
    for i in range(N):
        blocks.append(bits[i*M:(i+1)*M])

    # Count the distribution of matches of the template across blocks: Vj
    v=[0 for x in range(K+1)] 
    for block in blocks:
        count = 0
        for position in range(M-m):
            if block[position:position+m] == B:
                count += 1
            
        if count >= (K):
            v[K] += 1
        else:
            v[count] += 1

    #lamd = float(M-m+1)/float(2**m) # Compute lambda and nu
    #nu = lamd/2.0

    chisq = 0.0  # Compute Chi-Square
    #pi = [0.324652,0.182617,0.142670,0.106645,0.077147,0.166269] # From spec
    pi = [0.364091, 0.185659, 0.139381, 0.100571, 0.0704323, 0.139865] # From STS
    piqty = [int(x*N) for x in pi]
    
    lambd = (M-m+1.0)/(2.0**m)
    eta = lambd/2.0
    sum = 0.0
    for i in range(K): #  Compute Probabilities
        pi[i] = Pr(i, eta)
        sum += pi[i]

    pi[K] = 1 - sum;

    #for block in blocks:
    #    count = 0
    #    for j in xrange(M-m+1):
    #        if B == block[j:j+m]:
    #            count += 1
    #    if ( count <= 4 ):
    #        v[count]+= 1
    #    else:
    #        v[K]+=1

    sum = 0    
    chisq = 0.0
    for i in range(K+1):
        chisq += ((v[i] - (N*pi[i]))**2)/(N*pi[i])
        sum += v[i]
        
    p = gammaincc(5.0/2.0, chisq/2.0) # Compute P value

    print("  B = ",B)
    print("  m = ",m)
    print("  M = ",M)
    print("  N = ",N)
    print("  K = ",K)
    print("  model = ",piqty)
    print("  v[j] =  ",v) 
    print("  chisq = ",chisq)
    
    success = ( p >= 0.01)
    return (success,p,None)


# # 12_ Random Excursion Test

# In[ ]:


from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

# RANDOM EXCURSION TEST
def random_excursion_test(bits):
    n = len(bits)

    x = list()             # Convert to +1,-1
    for bit in bits:
        #if bit == 0:
        x.append((bit*2)-1)

    #print "x=",x
    # Build the partial sums
    pos = 0
    s = list()
    for e in x:
        pos = pos+e
        s.append(pos)    
    sprime = [0]+s+[0] # Add 0 on each end
    
    #print "sprime=",sprime
    # Build the list of cycles
    pos = 1
    cycles = list()
    while (pos < len(sprime)):
        cycle = list()
        cycle.append(0)
        while sprime[pos]!=0:
            cycle.append(sprime[pos])
            pos += 1
        cycle.append(0)
        cycles.append(cycle)
        pos = pos + 1
    
    J = len(cycles)
    print("J="+str(J))    
    
    vxk = [['a','b','c','d','e','f'] for y in [-4,-3,-2,-1,1,2,3,4] ]

    # Count Occurances  
    for k in range(6):
        for index in range(8):
            mapping = [-4,-3,-2,-1,1,2,3,4]
            x = mapping[index]
            cyclecount = 0
            #count how many cycles in which x occurs k times
            for cycle in cycles:
                oc = 0
                #Count how many times x occurs in the current cycle
                for pos in cycle:
                    if (pos == x):
                        oc += 1
                # If x occurs k times, increment the cycle count
                if (k < 5):
                    if oc == k:
                        cyclecount += 1
                else:
                    if k == 5:
                        if oc >=5:
                            cyclecount += 1
            vxk[index][k] = cyclecount
    
    # Table for reference random probabilities
    pixk=[[0.5     ,0.25   ,0.125  ,0.0625  ,0.0312 ,0.0312],
          [0.75    ,0.0625 ,0.0469 ,0.0352  ,0.0264 ,0.0791],
          [0.8333  ,0.0278 ,0.0231 ,0.0193  ,0.0161 ,0.0804],
          [0.875   ,0.0156 ,0.0137 ,0.012   ,0.0105 ,0.0733],
          [0.9     ,0.01   ,0.009  ,0.0081  ,0.0073 ,0.0656],
          [0.9167  ,0.0069 ,0.0064 ,0.0058  ,0.0053 ,0.0588],
          [0.9286  ,0.0051 ,0.0047 ,0.0044  ,0.0041 ,0.0531]]
    
    success = True
    plist = list()
    for index in range(8):
        mapping = [-4,-3,-2,-1,1,2,3,4]
        x = mapping[index]
        chisq = 0.0
        for k in range(6):
            top = float(vxk[index][k]) - (float(J) * (pixk[abs(x)-1][k]))
            top = top*top
            bottom = J * pixk[abs(x)-1][k]
            chisq += top/bottom
        p = gammaincc(5.0/2.0,chisq/2.0)
        plist.append(p)
        if p < 0.01:
            err = " Not Random"
            success = False
        else:
            err = ""
        print("x = %1.0f\tchisq = %f\tp = %f %s"  % (x,chisq,p,err))
    if (J < 500):
        print("J too small (J < 500) for result to be reliable")
    elif success:
        print("PASS")
    else:    
        print("FAIL: Data not random")
    return (success, None, plist)

if __name__ == "__main__":
    bits = [0,1,1,0,1,1,0,1,0,1]
    success, _, plist = random_excursion_test(bits)
    
    print("success =",success)
    print("plist = ",plist)


# # 13_ Random Excursion Variant Test

# In[ ]:



from __future__ import print_function

import math

# RANDOM EXCURSION VARIANT TEST
def random_excursion_variant_test(bits):
    n = len(bits)

    x = list()             # Convert to +1,-1
    for bit in bits:
        x.append((bit * 2)-1)

    # Build the partial sums
    pos = 0
    s = list()
    for e in x:
        pos = pos+e
        s.append(pos)    
    sprime = [0]+s+[0] # Add 0 on each end

    # Count the number of cycles J
    J = 0
    for value in sprime[1:]:
        if value == 0:
            J += 1
    print("J=",J)
    # Build the counts of offsets
    count = [0 for x in range(-9,10)]
    for value in sprime:
        if (abs(value) < 10):
            count[value] += 1

    # Compute P values
    success = True
    plist = list()
    for x in range(-9,10):
        if x != 0:
            top = abs(count[x]-J)
            bottom = math.sqrt(2.0 * J *((4.0*abs(x))-2.0))
            p = math.erfc(top/bottom)
            plist.append(p)
            if p < 0.01:
                err = " Not Random"
                success = False
            else:
                err = ""
            print("x = %1.0f\t count=%d\tp = %f %s"  % (x,count[x],p,err))
            
    if (J < 500):
        print("J too small (J=%d < 500) for result to be reliable" % J)
    elif success:
        print("PASS")
    else:    
        print("FAIL: Data not random")
    return (success,None,plist)


# # 14_ Runs Test

# In[ ]:


from __future__ import print_function

import math
from fractions import Fraction
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *
import numpy
import cmath
import random

#ones_table = [bin(i)[2:].count('1') for i in range(256)]
def count_ones_zeroes(bits):
    ones = 0
    zeroes = 0
    for bit in bits:
        if (bit == 1):
            ones += 1
        else:
            zeroes += 1
    return (zeroes,ones)

def runs_test(bits):
    n = len(bits)
    zeroes,ones = count_ones_zeroes(bits)

    prop = float(ones)/float(n)
    print("  prop ",prop)

    tau = 2.0/math.sqrt(n)
    print("  tau ",tau)

    if abs(prop-0.5) > tau:
        return (False,0.0,None)

    vobs = 1.0
    for i in range(n-1):
        if bits[i] != bits[i+1]:
            vobs += 1.0

    print("  vobs ",vobs)
      
    p = math.erfc(abs(vobs - (2.0*n*prop*(1.0-prop)))/(2.0*math.sqrt(2.0*n)*prop*(1-prop) ))
    success = (p >= 0.01)
    return (success,p,None)


# # 15_ Serial Test

# In[ ]:



from __future__ import print_function

import math
#from scipy.special import gamma, gammainc, gammaincc
from gamma_functions import *

def int2patt(n,m):
    pattern = list()
    for i in range(m):
        pattern.append((n >> i) & 1)
    return pattern
    
def countpattern(patt,bits,n):
    thecount = 0
    for i in range(n):
        match = True
        for j in range(len(patt)):
            if patt[j] != bits[i+j]:
                match = False
        if match:
            thecount += 1
    return thecount

def psi_sq_mv1(m, n, padded_bits):
    counts = [0 for i in range(2**m)] 
    for i in range(2**m):
        pattern = int2patt(i,m)
        count = countpattern(pattern,padded_bits,n)
        counts.append(count)
        
    psi_sq_m = 0.0
    for count in counts: 
        psi_sq_m += (count**2)
    psi_sq_m = psi_sq_m * (2**m)/n 
    psi_sq_m -= n
    return psi_sq_m            
         
def serial_test(bits,patternlen=None):
    n = len(bits)
    if patternlen != None:
        m = patternlen  
    else:  
        m = int(math.floor(math.log(n,2)))-2
    
        if m < 4:
            print("Error. Not enough data for m to be 4")
            return False,0,None
        m = 4
    
    # Step 1
    padded_bits=bits+bits[0:m-1]
    
    # Step 2
    psi_sq_m   = psi_sq_mv1(m, n, padded_bits)
    psi_sq_mm1 = psi_sq_mv1(m-1, n, padded_bits)
    psi_sq_mm2 = psi_sq_mv1(m-2, n, padded_bits)    
    
    delta1 = psi_sq_m - psi_sq_mm1
    delta2 = psi_sq_m - (2*psi_sq_mm1) + psi_sq_mm2
    
    P1 = gammaincc(2**(m-2),delta1/2.0)
    P2 = gammaincc(2**(m-3),delta2/2.0)
        
    print("  psi_sq_m   = ",psi_sq_m)
    print("  psi_sq_mm1 = ",psi_sq_mm1)
    print("  psi_sq_mm2 = ",psi_sq_mm2)
    print("  delta1     = ",delta1)
    print("  delta2     = ",delta2)  
    print("  P1         = ",P1)
    print("  P2         = ",P2)
     
    success = (P1 >= 0.01) and (P2 >= 0.01)
    return (success, None, [P1,P2])

if __name__ == "__main__":
    bits = [0,0,1,1,0,1,1,1,0,1]
    success, _, plist = serial_test(bits, patternlen=3)
    
    print("success =",success)
    print("plist = ",plist)
    


# # Appendix 2 : Review of RNG & Matrix Transformation

# # 1_ RNG - Rejection Method : Gamma Function random number generator

# In[ ]:


from math import gamma,e

# Continued Fraction Computation
# 6.5.31 Handbook of Mathematical Functions, page 263
#    Recursive implementation
def upper_incomplete_gamma(a,x,d=0,iterations=100):
    if d == iterations:
        if ((d % 2) == 1):
            return 1.0 # end iterations
        else:
            m = d/2
            return x + (m-a)
    if d == 0:
        try:
            result = ((x**a) * (e**(-x)))/upper_incomplete_gamma(a,x,d=d+1)
        except OverflowError:
            result = 0.0
        return result
    elif ((d % 2) == 1):
        m = 1.0+((d-1.0)/2.0)
        return x+ ((m-a)/(upper_incomplete_gamma(a,x,d=d+1)))
    else:
        m = d/2
        return 1+(m/(upper_incomplete_gamma(a,x,d=d+1)))

# 6.5.31 Handbook of Mathematical Functions, page 263
#    Recursive implementation
def upper_incomplete_gamma2(a,x,d=0,iterations=100):
    if d == iterations:
        return 1.0 
    if d == 0:
        result = ((x**a) * (e**(-x)))/upper_incomplete_gamma2(a,x,d=d+1)
        return result
    else:
        m = (d*2)-1
        return (m-a)+x+ ((d*(a-d))/(upper_incomplete_gamma2(a,x,d=d+1)))

def lower_incomplete_gamma(a,x,d=0,iterations=100):
    if d == iterations:
        if ((d % 2) == 1):
            return 1.0 # end iterations
        else:
            m = d/2
            return x + (m-a)
    if d == 0:
        result = ((x**a) * (e**(-x)))/lower_incomplete_gamma(a,x,d=d+1)
        return result
    elif ((d % 2) == 1):
        m = d - 1
        n = (d-1.0)/2.0
        return a + m - (((a+n)*x)/lower_incomplete_gamma(a,x,d=d+1))
    else:
        m = d-1
        n = d/2.0
        return a+m+((n*x)/(lower_incomplete_gamma(a,x,d=d+1)))

def lower_incomplete_gamma2(a,x):
    return gamma(a)-upper_incomplete_gamma2(a,x)

def complimentary_incomplete_gamma(a,x):
    return 1.0-upper_incomplete_gamma(a,x)

# Scipy name mappings
def gammainc(a,x):
    return lower_incomplete_gamma(a,x)/gamma(a)

def gammaincc(a,x):
    return upper_incomplete_gamma(a,x)/gamma(a)


# # 2_ GF2MATRIX

# In[ ]:


from __future__ import print_function

import copy

MATRIX_FORWARD_ELIMINATION = 0
MATRIX_BACKWARD_ELIMINATION = 1

def print_matrix(matrix):
    #print "PRINT MATRIX"
    #print "len matrix = ",str(len(matrix))
    #for line in matrix:
    #    print line
    for i in range(len(matrix)):
        #print "Line %d" % i
        line = matrix[i]
        #print "Line %d = %s" % (i,str(line))
        if i==0:
            astr = "["+str(line)+" : "
        else:
            astr += " "+str(line)+" : "
        for ch in line:
            astr = astr + str(ch)
        if i == (len(matrix)-1):
            astr += "]"
        else:
            astr = astr + "\n"
    print(astr)
    #print "END PRINT MATRIX"

    
def row_echelon(M,Q,matrix,blknum):
    lm = copy.deepcopy(matrix)
    
    pivotstartrow = 0
    pivotstartcol = 0
    for i in range(Q):
        # find pivotrow
        found = False
        for k in range(pivotstartrow,Q):
            if lm[k][pivotstartcol] == 1:
                found = True
                pivotrow = k
                break
        
        if found:        
            # Swap with pivot
            if pivotrow != pivotstartrow:
                lm[pivotrow],lm[pivotstartrow] = lm[pivotstartrow],lm[pivotrow]
                    
            # eliminate lower triangle column
            for j in range(pivotstartrow+1,Q):
                if lm[j][pivotstartcol]==1:
                    lm[j] = [x ^ y for x,y in zip(lm[pivotstartrow],lm[j])]  
                
            pivotstartcol += 1
            pivotstartrow += 1
        else:
            pivotstartcol += 1
        
    return lm

def rank(M,Q,matrix,blknum):
    lm = row_echelon(M,Q,matrix,blknum)
    rank = 0
    for i in range(Q):
        nonzero = False
        for bit in lm[i]:
            if bit == 1:
                nonzero = True
        if nonzero:
            rank += 1
    return rank
    
def computeRank(M, Q, matrix):
    m = min(M,Q)
    
    localmatrix = copy.deepcopy(matrix)
    # FORWARD APPLICATION OF ELEMENTARY ROW OPERATIONS  
    for i in range(m-1):
        if ( localmatrix[i][i] == 1 ): 
            localmatrix = perform_elementary_row_operations(MATRIX_FORWARD_ELIMINATION, i, M, Q, localmatrix)
        else: # localmatrix[i][i] = 0 
            row_op,localmatrix = find_unit_element_and_swap(MATRIX_FORWARD_ELIMINATION, i, M, Q, localmatrix)
            if row_op == 1: 
                localmatrix = perform_elementary_row_operations(MATRIX_FORWARD_ELIMINATION, i, M, Q, localmatrix)
        


    # BACKWARD APPLICATION OF ELEMENTARY ROW OPERATIONS  
    for i in range(m-1,0,-1):
    #for ( i=m-1; i>0; i-- ) {
        if ( localmatrix[i][i] == 1 ):
            localmatrix = perform_elementary_row_operations(MATRIX_BACKWARD_ELIMINATION, i, M, Q, localmatrix)
        else: #  matrix[i][i] = 0 
            row_op,localmatrix = find_unit_element_and_swap(MATRIX_BACKWARD_ELIMINATION, i, M, Q, localmatrix) 
            if row_op == 1:
                localmatrix = perform_elementary_row_operations(MATRIX_BACKWARD_ELIMINATION, i, M, Q, localmatrix)

    #for aline in localmatrix:
    #    print " UUU : ",aline
    #print
    
    rank = determine_rank(m, M, Q, localmatrix)

    return rank

def perform_elementary_row_operations(flag, i, M, Q, A):
    j = 0
    k = 0
    
    if ( flag == MATRIX_FORWARD_ELIMINATION ):
        for j in range(i+1,M):
        #for ( j=i+1; j<M;  j++ )
            if ( A[j][i] == 1 ):
                for k in range(i,Q):
                #for ( k=i; k<Q; k++ ) 
                    A[j][k] = (A[j][k] + A[i][k]) % 2
    else: 
        #for ( j=i-1; j>=0;  j-- )
        for j in range(i-1,-1,-1):
            if ( A[j][i] == 1 ):
                for k in range(Q):
                #for ( k=0; k<Q; k++ )
                    A[j][k] = (A[j][k] + A[i][k]) % 2

    return A

def find_unit_element_and_swap(flag, i, M, Q, A):
    index  = 0
    row_op = 0

    if ( flag == MATRIX_FORWARD_ELIMINATION ):
        index = i+1
        while ( (index < M) and (A[index][i] == 0) ):
            index += 1
            if ( index < M ):
                row_op = 1
                A = swap_rows(i, index, Q, A)
    else:
        index = i-1
        while ( (index >= 0) and (A[index][i] == 0) ): 
            index = index -1
            if ( index >= 0 ):
                row_op = 1
                A = swap_rows(i, index, Q, A)
    return row_op,A

def swap_rows(i, index, Q, A):
    A[i],A[index] = A[index],A[i]
    #for p in xrange(Q): 
    #    temp = A[i][p]
    #    A[i][p] = A[index][p]
    #    A[index][p] = temp
    return A

def determine_rank(m, M, Q, A):
    i = 0
    j = 0
    rank = 0
    allZeroes = 0
   
    # DETERMINE RANK, THAT IS, COUNT THE NUMBER OF NONZERO ROWS
    
    rank = m
    for i in range(M):
    #for ( i=0; i<M; i++ ) {
        allZeroes = 1 
        for j in range(Q):
        #for ( j=0; j<Q; j++)  {
            if ( A[i][j] == 1 ):
                allZeroes = 0
                #break
        if ( allZeroes == 1 ):
            rank -= 1
    return rank

def create_matrix(M, Q):
    matrix = list()
    for rownum in range(Q):
        row = [0 for x in range(M)]
        matrix.append(row)
        
    return matrix

def matrix_from_bits(M,Q,bits,blknum):
    m = list()
    for rownum in range(Q):
        row = bits[rownum*M:(rownum+1)*M]
        m.append(row)
    return m[:]


# In[ ]:




