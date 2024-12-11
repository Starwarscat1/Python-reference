""" Triple quotes can be used for
multi-line comments """

from scipy.stats import stats
# Common packages

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.linalg as linalg
import math
import scipy

# Statistics packages

import statistics
import scipy.stats
#from scipy.stats import chi2
#from scipy.stats import f
#from scipy.stats import t
#from scipy.stats import norm
#from scipy.stats import binom
#from scipy.stats import poisson

# Operators

num1 = 6
num2 = 9
print("Sum:", num1 + num2) # default separation character is space
print("Difference:", num2 - num1)
print("Product:", num1 * num2)
print("Division:", num2 / num1)
print("Integer division:", num2 // num1)
print("Modulo:", num2 % num1)
print("Power:", num1 ** num2)
print("Cats are " + "awesome " * 2 + "!")

# Math functions

print("sin(pi / 2) = ", math.sin(math.pi / 2))
print("atan2(1, 1) = ", math.atan2(1, 1) * 180 / math.pi, "degrees")
print("atan2(1, 1) = ", math.atan2(-1, -1) * 180 / math.pi + 360, "degrees")
print("sqrt(4) = ", math.sqrt(4))

# Define function

def mysum(a, b = 1, c = 2):
  return a + b + c

def greeting(name):
  print("Hello", name, "!!!")

# base conversions

def ConvTo2sComplement(num, bits):
  return bin(num & (2 ** bits - 1))

print("convert to binary:", bin(41)) # convert base 10 to binary
print("convert to base 10:", int(0b101001)) # convert binary to base 10
print("convert to hex: ", hex(41)) # convert base 10 to hex
print("Shift right 3 bits:", format(0x29 >> 3, "#b"))
print("Shift left 3 bits:", format(0x29 << 3, "#b")) # new digits filled with 0
print("5 in binary", bin(5))
print("2's complement representation of -5:", ConvTo2sComplement(-5, 4))
print("python represents neg nums as: ", bin(-5))
print("Notice binary shown is of pos version:", bin(5))

# String operations

myStr = """The line breaks in this triple quoted
string are preserved.
This illustates the use of triple quotes."""
print(myStr)
myStr = "The cat landed on the moon"
print("First and second chars:", myStr[0], myStr[1])
print(myStr[4:6]) # Notice char at last index isn't included
print(myStr[:6])
print(myStr[8:])
part = myStr[4:7]
print("Length of", part, "is", len(part))
print("Index pos of substr:", part.find("at"))
print("Uppercase version:", part.upper())
print("Lowercase version:", part.lower())
print(part.zfill(5))
#help("FORMATTING")
width = 20
print(("{0:a>" + str(width) + "b}").format(45))
print(f"The value of width is {width}")

# Lists

list1 = [1, 2, 3, 4, 5]
print("List item #1:", list1[0])
print("List item #2:", list1[1])
list2 = [6, 7, 8, 9, 10]
list3 = list1 + list2
print("Combined list:", list3)
print("First 3 numbers for combined list:", list3[0:3])
print("8th and 9th numbers for combined list:", list3[-3:-1])
print("All numbers except first two for combined list:", list3[2:])
print("Length of conbined list:", len(list3))
list3.append(11)
print("Adding 11 to list:", list3)
list3[1] = 200
print("Modified combined list:", list3)

# Tuples

mytup = 1, 2, 3
print("First element of tuple:", mytup[0])
print("Second element of tuple:", mytup[1])
tup_elem1, tup_elem2, tup_elem3 = mytup
print("Last element of tuple:", tup_elem3)
print("Length of tuple:", len(mytup))

# Dictionaries

mytelbook = {"bob" : "111-2222", "sam" : "333-4444", "tim" : "777-8888"}
print("Bob's phone #:", mytelbook["bob"])
mytelbook["oscar"] = "999-0000"
print(mytelbook)
del mytelbook["bob"]
mytelbook["sam"] = "999-9999"
print(mytelbook)
print("Sam's phone #:", mytelbook["sam"])

# Loops

print("equals:", 1 == 1)
print("not equals:", 1 != 2)
print("less than:", 1 < 2)
print("greater than:", 2 > 1)
print("less than or equal:", 1 <= 2)
print("greater than or equal:", 2 >= 1)

x = 5

if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')

words = ['cat', 'window', 'defenestrate']
for w in words:
    print(w, len(w))

for i in range(5):
    print(i)

count = 0
while count < 9:
   print('The count is:', count)
   count = count + 1
   if count == 3:
     break

# Example of using a try block

try:
    print(2/0)
except ZeroDivisionError as e:
    print(f"Exception occurred: {e}")
finally:
    print("This gets executed no matter what")

# Reading and writing to a file

text_list = [
    "The quick brown fox jumps over the lazy dog.",
    "To be, or not to be, that is the question."
]

print("enumerate output:")
for item in enumerate(text_list, start = 1):
    print(item)
file_path = "/home/runner/Python-Quiz/text.txt"

with open(file_path, 'w') as file:
    for line in text_list:
        file.write(line + '\n')

with open(file_path, 'r') as file:
    print("File contents with line numbers:")
    for line_number, line in enumerate(file, start=1):
        print(f"{line_number}. {line.strip()}")

with open(file_path, 'r') as file:
    print("File contents without line numbers:")
    for line in file:
        print(line.strip())

# statistics functions

values = np.array([1, 2, 3, 4, 5])
print("Mean is", np.mean(values))
print("Median is", np.median(values))
#print("Mode is", statistics.multimode(values))
print("Range is", np.ptp(values))
print("1st quartile is", np.quantile(values, .25))
print("25th percentile is", np.percentile(values, 25))
print("Population standard deviation", np.std(values))
print("Sample standard deviation", np.std(values, ddof=1))
print("Z value of 0 corresponds to area = ", scipy.stats.norm.cdf(0))
print("Area of .5 corresponds to z = ", scipy.stats.norm.ppf(.5))
print("t value of 2.1 with df = 29 corresponds to area = ", scipy.stats.t.cdf(2.1, 29))
print("Area of .75 with df = 29 corresponds to t value = ", scipy.stats.t.ppf(.75, 29))

# Matrix operations

randNumGen = np.random.default_rng()
a = np.array([4, 2, 3, 10])
b = np.array([[1, 2], [4, 5]])
c = np.array([[1, 5], [4, 7]])
print("second element is :", a[1])
print("element in first row, second column:", b[0, 1])
print("b multiplied by c =", b @ c)
print("b transpose =", b.transpose())
print("inverse of b =", np.linalg.inv(b))
print("random 1d floating point array =", randNumGen.random(4))
print("random 2d floating point array =", randNumGen.random((2, 3)))
print("random 2d integer array =", randNumGen.integers(10, size = (3, 4))) # doesn't include 10

# Using sympy

x = sp.Symbol('x')
display(sp.expand((x - 1)*(x + 3)))
sp.sqrt(8)
sp.sin(sp.pi / 2)
sp.cos(sp.pi / 2)
f = sp.exp(x)/(x-1)
print("f: ")
display(f)
fprime_calculated = sp.diff(f, x)
print()
print("fprime: ")
display(fprime_calculated)
print()
fprime_answer = (8*(x*sp.cos(x)-sp.sin(x))+4)/(2*x+sp.cos(x))**2
print("fprime answer:")
display(fprime_answer)
display(sp.simplify(fprime_answer))
print()
#print("Verify equals to 0:")
#display(sp.simplify(fprime_answer - fprime_calculated))
#fprime_calculated.equals(fprime_answer)

soln = sp.solveset(sp.Eq(fprime_calculated, 0), x)
#soln = soln.args[0]
#soln.evalf()
soln
#display(f.subs(x, soln.args[0]))
#display(f.subs(x, soln.args[1]))

y = -2*x**2-7*x
#F = sp.integrate(f, x)
t = sp.Symbol('t')
m = (y.subs(x, t) - y.subs(x, -4))/(t-(-4))
display(m)
display(sp.simplify(m))

print(375-(60*8/2+60*2))
x1 = 0
y1 = 10
x2 = 60
y2 = 2
m = sp.Rational(y2-y1, x2-x1)
y = m*(x-x1)+y1
#display(y.subs(x, x2))
display(375-sp.integrate(y, (x, 0, 60)))

sp.solveset(sp.Eq(x-2,x**2-5*x+3), x)

vel = 6*t-20
C = sp.Symbol('C')
pos = sp.integrate(vel, t)+C
soln = sp.solveset(sp.Eq(9,pos.subs(t, 1)), C)
soln = soln.args[0]
pos = pos.subs(C, soln)
display(pos.subs(t, 4))
#display(pos.subs(t, 1))
display(pos.subs(t, 4)-pos.subs(t, 1)+9)

y = sp.Symbol('y')
left = sp.integrate(-1/(5*x), x)
right = sp.integrate(y, y)
soln = sp.solveset(sp.Eq(left, right), y)
sp.simplify(soln.args[1]-(sp.sqrt(-2*sp.log(x)/5)))

soln = sp.solveset(sp.Eq(sp.sin(x)+sp.cos(sp.pi/4), sp.sqrt(2)), x)
soln = sp.pi/4
display(sp.cos(soln)*5/sp.sin(sp.pi/4))

sp.limit((sp.exp(x)-sp.cos(x))/(4*sp.sin(x)), x, 0)

f1 = sp.sqrt(x-2)
f2 = 14-x
soln = sp.solveset(sp.Eq(f1,f2), x)
soln = soln.args[0]
print(soln)
display(sp.integrate(f2-f1, (x, 2, soln)))
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1)
t = np.arange(2, 11.001, .1)
y1 = np.sqrt(t-2)
y2 = 14-t
axes.plot(t, y1, t, y2)

M = sp.Matrix([[1, 2, 3], [-2, 0, 4]])
display(M)
display(M.shape)

display(M.row(0))
display(M.col(-1))

M.col_del(0)
display(M)
M.row_del(1)
display(M)
M = M.row_insert(1, sp.Matrix([[0, 4]]))
display(M)
M = M.col_insert(0, sp.Matrix([1, -2]))
display(M)

M = sp.Matrix([[1, 3], [-2, 3]])
N = sp.Matrix([[0, 3], [0, 7]])

print("M + N")
display(M + N)
print("M*N")
display(M*N)
print("3*M")
display(3*M)
print("M**2")
display(M**2)
print("M**-1")
display(M**-1)

M = sp.Matrix([[1, 2, 3], [4, 5, 6]])
display(M)
print("M transpose")
display(M.T)

display(sp.eye(3))
display(sp.eye(4))
display(sp.zeros(2, 3))
display(sp.ones(3, 2))

M = sp.Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
display(M)
print("M det")
display(M.det())

M = sp.Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
display(M)
print("M rref")
display(M.rref())
