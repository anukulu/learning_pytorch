import torch as th
import numpy as np
import matplotlib.pyplot as plt

x = th.empty(2, 2)
y = th.rand(2,2, dtype=th.float16)
print(y.dtype)
z = th.tensor([2, 4, 5.7 ])
print(z)

# can use +, -, / , * for elementwise operation
a = x + y
print(a)

# can use add , sub, mul, div for elementwise operations
b= th.add(x, y)
print(b)

# functions with _ are inplace functions
# this is an inplace operation and changes the value of y
# all the add, mul, sub and div are valid here also
y.add_(x)
print(y)

h = th.rand(5,3)

# slicing can be done here also
print(h[1, :])
# item can be used to get the number instead of a tensor object, use only if there is one single item
print(h[1, 1]) 
print(h[1, 1].item())

