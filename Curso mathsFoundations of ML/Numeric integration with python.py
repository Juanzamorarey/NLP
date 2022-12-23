from scipy.integrate import quad

def g(x):
    return x/2

# Remember the function follow this format
# 
#  

print(quad(g,1,2))
# (0.75, 8.326672684688674e-15)

# The first part is the result, the second is the margin of error.

# repeat the exercise for the function y = 2x in the range of 3-4 in the integral

def exercise_function(x):
    return 2*x

print(quad(exercise_function,3,4))
# (7.0, 7.771561172376096e-14)