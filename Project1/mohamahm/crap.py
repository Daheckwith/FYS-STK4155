class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart

    def __mul__(self, other):
        # return (str(self.getReal() + other.getReal()), str(self.getImag() + other.getImag())+"i")
        return (str(self.r + other.r), str(self.i + other.i)+"i")

    @property
    def getReal(self):
        return self.r
    
    @property
    def getImag(self):
        return self.i

#if __name__ == '__main__':
X = Complex(3.0, -4.5)
Y = Complex(1.0, -1.0)

print(X * Y)
# print(X.r, X.i)
# print(__name__)

# Y = Complex(2, 1)

# print(Complex(1,1) * Complex(2,2))

# print(complex(1,1) + complex(2,2))
print(X.getReal + Y.getReal)

# print(len(X))

def hei(func):
    def hallo(*args):
        func(*args)
        for arg in args:
            if arg is not None:
                print(arg)
        #print()
    return hallo

@hei
def fu(a,b,c,d):
    print(a)
    
fu(2,3,4,5)


def mo_s_func(f, u):
    print(f + u)
    
shit = (6, 9) 

mo_s_func(*shit)


def superAwesomeFunc(arg1: int = 5,
                     arg2: int = 5
        ) -> None:
        """
        This function is super awesome!!!!!
        """

        print(arg1 + arg2)
     
import numpy as np
help(np.ndarray)
