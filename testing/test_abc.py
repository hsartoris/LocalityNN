from abc import ABC, abstractmethod

class t(ABC):
    super_name = "test"
    
    def __init__(self, n):
        self.x = self.nx3(n)

    @classmethod
    def nx3(cls, n):
        return n * 3

    @classmethod
    def name(cls):
        return cls.__name__

    @classmethod
    def thing(cls, test = super_name):
        print(test)

class t2(t):
    pass
