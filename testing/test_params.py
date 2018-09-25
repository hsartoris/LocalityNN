from abc import ABC, abstractmethod

class test(ABC):
    @classmethod
    def _thing(cls):
        return 1

    @classmethod
    @abstractmethod
    def thing1(cls):
        """ok"""

    @classmethod
    def thing(cls):
        print("got here")
        out = cls.thing1()
        print(out)
        out.append(cls._thing())
        print(out)
        return out

class t2(test):
    @classmethod
    def thing1(cls):
        return [2]


