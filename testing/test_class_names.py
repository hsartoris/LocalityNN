class t1():
    @classmethod
    def get_parent_name(cls):
        return cls.__class__.__base__.__name__

    @classmethod
    def get_class_name(cls):
        return cls.__name__

class t2(t1):
    @classmethod
    def get_class_name_2(cls):
        return cls.__name__

print(t2.get_class_name_2())

print(t2.get_class_name())

print(t2.get_parent_name())
