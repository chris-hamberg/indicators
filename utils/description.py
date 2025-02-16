class Descriptor:


    def __init__(self, method, representation):
        self.method         = method
        self.representation = representation


    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


    def __repr__(self):
        if self.representation.__doc__ is not None:
            return self.representation.__doc__
        return f"{self.method}"


    def __get__(self, instance, owner):
        if instance is None: return self
        bound_method = self.method.__get__(instance, owner)
        return Descriptor(bound_method, self.representation)


class Indicator:

    @staticmethod
    def description(representation):
        
        def decorator(method):
            return Descriptor(method, representation)

        return decorator
