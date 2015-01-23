class Util:

    @classmethod
    def is_iterable(self, obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False
