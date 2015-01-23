class Util:

    @classmethod
    def is_iterable(self, obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    @classmethod
    def convert_to_list(self, obj):
        """Useful when writing functions that can accept multiple types of
        input (list, tuple, ndarray, iterator).  Checks if the object is a list.
        If it is not a list, converts it to a list.
        """
        if not isinstance(obj, list) and self.is_iterable(obj):
            obj = list(obj)
        return obj
