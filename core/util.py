class Util:

    @classmethod
    def is_iterable(self, obj):
        """Determines if obj is iterable.

        Useful when writing functions that can accept multiple types of
        input (list, tuple, ndarray, iterator).  Pairs well with
        convert_to_list.
        """
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    @classmethod
    def convert_to_list(self, obj):
        """Converts obj to a list if it is not a list and it is iterable, else
        returns the original obj.
        """
        if not isinstance(obj, list) and self.is_iterable(obj):
            obj = list(obj)
        return obj
