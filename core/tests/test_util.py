from nose.tools import assert_equal
from nose.tools import raises
from pydatasnippets.core.util import Util


class TestUtil():

    @classmethod
    def setup_class(cls):
        """This method is run once for each class before any tests are run"""

    @classmethod
    def teardown_class(cls):
        """This method is run once for each class _after_ all tests are run"""

    def setUp(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_isinstance(self):
        assert_equal(isinstance(7, (int)), True)
        assert_equal(isinstance(7.5, (int, float)), True)
        assert_equal(isinstance('foo', (int, float, str)), True)

    def test_attributes(self):
        """Functions getattr, hasattr, setattr can be used to write generic,
        reusable code.  Objects in python have both attributes and methods
        """
        # getattr will throw an AttributeError exception if the attribute
        # does not exist
        getattr('foo', 'split')

    @raises(Exception)
    def test_attributes_fail(self):
        return getattr('foo', 'bar')

    def test_is_iterable(self):
        assert_equal(Util.is_iterable('foo'), True)
        assert_equal(Util.is_iterable(7), False)

    def test_convert_to_list(obj):
        assert_equal(isinstance(Util.convert_to_list('foo'), list), True)
        assert_equal(isinstance(Util.convert_to_list(7), list), False)

    def test_references(self):
        a = [1, 2, 3]
        b = a
        c = list(a)  # list always creates a new list
        assert_equal(a == b, True)
        assert_equal(a is b, True)
        assert_equal(a == c, True)
        assert_equal(a is c, False)

