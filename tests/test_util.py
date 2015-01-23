from nose.tools import assert_equal
from nose.tools import raises


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



