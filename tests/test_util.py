from nose.tools import assert_equal


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
