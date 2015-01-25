from nose.tools import assert_equal
from pydatasnippets.core.type_util import Util


class TestUtil():

    def test_is_iterable(self):
        assert_equal(Util.is_iterable('foo'), True)
        assert_equal(Util.is_iterable(7), False)

    def test_convert_to_list(self):
        assert_equal(isinstance(Util.convert_to_list('foo'), list), True)
        assert_equal(isinstance(Util.convert_to_list(7), list), False)