from nose.tools import assert_equal
from datetime import datetime, date, time


class TestDateUtil():

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

    year = 2015
    month = 1
    day = 20
    hour = 7
    minute = 28
    second = 15

    def test_datetime(self):
        dt = datetime(self.year, self.month, self.day,
                      self.hour, self.minute, self.second)
        assert_equal(dt.day, self.day)
        assert_equal(dt.minute, self.minute)
        assert_equal(dt.date(), date(self.year, self.month, self.day))
        assert_equal(dt.time(), time(self.hour, self.minute, self.second))

    def test_strftime(self):
        """Format the datetime string"""
        dt = datetime(self.year, self.month, self.day,
                      self.hour, self.minute, self.second)
        assert_equal(dt.strftime('%m/%d/%Y %H:%M'), '01/20/2015 07:28')

    def test_strptime(self):
        """Convert/parse string into datetime objects"""
        dt = datetime(self.year, self.month, self.day)
        assert_equal(datetime.strptime('20150120', '%Y%m%d'), dt)