
from StringIO import StringIO
import unittest2 as unittest
from mr_s3_log_parser import MrS3LogParser


class MrTestsUtil:

    def run_mr_sandbox(self, mr_job, stdin):
        # inline runs the job in the same process so small jobs tend to
        # run faster and stack traces are simpler
        # --no-conf prevents options from local mrjob.conf from polluting
        # the testing environment
        # "-" reads from standard in
        mr_job.sandbox(stdin=stdin)

        # make_runner ensures job cleanup is performed regardless of
        # success or failure
        with mr_job.make_runner() as runner:
            runner.run()
            for line in runner.stream_output():
                key, value = mr_job.parse_output_line(line)
                yield value

                
class TestMrS3LogParser(unittest.TestCase):

    mr_job = None
    mr_tests_util = None

    RAW_LOG_LINE_INVALID = \
        '00000fe9688b6e57f75bd2b7f7c1610689e8f01000000' \
        '00000388225bcc00000 ' \
        's3-storage [22/Jul/2013:21:03:27 +0000] ' \
        '00.111.222.33 ' \

    RAW_LOG_LINE_VALID = \
        '00000fe9688b6e57f75bd2b7f7c1610689e8f01000000' \
        '00000388225bcc00000 ' \
        's3-storage [22/Jul/2013:21:03:27 +0000] ' \
        '00.111.222.33 ' \
        'arn:aws:sts::000005646931:federated-user/user 00000AB825500000 ' \
        'REST.HEAD.OBJECT user/file.pdf ' \
        '"HEAD /user/file.pdf?versionId=00000XMHZJp6DjM9x500000' \
        '00000SDZk ' \
        'HTTP/1.1" 200 - - 4000272 18 - "-" ' \
        '"Boto/2.5.1 (darwin) USER-AGENT/1.0.14.0" ' \
        '00000XMHZJp6DjM9x5JVEAMo8MG00000'

    DATE_TIME_ZONE_INVALID = "AB/Jul/2013:21:04:17 +0000"
    DATE_TIME_ZONE_VALID = "22/Jul/2013:21:04:17 +0000"
    DATE_VALID = "2013-07-22"
    DATE_TIME_VALID = "2013-07-22 21:04:17"
    TIME_ZONE_VALID = "+0000"

    def __init__(self, *args, **kwargs):
        super(TestMrS3LogParser, self).__init__(*args, **kwargs)
        self.mr_job = MrS3LogParser(['-r', 'inline', '--no-conf', '-'])
        self.mr_tests_util = MrTestsUtil()

    def test_invalid_log_lines(self):
        stdin = StringIO(self.RAW_LOG_LINE_INVALID)

        for result in self.mr_tests_util.run_mr_sandbox(self.mr_job, stdin):
            self.assertEqual(result.find("Error"), 0)

    def test_valid_log_lines(self):
        stdin = StringIO(self.RAW_LOG_LINE_VALID)

        for result in self.mr_tests_util.run_mr_sandbox(self.mr_job, stdin):
            self.assertEqual(result.find("Error"), -1)

    def test_clean_date_time_zone(self):
        date, date_time, time_zone_parsed = \
            self.mr_job.clean_date_time_zone(self.DATE_TIME_ZONE_VALID)
        self.assertEqual(date, self.DATE_VALID)
        self.assertEqual(date_time, self.DATE_TIME_VALID)
        self.assertEqual(time_zone_parsed, self.TIME_ZONE_VALID)

        # Use a lambda to delay the calling of clean_date_time_zone so that
        # assertRaises has enough time to handle it properly
        self.assertRaises(ValueError,
            lambda: self.mr_job.clean_date_time_zone(
                self.DATE_TIME_ZONE_INVALID))

if __name__ == '__main__':
    unittest.main()
