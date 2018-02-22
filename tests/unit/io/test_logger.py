from __future__ import absolute_import
from scanomatic.io.logger import Logger, parse_log_file
import os


class TestLoggerAndParser:

    def test_has_all_records(self, tmpdir):
        log_file = os.path.join(str(tmpdir), 'test.log')
        log = Logger('hello')
        log.log_to_file(log_file)

        log.warning('world')
        log.info('good-bye')

        parsed = parse_log_file(log_file)
        assert len(parsed['records']) == 2

    def test_record_has_the_data(self, tmpdir):
        log_file = os.path.join(str(tmpdir), 'test2.log')
        log = Logger('hello')
        log.log_to_file(log_file)

        log.warning('world')

        parsed = parse_log_file(log_file)
        rec1 = parsed['records'][0]
        assert rec1['date']
        assert rec1['time']
        assert rec1['source'] == 'hello'
        assert rec1['status'] == 'WARNING'
        assert rec1['message'] == 'world'
