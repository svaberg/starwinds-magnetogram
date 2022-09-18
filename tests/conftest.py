import logging
log = logging.getLogger(__name__)
import psutil
import pytest

from tests import context  # Test context


@pytest.fixture(autouse=True)
def track_open_file_descriptors(request):
    """
    Log open file descriptors as this was a problem when matplotlib left plots open.
    Give a warning if it appears that file descriptors have been left open.
    psutil.Process().num_fds is not available on MS Windows, so check whether it is
    available before using it.

    As matplotlib opens a certain number of file descriptors (this will happen in the first tests
    that uses matplotlib), only give a warning if the number of file descriptors grows unmanageable
    """
    max_fd = 25
    num_fd_before = -1

    if hasattr(psutil.Process(), 'num_fds'):
        # This is not available on MS Windows.
        num_fd_before = psutil.Process().num_fds()
        log.debug('Before, there are %d open file descriptors.' % num_fd_before)
    else:
        log.debug('Number of file descriptors unavailable (Expected on MS Windows).')

    # The test function will be run at the yield statement
    yield

    if hasattr(psutil.Process(), 'num_fds'):
        num_fd_after = psutil.Process().num_fds()
        if num_fd_after > num_fd_before:
            _log_level = logging.INFO
            if num_fd_after > max_fd:
                _log_level = logging.WARNING

            log.log(_log_level, "Test \"%s\" left %d dangling file descriptors; total %d." %
                    (request.node.name,
                     num_fd_after - num_fd_before,
                     num_fd_after))
        else:
            log.debug('After,  there are %d open file descriptors.' % num_fd_after)

    else:
        log.debug('Number of file descriptors unavailable (Expected on MS Windows).')


@pytest.fixture(autouse=True)
def log_test_name(request):
    """
    Log the name of the test in a pretty box.
    """
    _str = '# Starting test \"%s\". #' % request.node.name
    log.info('#' * len(_str))
    log.info(_str)
    log.info('#' * len(_str))

    # The test function will be run at the yield statement
    yield

    log.debug('Finished test \"%s\".' % request.node.name)


@pytest.fixture
def zdi_file(request):
    content = r"""General poloidal plus toroidal field
    4 3 -3
     1  0 1. 1.
     1  1 1. 1.
     2  0 1. 1.
     2  1 1. 1.
     2  2 1. 1.

     1  0 100. 101.
     1  1 110. 111.
     2  0 200. 201.
     2  1 210. 211.
     2  2 220. 221.

     1  0 1000. 1010.
     1  1 1100. 1110.
     2  0 2000. 2010.
     2  1 2100. 2110.
     2  2 2200. 2210.
     """

    path = context.default_artifact_directory + '/test_field_zdipy.dat'

    with open(path, 'w') as f:
        f.write(content)

    return path


@pytest.fixture
def pfss_file(request):
    content = r"""Some header lines
    blah blah.
     1  0 1. 1.
     1  1 1. 1.
     2  0 1. 1.
     2  1 1. 1.
     2  2 1. 1.
     """

    path = context.default_artifact_directory + '/test_field_pfss.dat'

    with open(path, 'w') as f:
        f.write(content)

    return path


def list_open_files(proc=None):
    """Debug method to list open files

    :param proc: System process ID (PID)
    :return:
    """
    if proc is None:
        proc = psutil.Process()

    try:
        log.info('Number of file descriptors in use: %d for process \"%s\".' % (proc.num_fds(), str(proc)))
    except AttributeError:
        log.info('Number of file descriptors unavailable for process \"%s\".' % str(proc))
        pass

    log.info('Number of open files: %d for process \"%s\".' % (len(proc.open_files()), str(proc)))
    for of in proc.open_files():
        log.debug(of)


def list_all_open_files():
    """Debug method to list all open processes on the system."""
    for proc in psutil.process_iter():
        list_open_files(proc)


def prune_log_streamhandlers(keep=1):
    """
    There is this weird double logging going on.
    Who registers the log handlers? One is surely Tecplot, the other is pytest.
    Remove all but one stream handler so that double logging disappears.
    :param keep: Stream handler to keep.
    :return: None
    """
    stream_handlers = [h for h in logging.getLogger().handlers if type(h) == logging.StreamHandler]

    if len(stream_handlers) > keep:
        stream_handlers.pop(keep)
    else:
        stream_handlers.pop()

    for sh in stream_handlers:
        log.warning("Removing log handler \"%s\"." % sh)
        logging.getLogger().removeHandler(sh)

