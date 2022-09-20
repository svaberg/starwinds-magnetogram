#
# Test 'context' to separate test code from application code; this approach
# is taken from http://docs.python-guide.org/en/latest/writing/structure/
#
import logging
log = logging.getLogger(__name__)
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

# Append sys.path to import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import conftest  # Test context

# Added this because of problems with pdb;
#
# > (Pdb) called Tcl_FindHashEntry on deleted table
# > Abort trap: 6
#
# It seems is is caused by the default back-end which would default to 'Qt5Agg' but is overridden by
# ~/.matplotlib/matplotlibrc to 'TkAgg'.
# matplotlib.use('agg')
#
log.info("Using matplotlib backend \"%s\"" % matplotlib.get_backend())
gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
non_gui_backends = matplotlib.rcsetup.non_interactive_bk
log.debug("Non Gui backends are: %s" % non_gui_backends)
log.debug("    Gui backends are: %s" % gui_env)

# Silence matplotlib a bit, as it logs a lot.
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# requests_logger.setLevel(logging.DEBUG)


# Add more rcParams here?
plt.rcParams["figure.figsize"] = (14, 7)
# plt.rcParams["axes.grid"] = True  # This produces a warning with matplotlib >= 3.5


default_artifact_directory = "artifacts"


class PlotNamer:
    """
    Give unique names to test artifacts (test output files). The files are named e.g.
    test_along_orbit.py-intial_test-0.png
    """

    def __init__(self, file_name, function_name, extension="png"):
        """
        Initialize.
        :param file_name: file name
        :param function_name: function name
        :param extension: default extension
        """
        self.basename = os.path.basename(file_name)
        self.function = function_name
        self.extension = extension
        self._counters = defaultdict(int)
        self.dirname = default_artifact_directory
        os.makedirs(self.dirname, exist_ok=True)  # Ensure artifact path exists.

    def __enter__(self):
        """
        Enter context
        :return: PlotNamer, pyplot reference
        """
        # Returning plt is a bit weird, really.
        return self, plt

    def __exit__(self, *args):
        """
        Exit context.
        :param args: Ignored.
        :return:
        """
        plt.close("all")
        conftest.list_open_files()  # For debugging.

    def get(self, extension=None):
        """
        Get a unique plot name.
        :param extension: Override default extension.
        :return:
        """

        if extension is None:
            extension = self.extension

        name = '{}-{}-{}.{}'.format(self.basename, self.function, self._counters[extension], extension)

        self._counters[extension] += 1

        return os.path.join(self.dirname, name)
