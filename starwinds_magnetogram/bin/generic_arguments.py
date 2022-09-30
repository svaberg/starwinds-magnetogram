import logging
log = logging.getLogger(__name__)

from contextlib import contextmanager

import starwinds_magnetogram as _package


def add(parser):
    parser.add_argument('-q', '--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    parser.add_argument('--debug', action='store_const', const=True, help="Start pdb debugger")
    parser.add_argument('--version', action='version',
                    version='%(prog)s {version}'.format(version=_package.__version__))

    return parser


def handle(parser):
    args = parser.parse_args()
    logging.getLogger(_package.__package__).setLevel(args.log_level)  # Set for entire package.
    

@contextmanager
def defaults(parser):
    yield 
    add(parser)
    handle(parser)
    
