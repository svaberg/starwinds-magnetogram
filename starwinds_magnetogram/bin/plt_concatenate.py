import argparse
import logging
log = logging.getLogger(__name__)

# from stellarwinds.hypercube_rejection import hypercube_rejection


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Carry out initial processing on SWMF results. Should be called '
                                                 'after Preplot.')
    parser.add_argument('file', type=argparse.FileType('r'), nargs='+')
    # parser.add_argument('--output', type=str, help='Specify the output layout file name')
    parser.add_argument('--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('--debug', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    log.setLevel(args.log_level) #If the logging disappears look at the convert_magnetogram.py.

    # log.warning("Files:")
    # log.warning(args.file)
    # log.warning("output_layout:")
    # log.warning(args.output)

    plt_files = " ".join([s.name for s in args.file])
    # print(dir(args.file[0]))
    # log.warning(plt_files)

    import os
    os.environ['files'] = str(plt_files)


    tecplot_macro_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tecplot/"))

    # print(tecplot_macro_folder)
    # parent_of_parent_dir = os.path.join(current_file, '../../')

    # print(os.path.dirname(os.path.realpath(__file__)))


    import tecplot as tp
    tp.macro.execute_file(os.path.join(tecplot_macro_folder, "plt-concatenate.mcr"))
    tp.macro.execute_file(os.path.join(tecplot_macro_folder, "2d-animations.mcr"))


if __name__ == "__main__":
    main()

