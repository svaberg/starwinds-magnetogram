import os
import tecplot as tp
import argparse
import scipy.constants
import shutil
import logging
log = logging.getLogger(__name__)

import stellarwinds.tecplot.units
import stellarwinds.tecplot.streamtrace_rakes
import stellarwinds.tecplot.derived_variables
import stellarwinds.tecplot.utils as ut


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Carry out initial processing on SWMF results. '
                                                 'Should be called after Preplot.')
    parser.add_argument('plt_file', type=argparse.FileType('r'), nargs='+',
                        help='Tecplot .plt files to join, '
                             'or glob pattern expanded by the shell (e.g. 3d__*.plt).')
    # parser.add_argument('--output', type=str, help='Specify the output layout file name')
    parser.add_argument('-q', '--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('-v', '--verbose', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    log.setLevel(args.log_level)  # If the logging disappears look at the convert_magnetogram.py.
    # ch = logging.StreamHandler()
    # ch.setLevel(args.log_level)
    # log.addHandler(ch)

    # ArgumentParser files are of type _io.TextIOWrapper but we want strings.
    plt_filenames = [f.name for f in args.plt_file]

    # Put together in a big file.
    plt_concatenate(plt_filenames)

    plot_me("ti [K]")
    plot_me("te [K]")
    quit()
    # Create some animations by running the 2d-animations.mcr script.
    # If the program ffmpeg is not found by Tecplot the animations cannot be created.
    if not os.path.exists('ffmpeg'):
        ffmpeg_path = shutil.which('ffmpeg')
        os.symlink(ffmpeg_path, 'ffmpeg')

    tecplot_macro_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tecplot/"))
    tp.macro.execute_file(os.path.join(tecplot_macro_folder, "2d-animations.mcr"))


def plt_concatenate(plt_files):

    #
    # First read in the dataset.
    #
    tp.data.load_tecplot(plt_files)
    log.info("Read %d variables." % tp.active_frame().dataset.num_variables)
    stellarwinds.tecplot.derived_variables.log_vars()


    #
    # Convert to SI units (might as well)
    #
    log.info("Converting to SI units...")
    stellarwinds.tecplot.units.convert_variables_to_SI(tp.active_frame().dataset)
    stellarwinds.tecplot.derived_variables.log_vars()

    #
    # Calculate some variables.
    #
    stellarwinds.tecplot.derived_variables.calculate_the_other_variables()

    log.debug("After calculations: Read %d variables:" % tp.active_frame().dataset.num_variables)
    stellarwinds.tecplot.derived_variables.log_vars()

    #
    # Set contour values. Most interesting to see scalar quantities. There are 8 contour value slots in Tecplot.
    #
    contour_vars = ["R [m]",
                    "U [m/s]",
                    "B [T]",
                    "Rho [kg/m^3]",
                    "ti [K]",
                    "te [K]",
                    "I01 [J/m^3]",
                    "I02 [J/m^3]",
                    "P [Pa]"]

    for _id, _name in enumerate(contour_vars):
        if _id > 7:
            log.warning("Not setting contour group %d to %3d: %s (there are only 8 slots)." % (_id, _var_id, _name))
            continue
        _var_id = tp.active_frame().dataset.variable(ut.wrap_brackets_in_brackets(_name)).index
        log.debug("Setting contour group %d to %3d: %s." % (_id, _var_id, _name))
        tp.active_frame().plot().contour(_id).variable_index = _var_id
        tp.active_frame().plot().contour(_id).levels.reset_to_nice(num_levels=11)

    #
    # Turn on a 3D plot.
    #
    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).show_slices = True

    tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')  # Turn of ugly orange bounding box

    tp.macro.execute_command('$!Paper PaperSize = A4')
    tp.macro.execute_command('$!WorkspaceView FitPaper')

    #
    # Show slices
    #
    for _id in range(3):
        tp.active_frame().plot().slice(_id).show = True
        tp.active_frame().plot().slice(_id).contour.flood_contour_group_index = 1

        # Show mesh and cell average values, rather than the default smoothed look.
        tp.active_frame().plot().slice(_id).contour.contour_type = tp.constant.ContourType.AverageCell
        tp.active_frame().plot().slice(_id).mesh.show = True

    tp.active_frame().plot().view.fit(consider_blanking=True)

    #
    # Show iso-surface representing the star.
    #
    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).show_isosurfaces = True
    tp.active_frame().plot().isosurface(0).isosurface_values[0] = 1
    tp.active_frame().plot().isosurface(0).contour.flood_contour_group_index = 1

    #
    # Assign animation strands if more than one file was read in.
    #
    if tp.active_frame().dataset.num_zones > 1:
        tp.macro.execute_extended_command(command_processor_id='Strand Editor',
                                          command='ZoneSet=1-%s;'
                                                  'AssignStrands=TRUE;'
                                                  'StrandValue=1;'
                                                  'AssignSolutionTime=TRUE;'
                                                  'TimeValue=0;'
                                                  'DeltaValue=1;'
                                                  'TimeOption=ConstantDelta;' % tp.active_frame().dataset.num_zones)

        tp.macro.execute_command('$!GlobalTime SolutionTime = %d' % tp.active_frame().dataset.num_zones)

    macro = r"""
    $!SLICELAYERS SHOW = YES
    $!SLICEATTRIBUTES 1  CONTOUR{FLOODCOLORING = GROUP2}
    $!SLICEATTRIBUTES 1  MESH{SHOW = YES}
    $!SLICEATTRIBUTES 1  MESH{LINETHICKNESS = 0.02}

    $!ISOSURFACELAYERS SHOW = YES
    $!ISOSURFACEATTRIBUTES 1  ISOVALUE1 = 1
    $!ISOSURFACEATTRIBUTES 1  CONTOUR{FLOODCOLORING = GROUP2}


    #!
    #! Some generic setup
    #! 
    $!INTERFACE ZONEBOUNDINGBOXMODE = OFF

    $!THREEDVIEW VIEWERPOSITION{X = 300}
    $!THREEDVIEW VIEWERPOSITION{Y = 0}
    $!THREEDVIEW VIEWERPOSITION{Z = 0}
    $!THREEDVIEW PSIANGLE = 90
    $!THREEDVIEW THETAANGLE = -90

    $!VIEW FIT
      CONSIDERBLANKING = YES

    $!ISOSURFACEATTRIBUTES 1  ISOVALUE1 = 1
    $!ISOSURFACELAYERS SHOW = YES

    $!PRINTSETUP PALETTE = COLOR
    $!EXPORTSETUP EXPORTFORMAT = AVI
    $!EXPORTSETUP IMAGEWIDTH = 1024
    $!EXPORTSETUP USESUPERSAMPLEANTIALIASING = YES

    #! $!SLICEATTRIBUTES 1  MESH{SHOW = YES}
    $!SLICEATTRIBUTES 1  MESH{SHOW = NO}
    $!SLICEATTRIBUTES 1  CONTOUR{CONTOURTYPE = AVERAGECELL}
    """
    tp.macro.execute_command(macro)

    tp.macro.execute_command('$!RedrawAll')

    # Save style sheet used later by 2d-animations.
    tp.active_frame().save_stylesheet('temp_page_style.sty')

    # Save layout and data as single file.
    tp.save_layout('3d__var_4.lpk',
                   include_data=True)


def plot_me(var_name="ti [K]"):
    tp.add_page()
    tp.active_page().name = "PlotMe"

    tp.active_frame().plot_type = tp.constant.PlotType.Cartesian3D

    contour_group_id = 0
    var_id = tp.active_frame().dataset.variable(ut.wrap_brackets_in_brackets(var_name)).index
    log.debug("Variable %2d: %s" % (var_id, var_name))


    var_id = tp.active_frame().dataset.variable(ut.wrap_brackets_in_brackets(var_name)).index
    tp.active_frame().plot().contour(contour_group_id).variable_index = var_id
    tp.active_frame().plot().contour(contour_group_id).levels.reset_to_nice(num_levels=11)
    tp.active_frame().plot().contour(contour_group_id).colormap_name = 'Hot Metal'
    tp.active_frame().plot().contour(contour_group_id).colormap_filter.distribution = tp.constant.ColorMapDistribution.Banded
    tp.active_frame().plot().contour(contour_group_id).levels.reset_to_nice(num_levels=16)
    tp.active_frame().plot().contour(contour_group_id).labels.step = 2
    tp.active_frame().plot().contour(contour_group_id).legend.box.box_type = tp.constant.TextBox.Filled


    tp.active_frame().plot().view.psi = 90
    tp.active_frame().plot().view.theta = -90
    tp.active_frame().plot().view.fit(consider_blanking=True)


    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).show_slices = True
    for slice_id in range(3):
        tp.active_frame().plot().slice(slice_id).contour.flood_contour_group_index = contour_group_id
        tp.active_frame().plot().slice(slice_id).contour.contour_type = tp.constant.ContourType.AverageCell


    tp.active_frame().plot().vector.u_variable = tp.active_frame().dataset.variable(ut.wrap_brackets_in_brackets('Zero'))
    tp.active_frame().plot().vector.v_variable = tp.active_frame().dataset.variable(ut.wrap_brackets_in_brackets('U_y [m/s]'))
    tp.active_frame().plot().vector.w_variable = tp.active_frame().dataset.variable(ut.wrap_brackets_in_brackets('U_z [m/s]'))


    tp.active_frame().plot().streamtraces.delete_all()
    stellarwinds.tecplot.streamtrace_rakes.add_circular_rake(64, 2, 'x', offset=0)
    tp.active_frame().plot().streamtraces.color = tp.constant.Color.Custom12
    tp.active_frame().plot().show_streamtraces = True


    try:
        num_solution_times = tp.active_frame().plot().num_solution_times
    except tp.exception.TecplotOutOfDateEngineError:
        num_solution_times = 10

    for solution_time in range(num_solution_times):
        tp.macro.execute_command('$!GlobalTime SolutionTime = %d' % solution_time)

        png_file_name = 'plot_yz_%s_%03d.png' % (ut.sanitize_as_filename(var_name), solution_time)
        tp.export.save_png(png_file_name,
                           width=1024,
                           region=tp.constant.ExportRegion.CurrentFrame,
                           supersample=3,
                           convert_to_256_colors=False)

        tp.macro.execute_command('$!RedrawAll')


if __name__ == "__main__":
    main()


