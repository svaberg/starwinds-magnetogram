import os
import tecplot as tp
import argparse
import logging
import scipy.constants
import shutil
log = logging.getLogger(__name__)

import stellarwinds.tecplot.units
import stellarwinds.tecplot.streamtrace_rakes


#
# Main method. Use -h for usage and help.
#
def main():
    parser = argparse.ArgumentParser(description='Carry out initial processing on SWMF results. '
                                                 'Should be called after Preplot.')
    parser.add_argument('plt_file', type=argparse.FileType('r'), nargs='+',
                        help='TecPlot .plt files to join, '
                             'or glob pattern expanded by the shell (e.g. 3d__*.plt).')
    # parser.add_argument('--output', type=str, help='Specify the output layout file name')
    parser.add_argument('--quiet', dest='log_level', action='store_const',
                        const=logging.WARNING, default=logging.INFO, help='only log warnings and errors')
    parser.add_argument('--debug', dest='log_level', action='store_const',
                        const=logging.DEBUG, help='generate and log detailed debug output')
    args = parser.parse_args()

    log.setLevel(args.log_level) #If the logging disappears look at the convert_magnetogram.py.
    # ch = logging.StreamHandler()
    # ch.setLevel(args.log_level)
    # log.addHandler(ch)

    # ArgumentParser files are of class _io.TextIOWrapper but we want strings.
    plt_filenames = [f.name for f in args.plt_file]

    # Put together in a big file.
    plt_concatenate(plt_filenames)

    plot_me()
    quit()
    # Create some animations by running the 2d-animations.mcr script.
    # If the program ffmpeg is not found by TecPlot the animations cannot be created.
    if not os.path.exists('ffmpeg'):
        ffmpeg_path = shutil.which('ffmpeg')
        os.symlink(ffmpeg_path, 'ffmpeg')

    tecplot_macro_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tecplot/"))
    tp.macro.execute_file(os.path.join(tecplot_macro_folder, "2d-animations.mcr"))


def wrap_brackets_in_brackets(str):
    temp = str.replace("[", "aaa").replace("]", "bbb")
    wrapped = temp.replace("aaa", "[[]").replace("bbb", "[]]")
    return wrapped


def log_vars(one_liner=True):
    if one_liner:
        list_of_vars = []
        for variable_id in range(tp.active_frame().dataset.num_variables):
            list_of_vars.append("%d: %s" % (variable_id, tp.active_frame().dataset.variable(variable_id)))
        log.debug("Variables: " + ", ".join(list_of_vars))
    else:
        for variable_id in range(tp.active_frame().dataset.num_variables):
            log.debug("%4d: %s" % (variable_id, tp.active_frame().dataset.variable(variable_id)))


def plt_concatenate(plt_files):

    #
    # First read in the dataset.
    #
    tp.data.load_tecplot(plt_files)
    log.info("Read %d variables." % tp.active_frame().dataset.num_variables)
    log_vars()


    #
    # Convert to SI units (might as well)
    #
    log.info("Converting to SI units...")
    stellarwinds.tecplot.units.convert_variables_to_SI(tp.active_frame().dataset)
    log_vars()

    #
    # Calculate some variables.
    #
    tp.data.operate.execute_equation(equation='{R [m]} = sqrt({X [m]}**2 + {Y [m]}**2 + {Z [m]}**2)')
    tp.data.operate.execute_equation(equation='{U [m/s]} = sqrt({U_x [m/s]}**2 + {U_y [m/s]}**2 + {U_z [m/s]}**2)')
    tp.data.operate.execute_equation(equation='{B [T]} = sqrt({B_x [T]}**2 + {B_y [T]}**2 + {B_z [T]}**2)')
    calculate_the_other_variables()

    log.debug("After calculations: Read %d variables:" % tp.active_frame().dataset.num_variables)
    log_vars()

    #
    # Set contour values. Most interesting to see scalar quantities. There are 8 contour value slots in TecPlot.
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
        _var_id = tp.active_frame().dataset.variable(wrap_brackets_in_brackets(_name)).index
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
    # Assign animation strands.
    #
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


def calculate_the_other_variables():

    log.info("Calculating sonic speed.")
    tp.data.operate.execute_equation(equation='{c_s [m/s]} = sqrt(AUXZONE[1]:GAMMA*{P [Pa]} / {Rho [kg/m^3]})')

    # log.info("Calculating current field strength")
    # !    EQUATION = '{J (`mA/m^2)}=sqrt({J_x (`mA/m^2)}**2+{J_y (`mA/m^2)}**2+{J_z (`mA/m^2)}**2)'

    log.info("Calculating sonic Mach number.")
    tp.data.operate.execute_equation(equation='{Ma [U/c_s]} = {U [m/s]} / {c_s [m/s]}')

    log.info("Calculating Alfvén speed.")
    tp.data.operate.execute_equation(equation='{c_A [m/s]} = {B [T]} / sqrt(%s * {Rho [kg/m^3]})' % scipy.constants.mu_0,
                                     ignore_divide_by_zero=True)

    log.info("Calculating Alfvénic Mach number.")
    tp.data.operate.execute_equation(equation='{MA [U/c_A]} = {U [m/s]} / {c_A [m/s]}', ignore_divide_by_zero=True)
    
    log.info("Calculating radial magnetic field strength.")
    tp.data.operate.execute_equation(equation='{B_r [T]}=('
                                              '  {B_x [T]}*{X [m]}'
                                              '+ {B_y [T]}*{Y [m]}'
                                              '+ {B_z [T]}*{Z [m]}'
                                              ') / {R [m]}',
                                     ignore_divide_by_zero=True)

    log.info("Calculating magnetic pressure.")
    tp.data.operate.execute_equation(equation='{P_b [Pa]} = {B [T]}**2 / (2 * %s)' % scipy.constants.mu_0)

    log.info("Calculating plasma-beta.")
    tp.data.operate.execute_equation(equation='{beta [P/P_b]} = {P [Pa]} / {P_b [Pa]}', ignore_divide_by_zero=True)

    log.info("Calculating zero values (useful for forcing streamlines into planes).")
    tp.data.operate.execute_equation(equation='{Zero} = 0 * {X [R]}')

    # log.info("Calculating")
    # tp.data.operate.execute_equation(equation='')
    # log.info("Calculating")
    # tp.data.operate.execute_equation(equation='')

#
# # Number density is density divided by mass per particle. Mass per particle is 1.673532758E-27 kg (for protons only).
# # This is equal to 1.673532758E-24 grams; Divide by two to account for electrons, 8.367663791E-25 grams.
# $!ALTERDATA
#   EQUATION = '{`r (1/cm^3)} = {`r (g/cm^3)} / 8.367663791E-25'
#
# # Temperature from ideal gas law and Boltzmann constant in cgs units.
# $!ALTERDATA
#   EQUATION = '{T (K)} = {p [Pa]} / ({`r (1/cm^3)} * 1.38065e-16)'
#
# #Energy, thermal (microscopic motion)
# $!ALTERDATA
#   EQUATION = '{ET [J/m^3]} = {p [Pa]} / (AUXZONE[1]:GAMMA - 1)'
#
# #Energy, potential (bulk fluid motion). Velocity converted to cm/s.
# $!ALTERDATA
#   EQUATION = '{EK [J/m^3]} = 0.5 * {`r (g/cm^3)} * (1e5 * {U (km/s)})**2'
#
# #Energy, magnetic. Constant is vacuum permeability in CGS.
# $!ALTERDATA
#   EQUATION = '{EM [J/m^3]} = .0397887357729 * {B [T]}**2'
#
# #Total energy (sum of the above).
# $!ALTERDATA
#   EQUATION = '{ETKM [J/m^3]} = {ET [J/m^3]} + {EK [J/m^3]} + {EM [J/m^3]}'
#

def plot_me(var_name="ti [K]"):

    var_id = tp.active_frame().dataset.variable(wrap_brackets_in_brackets(var_name)).index
    log.debug("Variable %2d: %s" % (var_id, var_name))

    tp.add_page()
    tp.active_page().name = "PlotMe"

    tp.active_frame().plot_type = tp.constant.PlotType.Cartesian3D

    var_id = tp.active_frame().dataset.variable(wrap_brackets_in_brackets(var_name)).index
    tp.active_frame().plot().contour(0).variable_index = var_id
    tp.active_frame().plot().contour(0).levels.reset_to_nice(num_levels=11)


    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).show_slices = True

    tp.active_frame().plot().view.psi = 90
    tp.active_frame().plot().view.theta = -90
    tp.active_frame().plot().view.fit(consider_blanking=True)

    tp.active_frame().plot().slice(0).contour.flood_contour_group_index = 0

    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).vector.u_variable_index = 28
    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).vector.v_variable_index = 5
    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).vector.w_variable_index = 6

    tp.active_frame().plot().show_streamtraces = True

    tp.active_frame().plot().contour(0).colormap_name = 'Hot Metal'

    tp.active_frame().plot().contour(0).colormap_filter.distribution = tp.constant.ColorMapDistribution.Banded
    tp.active_frame().plot().contour(0).levels.reset_to_nice(num_levels=16)
    tp.active_frame().plot().slice(0).contour.contour_type = tp.constant.ContourType.AverageCell

    tp.active_frame().plot().streamtraces.delete_all()

    stellarwinds.tecplot.streamtrace_rakes.add_circular_rake(64, 2, 'x', offset=0)

    tp.active_frame().plot().streamtraces.color = tp.constant.Color.Custom12

    tp.active_frame().plot().contour(0).labels.step = 2
    tp.active_frame().plot().contour(0).legend.box.box_type = tp.constant.TextBox.Filled

    try:
        num_solution_times = tp.active_frame().plot().num_solution_times
    except tp.exception.TecplotOutOfDateEngineError:
        num_solution_times = 10

    for solution_time in range(num_solution_times):
        tp.macro.execute_command('$!GlobalTime SolutionTime = %d' % solution_time)

        tp.export.save_png('plot_yz_ti_%03d.png' % solution_time,
                           width=1024,
                           region=tp.constant.ExportRegion.CurrentFrame,
                           supersample=3,
                           convert_to_256_colors=False)

        tp.macro.execute_command('$!RedrawAll')


if __name__ == "__main__":
    main()


