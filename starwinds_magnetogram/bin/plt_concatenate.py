import os
import tecplot as tp
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


    plt_files = " ".join([s.name for s in args.file])






    import pdb; pdb.set_trace()
    plt_concatenate(plt_files)
    # tp.macro.execute_file(os.path.join(tecplot_macro_folder, "plt-concatenate.mcr"))
    tecplot_macro_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tecplot/"))
    tp.macro.execute_file(os.path.join(tecplot_macro_folder, "2d-animations.mcr"))


def plt_concatenate(plt_files):

    os.environ['files'] = str(plt_files)

    macro=r"""
#!MC 1410

$!NEWLAYOUT 
$!PAGE NAME = 'Dataset'

#!
#! Read in the frames.
#!
$!READDATASET  '%s'
  READDATAOPTION = NEW
  RESETSTYLE = YES
  ASSIGNSTRANDIDS = YES
  VARLOADMODE = BYNAME
  VARNAMELIST = '"X (r)";"X (r_0)";"X [R]" "Y (r)";"Y (r_0)";"Y [R]" "Z (r)";"Z (r_0)";"Z [R]" "B_x (G)";"B_x [Gauss]" "B_y (G)";"B_y [Gauss]" "B_z (G)";"B_z [Gauss]" "U_x (km/s)";"U_x [km/s]" "U_y (km/s)";"U_y [km/s]" "U_z (km/s)";"U_z [km/s]" "p (dyn/cm^2)";"P [dyne/cm^2]" "`r (g/cm^3)";"Rho [g/cm^3]" "ti [K]" "te [K]" "I01 [erg/cm^3]" "I02 [erg/cm^3]"'
""" % plt_files

    tp.macro.execute_command(macro)

    tp.macro.execute_command("""$!ReadDataSet  '\"/Users/u1092841/temp/files/3d__var_4_n00000500.plt\" \"/Users/u1092841/temp/files/3d__var_4_n00000300.plt\" \"/Users/u1092841/temp/files/3d__var_4_n00000400.plt\" \"/Users/u1092841/temp/files/3d__var_4_n00000200.plt\" \"/Users/u1092841/temp/files/3d__var_4_n00000100.plt\" \"/Users/u1092841/temp/files/3d__var_4_n00000000.plt\" '
      ReadDataOption = New
      ResetStyle = Yes
      VarLoadMode = ByName
      AssignStrandIDs = Yes
      VarNameList = '\"X [R]\" \"Y [R]\" \"Z [R]\" \"Rho [g/cm^3]\" \"U_x [km/s]\" \"U_y [km/s]\" \"U_z [km/s]\" \"ti [K]\" \"te [K]\" \"B_x [Gauss]\" \"B_y [Gauss]\" \"B_z [Gauss]\" \"I01 [erg/cm^3]\" \"I02 [erg/cm^3]\" \"P [dyne/cm^2]\"'""")

    tp.data.operate.execute_equation(equation='{R [r]} = sqrt({X [R]}**2 + {Y [R]}**2 + {Z [R]}**2)')
    tp.data.operate.execute_equation(equation='{U [km/s]} = sqrt({U_x [km/s]}**2 + {U_y [km/s]}**2 + {U_z [km/s]}**2)')

    tp.active_frame().plot().contour(0).variable_index = 15
    tp.active_frame().plot().contour(0).levels.reset_to_nice(num_levels=11)
    tp.active_frame().plot().contour(1).variable_index = 16
    tp.active_frame().plot().contour(1).levels.reset_to_nice(num_levels=11)

    # tp.macro.execute_command('$!GlobalRGB RedChannelVar = 4')
    # tp.macro.execute_command('$!GlobalRGB GreenChannelVar = 4')
    # tp.macro.execute_command('$!GlobalRGB BlueChannelVar = 10')

    tp.active_frame().plot().contour(2).variable_index = 4
    tp.active_frame().plot().contour(3).variable_index = 5
    tp.active_frame().plot().contour(4).variable_index = 6
    tp.active_frame().plot().contour(6).variable_index = 7
    tp.active_frame().plot().contour(7).variable_index = 8

    # Show slices
    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).show_slices = True
    tp.active_frame().plot().slice(0).show = True
    tp.active_frame().plot().slice(1).show = True
    tp.active_frame().plot().slice(2).show = True
    tp.active_frame().plot().slice(0).contour.flood_contour_group_index = 1
    tp.active_frame().plot().slice(1).contour.flood_contour_group_index = 1
    tp.active_frame().plot().slice(2).contour.flood_contour_group_index = 1
    tp.active_frame().plot().view.fit(consider_blanking=True)

    # Turn of ugly orange bounding box
    tp.macro.execute_command('$!Interface ZoneBoundingBoxMode = Off')

    # Show isosurface representing the star.
    tp.active_frame().plot(tp.constant.PlotType.Cartesian3D).show_isosurfaces = True
    tp.active_frame().plot().isosurface(0).isosurface_values[0] = 1
    tp.active_frame().plot().isosurface(0).contour.flood_contour_group_index = 1

    tp.macro.execute_command('$!Paper PaperSize = A4')
    tp.macro.execute_command('$!WorkspaceView FitPaper')

    # Assign animation strands. Is intended to replace the macro code below.
    # $!IF |NUMZONES| > 1
    #   $!EXTENDEDCOMMAND
    #     COMMANDPROCESSORID = 'Strand Editor'
    #     COMMAND = 'ZoneSet=1-|NUMZONES|;AssignStrands=TRUE;StrandValue=1;AssignSolutionTime=TRUE;TimeValue=0;DeltaValue=1;TimeOption=ConstantDelta;'
    # $!GLOBALTIME SOLUTIONTIME = |NUMZONES|
    # $!ENDIF
    tp.macro.execute_extended_command(command_processor_id='Strand Editor',
                                      command='ZoneSet=1-%s;'
                                              'AssignStrands=TRUE;'
                                              'StrandValue=1;'
                                              'AssignSolutionTime=TRUE;'
                                              'TimeValue=0;'
                                              'DeltaValue=1;'
                                              'TimeOption=ConstantDelta;' % tp.active_frame().dataset.num_zones)


    # TODO assign last strand rather than first.
    tp.macro.execute_command('$!GlobalTime SolutionTime = 0')
    tp.macro.execute_command('$!RedrawAll')

    # Save style sheet used later by 2d-animations.
    tp.active_frame().save_stylesheet('temp_page_style.sty')

    # Save layout and data as single file.
    tp.save_layout('3d__var_4.lpk',
                   include_data=True)
#
#
#     macro=r"""
# #!
# #! Add useful quantities.
# #!
#
# #! Distance from domain centre (should be the star centre as well)
# $!ALTERDATA
#   EQUATION = '{R (r)}=sqrt({X (r)}**2+{Y (r)}**2+{Z (r)}**2)'
#
# #! Magnetic field strength
# $!ALTERDATA
#   EQUATION = '{B (G)}=sqrt({B_x (G)}**2+{B_y (G)}**2+{B_z (G)}**2)'
#
# #! Flow speed
# $!ALTERDATA
#   EQUATION = '{U (km/s)}=sqrt({U_x (km/s)}**2+{U_y (km/s)}**2+{U_z (km/s)}**2)'
#
# !#! Current field strength
# !$!GETVARNUMBYNAME |j_var|
# !  NAME = "J_x `mA/m^2"
# !$! IF |j_var| != 0
# !  $!ALTERDATA
# !    EQUATION = '{J (`mA/m^2)}=sqrt({J_x (`mA/m^2)}**2+{J_y (`mA/m^2)}**2+{J_z (`mA/m^2)}**2)'
# !$!ENDIF
# !$!REMOVEVAR |j_var|
#
# #! Sonic speed
# $!ALTERDATA
#   EQUATION = '{c_s (km/s)}=1e-3 * 1e-2 * sqrt(AUXZONE[1]:GAMMA*{p (dyn/cm^2)} / {`r (g/cm^3)})'
#
# #! Sonic Mach number
# $!ALTERDATA
#   EQUATION = '{Ma (U/c_s)} = {U (km/s)} / {c_s (km/s)}'
#   IGNOREDIVIDEBYZERO = YES
#
# #! Alfven speed
# $!ALTERDATA
#   EQUATION = '{c_A (km/s)}=2.821e-6 * {B (G)} / sqrt({`r (g/cm^3)})'
#
# #! Alfvenic Mach number
# $!ALTERDATA
#   EQUATION = '{MA (U/c_A)} = {U (km/s)} / MAX({c_A (km/s)}, 1e-9)'
#
# #! Radial magnetic field strength
# $!ALTERDATA
#   EQUATION = '{B_r (G)}=({B_x (G)}*{X (r)}+{B_y (G)}*{Y (r)}+{B_z (G)}*{Z (r)}) / MAX({R (r)}, 1e-9)'
#
# #! Magnetic pressure in dyn/cm2
# #! Wolfram alpha query: (n gauss)^2 / (2*mu_0)) in dyn/cm2
# $!ALTERDATA
#   EQUATION = '{p_b (dyn/cm^2)} = {B (G)}**2 / (8 * 3.14159265358979)'
#
# #! Plasma-beta (dimensionless)
# $!ALTERDATA
#   EQUATION = '{beta (p/p_b)} = {p (dyn/cm^2)} / MAX({p_b (dyn/cm^2)}, 1e-9)'
#
# #! Zero values (useful for forcing streamlines into planes)
# $!ALTERDATA
#   EQUATION = '{Zero} = 0 * {X (r)}'
#
# # Number density is density divided by mass per particle. Mass per particle is 1.673532758E-27 kg (for protons only).
# # This is equal to 1.673532758E-24 grams; Divide by two to account for electrons, 8.367663791E-25 grams.
# $!ALTERDATA
#   EQUATION = '{`r (1/cm^3)} = {`r (g/cm^3)} / 8.367663791E-25'
#
# # Temperature from ideal gas law and Boltzmann constant in cgs units.
# $!ALTERDATA
#   EQUATION = '{T (K)} = {p (dyn/cm^2)} / ({`r (1/cm^3)} * 1.38065e-16)'
#
# #Energy, thermal (microscopic motion)
# $!ALTERDATA
#   EQUATION = '{ET (erg/cm^3)} = {p (dyn/cm^2)} / (AUXZONE[1]:GAMMA - 1)'
#
# #Energy, potential (bulk fluid motion). Velocity converted to cm/s.
# $!ALTERDATA
#   EQUATION = '{EK (erg/cm^3)} = 0.5 * {`r (g/cm^3)} * (1e5 * {U (km/s)})**2'
#
# #Energy, magnetic. Constant is vacuum permeability in CGS.
# $!ALTERDATA
#   EQUATION = '{EM (erg/cm^3)} = .0397887357729 * {B (G)}**2'
#
# #Total energy (sum of the above).
# $!ALTERDATA
#   EQUATION = '{ETKM (erg/cm^3)} = {ET (erg/cm^3)} + {EK (erg/cm^3)} + {EM (erg/cm^3)}'
#
# #!
# #! If more than one file was loaded, assign the 'strands' into an animation.
# #!
# $!IF |NUMZONES| > 1
#   $!EXTENDEDCOMMAND
#     COMMANDPROCESSORID = 'Strand Editor'
#     COMMAND = 'ZoneSet=1-|NUMZONES|;AssignStrands=TRUE;StrandValue=1;AssignSolutionTime=TRUE;TimeValue=0;DeltaValue=1;TimeOption=ConstantDelta;'
# $!GLOBALTIME SOLUTIONTIME = |NUMZONES|
# $!ENDIF
# $!REDRAWALL
#
# #!
# #! Show a slice
# #! Show the star as a radius 1 isosurface (for now).
# #!
# $! GETVARNUMBYNAME |my_var|
#   NAME="R (r)"
# $!SETCONTOURVAR
#   VAR = |my_var|
#   CONTOURGROUP = 1
#   LEVELINITMODE = RESETTONICE
# $! GETVARNUMBYNAME |my_var|
#   NAME="U (km/s)"
# $!SETCONTOURVAR
#   VAR = |my_var|
#   CONTOURGROUP = 2
#
# $!GLOBALCONTOUR 2  COLORMAPFILTER{COLORMAPDISTRIBUTION = BANDED}
#
# $!CONTOURLEVELS RESETTONICE
#   CONTOURGROUP = 2
#   APPROXNUMVALUES = 11
#
#
#
# $!SLICELAYERS SHOW = YES
# $!SLICEATTRIBUTES 1  CONTOUR{FLOODCOLORING = GROUP2}
# $!SLICEATTRIBUTES 1  MESH{SHOW = YES}
# $!SLICEATTRIBUTES 1  MESH{LINETHICKNESS = 0.02}
#
# $!ISOSURFACELAYERS SHOW = YES
# $!ISOSURFACEATTRIBUTES 1  ISOVALUE1 = 1
# $!ISOSURFACEATTRIBUTES 1  CONTOUR{FLOODCOLORING = GROUP2}
#
#
# #!
# #! Some generic setup
# #!
# $!INTERFACE ZONEBOUNDINGBOXMODE = OFF
#
# $!THREEDVIEW VIEWERPOSITION{X = 300}
# $!THREEDVIEW VIEWERPOSITION{Y = 0}
# $!THREEDVIEW VIEWERPOSITION{Z = 0}
# $!THREEDVIEW PSIANGLE = 90
# $!THREEDVIEW THETAANGLE = -90
#
# $!VIEW FIT
#   CONSIDERBLANKING = YES
#
# $!ISOSURFACEATTRIBUTES 1  ISOVALUE1 = 1
# $!ISOSURFACELAYERS SHOW = YES
#
# $!PRINTSETUP PALETTE = COLOR
# $!EXPORTSETUP EXPORTFORMAT = AVI
# $!EXPORTSETUP IMAGEWIDTH = 1024
# $!EXPORTSETUP USESUPERSAMPLEANTIALIASING = YES
#
# #! $!SLICEATTRIBUTES 1  MESH{SHOW = YES}
# $!SLICEATTRIBUTES 1  MESH{SHOW = NO}
# $!SLICEATTRIBUTES 1  CONTOUR{CONTOURTYPE = AVERAGECELL}
#
#
# $!GLOBALTIME SOLUTIONTIME = |NUMZONES|
# $!REDRAWALL
#
#
# $!WRITESTYLESHEET  "temp_page_style.sty"
#   USERELATIVEPATHS = YES
#
#
# #!
# #! Save resulting layout
# #!
# $!SAVELAYOUT  "3d__mhd_2.plt"
#   USERELATIVEPATHS = YES
#
# #!
# #! Save resulting layout and data as lpk file
# #!
# $!SAVELAYOUT  "3d__mhd_2.lpk"
#   INCLUDEDATA = YES
#   INCLUDEPREVIEW = YES
#
# $!RemoveVar |MFBD|
# """
#     tp.macro.execute_command(macro)

if __name__ == "__main__":
    main()


