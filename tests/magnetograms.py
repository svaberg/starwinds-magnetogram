import numpy as np
import logging
log = logging.getLogger(__name__)

from stellarwinds.magnetogram import coefficients as shc


def data_mengel():
    """Set up some test data (from M. Mengel)"""

    data_0 = np.fromstring(r'''
    1  0 -8.001816e+01 -0.000000e+00
     1  1  6.474963e+01 -4.789556e+01
     2  0 -1.679751e+01 -0.000000e+00
     2  1  1.683802e+01 -3.321388e+01
     2  2 -1.248859e+01 -1.170199e+01
     3  0 -1.097443e+01 -0.000000e+00
     3  1 -1.048333e+00 -2.001201e+01
     3  2 -1.531429e+01 -1.574171e+01
     3  3 -2.679741e+01 -2.874833e+01
     4  0 -1.299011e+01 -0.000000e+00
     4  1  2.211721e+00 -1.048567e+01
     4  2 -1.711044e+01 -1.481635e+01
     4  3 -2.772366e+01 -1.217352e+01
     4  4 -4.268489e-01  9.130483e+00
     5  0 -9.481055e+00 -0.000000e+00
     5  1  4.767711e+00 -4.071863e+00
     5  2 -1.446052e+01 -1.087121e+01
     5  3 -2.820549e+01 -4.026882e-01
     5  4  1.659654e+00 -5.601834e+00
     5  5  1.136790e+01 -2.211857e+01
     6  0 -3.613828e+00 -0.000000e+00
     6  1  3.098916e+00 -1.686366e-01
     6  2 -8.264525e+00 -5.502018e+00
     6  3 -2.461693e+01 -4.425425e-01
     6  4  3.687756e+00 -1.674217e+01
     6  5  7.302930e+00 -1.965251e+01
     6  6  8.360615e+00  7.053699e+00
     7  0  5.359819e-01 -0.000000e+00
     7  1  9.527038e-01  1.899756e+00
     7  2 -3.330275e+00 -1.142496e+00
     7  3 -1.713662e+01 -2.356382e+00
     7  4  2.703080e+00 -1.557146e+01
     7  5  7.888743e+00 -1.608858e+01
     7  6 -3.025988e+00  2.778594e+00
     7  7  7.073648e+00 -2.043330e+01
     8  0  1.896802e+00 -0.000000e+00
     8  1 -4.820256e-01  2.523361e+00
     8  2 -7.691741e-01  6.767981e-01
     8  3 -1.102696e+01 -2.605981e+00
     8  4  2.750385e-01 -9.669539e+00
     8  5  8.958370e+00 -1.236757e+01
     8  6 -1.069470e+01 -4.156229e-01
     8  7  4.853118e+00 -1.097333e+01
     8  8 -1.464965e+01  5.753190e+00
     9  0  1.790837e+00 -0.000000e+00
     9  1 -9.089128e-01  2.025129e+00
     9  2  6.414118e-01  9.636024e-01
     9  3 -6.942465e+00 -2.138817e+00
     9  4 -9.240442e-01 -5.335339e+00
     9  5  7.660962e+00 -8.614747e+00
     9  6 -1.008022e+01 -1.210200e+00
     9  7  5.766151e+00 -4.393379e+00
     9  8 -8.493125e+00  1.803878e+00
     9  9 -2.398763e+00  3.541007e+00
    10  0  1.394236e+00 -0.000000e+00
    10  1 -7.567967e-01  1.216334e+00
    10  2  1.107759e+00  7.698001e-01
    10  3 -3.983771e+00 -1.341423e+00
    10  4 -9.608933e-01 -2.786215e+00
    10  5  5.532500e+00 -5.666252e+00
    10  6 -6.847445e+00 -1.251673e+00
    10  7  5.473721e+00 -1.812353e+00
    10  8 -5.150812e+00 -1.136755e-01
    10  9 -5.968216e-01  1.855748e+00
    10 10  1.397603e+01  4.729201e+00
    11  0  8.666710e-01 -0.000000e+00
    11  1 -5.699221e-01  6.438425e-01
    11  2  9.172719e-01  3.876202e-01
    11  3 -2.181833e+00 -6.707281e-01
    11  4 -7.634412e-01 -1.321138e+00
    11  5  3.451891e+00 -3.366024e+00
    11  6 -4.210769e+00 -9.829242e-01
    11  7  4.182901e+00 -4.651254e-01
    11  8 -3.162070e+00 -4.400133e-01
    11  9 -1.065665e-01  9.352031e-01
    11 10  7.010869e+00  2.444939e+00
    11 11 -3.924887e+00 -1.020019e+00
    12  0  4.213528e-01 -0.000000e+00
    12  1 -3.571819e-01  2.893289e-01
    12  2  6.484825e-01  1.268737e-01
    12  3 -1.152128e+00 -3.152099e-01
    12  4 -4.905268e-01 -6.555650e-01
    12  5  1.881012e+00 -1.840956e+00
    12  6 -2.351008e+00 -5.701531e-01
    12  7  2.883408e+00 -5.128706e-02
    12  8 -1.701325e+00 -3.470605e-01
    12  9 -2.110270e-02  5.013549e-01
    12 10  3.923421e+00  1.060061e+00
    12 11 -2.089278e+00 -9.201265e-01
    12 12  4.884357e+00  2.026825e-01
    13  0  1.888189e-01 -0.000000e+00
    13  1 -1.857026e-01  9.739672e-02
    13  2  3.988453e-01  1.893348e-02
    13  3 -5.455464e-01 -1.117772e-01
    13  4 -2.397591e-01 -3.299424e-01
    13  5  9.529705e-01 -9.487364e-01
    13  6 -1.218784e+00 -2.803567e-01
    13  7  1.665996e+00  2.007993e-02
    13  8 -9.381736e-01 -1.778601e-01
    13  9 -5.614203e-02  2.214267e-01
    13 10  2.114021e+00  3.905561e-01
    13 11 -1.280319e+00 -7.645325e-01
    13 12  1.997471e+00  3.808182e-01
    13 13 -7.506762e-01 -1.791107e+00
    14  0  5.053065e-02 -0.000000e+00
    14  1 -9.955092e-02  3.080776e-02
    14  2  1.933063e-01 -3.907908e-02
    14  3 -2.479703e-01 -3.559169e-02
    14  4 -1.226321e-01 -1.473734e-01
    14  5  4.398139e-01 -4.348960e-01
    14  6 -6.284517e-01 -1.190404e-01
    14  7  8.852434e-01  3.679370e-02
    14  8 -4.608723e-01 -7.250824e-02
    14  9 -4.228012e-02  1.186146e-01
    14 10  9.862841e-01  3.358325e-02
    14 11 -7.060039e-01 -4.801788e-01
    14 12  9.692614e-01  3.275638e-01
    14 13 -3.491819e-01 -7.803518e-01
    14 14  4.003257e-01 -3.051463e-01
    15  0 -1.101435e-02 -0.000000e+00
    15  1 -4.059526e-02  7.711260e-03
    15  2  9.377972e-02 -3.836531e-02
    15  3 -1.003276e-01 -2.010106e-02
    15  4 -5.712151e-02 -7.457530e-02
    15  5  1.854545e-01 -2.025866e-01
    15  6 -3.060687e-01 -2.438803e-02
    15  7  4.424188e-01  9.165107e-03
    15  8 -2.143308e-01 -5.990521e-03
    15  9 -1.787342e-02  5.316499e-02
    15 10  4.764688e-01 -3.126802e-02
    15 11 -3.111562e-01 -2.410947e-01
    15 12  4.346983e-01  1.778020e-01
    15 13 -1.710447e-01 -4.316968e-01
    15 14  1.114358e-01 -7.707553e-02
    15 15  3.190617e-01  1.603577e-01''', sep=' \n');

    data_1 = np.fromstring(r'''
     1  0  2.558237e+01 -0.000000e+00
     1  1 -2.028428e+02 -9.932192e+01
     2  0 -6.775166e+00 -0.000000e+00
     2  1 -6.802405e+01 -2.672012e+01
     2  2 -1.441837e+01  8.351885e+00
     3  0 -1.692470e+01 -0.000000e+00
     3  1 -5.364628e+00 -5.552905e+00
     3  2 -6.035021e+00  5.097532e+00
     3  3 -2.521617e+01  3.132341e+01
     4  0 -9.043257e+00 -0.000000e+00
     4  1  3.618985e+00 -2.378909e-02
     4  2 -2.190438e-01  1.049477e+00
     4  3 -1.327795e+01  2.024210e+01
     4  4  1.386642e+01 -3.647822e+01
     5  0 -5.987742e-01 -0.000000e+00
     5  1 -2.840789e-01  2.651153e+00
     5  2  5.102442e+00  4.124830e+00
     5  3 -6.320332e+00  1.274853e+00
     5  4  9.870506e+00 -2.407804e+01
     5  5 -5.931267e+00 -8.174645e+00
     6  0  2.792324e+00 -0.000000e+00
     6  1 -1.322045e+00  3.456884e+00
     6  2  4.814841e+00  6.521615e+00
     6  3  2.232634e+00 -2.435981e+00
     6  4  2.248436e+00 -6.865795e+00
     6  5  2.158311e+00  1.842917e+00
     6  6 -2.344421e+01 -3.889453e+00
     7  0  2.084787e+00 -0.000000e+00
     7  1 -1.637614e+00  2.479156e+00
     7  2  2.390682e+00  3.911699e+00
     7  3  3.491833e+00 -1.165471e+00
     7  4 -2.117107e+00  2.578878e+00
     7  5  6.262093e+00  9.305335e-01
     7  6 -1.274246e+01 -2.952419e+00
     7  7 -1.297906e-01  6.339604e+00
     8  0  1.041280e+00 -0.000000e+00
     8  1 -7.893330e-01  7.096079e-01
     8  2  1.692727e+00  1.511147e+00
     8  3  2.261122e+00 -5.530990e-01
     8  4 -1.363154e+00  3.228126e+00
     8  5  1.319040e+00  1.584560e+00
     8  6 -2.719327e+00 -9.512229e-01
     8  7  1.894526e+00  6.746816e+00
     8  8  2.439620e+00 -4.931391e+00
     9  0  6.721135e-01 -0.000000e+00
     9  1 -3.350874e-02 -3.108607e-01
     9  2  7.262463e-01  4.462779e-01
     9  3  2.213133e+00  5.728459e-01
     9  4 -4.851741e-02  2.071743e+00
     9  5 -1.109041e+00  1.976104e+00
     9  6  1.945302e+00 -3.220791e-01
     9  7  1.386395e+00  2.873600e+00
     9  8  3.196756e+00 -2.351641e+00
     9  9  2.120341e+00 -1.124437e+00
    10  0 -3.786204e-02 -0.000000e+00
    10  1 -2.546722e-02 -4.342280e-01
    10  2 -2.126351e-01 -3.925411e-01
    10  3  1.355459e+00  6.698525e-01
    10  4  7.755275e-02  1.369819e+00
    10  5 -1.655011e+00  1.755723e+00
    10  6  2.109047e+00  1.219220e-01
    10  7 -4.301076e-01  1.769342e+00
    10  8  1.662649e+00 -5.430191e-01
    10  9  5.819198e-01 -9.288630e-01
    10 10 -4.100180e+00 -1.478248e+00
    11  0 -3.213562e-01 -0.000000e+00
    11  1  1.557356e-01 -3.973874e-01
    11  2 -2.049868e-01 -3.906446e-01
    11  3  7.627640e-01  3.331718e-01
    11  4  2.809361e-01  5.868741e-01
    11  5 -1.655010e+00  1.288174e+00
    11  6  1.543641e+00  3.998994e-01
    11  7 -8.754119e-01  5.401131e-01
    11  8  1.350808e+00  1.486124e-01
    11  9 -5.819188e-03 -4.118940e-01
    11 10 -3.386501e+00 -1.557654e+00
    11 11  7.878563e-01 -6.814637e-01
    12  0 -1.969563e-01 -0.000000e+00
    12  1  1.414191e-01 -2.534878e-01
    12  2 -2.320317e-01 -2.001303e-01
    12  3  5.761002e-01  3.178256e-01
    12  4  2.857378e-01  3.603336e-01
    12  5 -9.999285e-01  8.686650e-01
    12  6  1.066575e+00  2.381506e-01
    12  7 -1.102206e+00  1.493702e-01
    12  8  7.126350e-01  2.596815e-01
    12  9 -1.579550e-01 -2.680865e-01
    12 10 -1.861871e+00 -8.672902e-01
    12 11  8.953191e-01  1.458499e-01
    12 12 -2.310927e+00  5.959519e-01
    13  0 -2.084649e-01 -0.000000e+00
    13  1  3.272887e-02 -7.282077e-02
    13  2 -2.273226e-01 -1.514508e-01
    13  3  2.751615e-01  9.824929e-02
    13  4  4.296440e-02  2.779626e-01
    13  5 -5.746056e-01  5.282492e-01
    13  6  5.501338e-01  1.339237e-01
    13  7 -7.246053e-01  6.703177e-02
    13  8  5.244970e-01  1.401277e-01
    13  9  9.963848e-02 -3.140792e-02
    13 10 -1.250313e+00 -4.402290e-01
    13 11  6.988827e-01  2.653404e-01
    13 12 -1.230659e+00 -5.957933e-02
    13 13  2.660852e-01  6.898447e-01
    14  0 -9.165212e-02 -0.000000e+00
    14  1  5.240982e-02 -2.063825e-02
    14  2 -4.745246e-02  1.683018e-02
    14  3  1.745047e-01  1.950068e-02
    14  4  4.395576e-02  9.548642e-02
    14  5 -3.213881e-01  2.201156e-01
    14  6  3.352754e-01  1.113877e-01
    14  7 -4.821484e-01 -6.198023e-02
    14  8  2.798579e-01  1.300259e-01
    14  9  6.513265e-02 -7.063535e-02
    14 10 -5.719933e-01  3.098064e-02
    14 11  5.103180e-01  2.518614e-01
    14 12 -5.834323e-01 -1.192599e-01
    14 13  2.184027e-01  4.029324e-01
    14 14 -3.064042e-01  2.569518e-01
    15  0 -1.005083e-03 -0.000000e+00
    15  1 -3.054470e-03  5.671895e-03
    15  2 -3.934917e-02  2.997935e-02
    15  3  9.576775e-02  4.941385e-02
    15  4  2.253273e-02  8.322748e-02
    15  5 -1.343550e-01  1.629459e-01
    15  6  1.805704e-01  6.002420e-03
    15  7 -3.229910e-01 -5.056832e-03
    15  8  1.686062e-01  6.504819e-03
    15  9  2.568684e-02 -1.291128e-02
    15 10 -3.126847e-01  7.697063e-03
    15 11  7.533087e-02  6.269378e-02
    15 12 -2.580560e-01 -6.345505e-02
    15 13  3.990331e-02  1.971539e-01
    15 14 -9.472395e-02  9.632652e-02
    15 15 -1.588316e-01 -1.082213e-01''', sep=' \n')

    data_2 = np.fromstring(r'''
     1  0  6.387791e+01 -0.000000e+00
     1  1  5.698422e+01 -9.444741e+01
     2  0  1.060132e+01 -0.000000e+00
     2  1 -2.012343e+01  2.584418e+01
     2  2 -1.354695e+01  8.005335e+00
     3  0 -3.237089e+01 -0.000000e+00
     3  1 -2.424551e+01  2.098538e+01
     3  2 -8.398750e+00  2.060556e+01
     3  3  2.686469e-01 -5.758366e+00
     4  0 -3.774065e+01 -0.000000e+00
     4  1 -1.645804e+01 -2.442136e+00
     4  2 -5.142185e+00  1.556344e+01
     4  3 -1.517790e+00  1.358396e+01
     4  4 -1.001827e+00  1.534020e+00
     5  0 -2.674318e+01 -0.000000e+00
     5  1 -1.006264e+01 -6.002410e+00
     5  2 -7.794885e+00  8.977026e+00
     5  3 -2.484334e+00  1.242019e+01
     5  4 -5.091824e+00 -2.396004e+00
     5  5  1.016679e+01  3.385190e-01
     6  0 -1.878617e+01 -0.000000e+00
     6  1 -1.969309e+00 -2.063241e+00
     6  2 -6.299470e+00  3.550359e+00
     6  3 -3.328601e+00  1.195289e+01
     6  4 -3.264305e+00 -2.126160e+00
     6  5 -3.351480e+00 -1.734757e+00
     6  6 -3.756314e+00  2.593405e+00
     7  0 -1.278021e+01 -0.000000e+00
     7  1  2.860883e+00  3.175597e-01
     7  2 -1.830595e+00 -6.427441e-01
     7  3 -3.020886e+00  9.840127e+00
     7  4 -4.683662e+00 -1.205360e+00
     7  5 -2.881153e+00 -7.169764e-01
     7  6 -4.357075e-01  2.155285e+00
     7  7  1.126079e+01  1.690438e+00
     8  0 -7.353333e+00 -0.000000e+00
     8  1  3.487037e+00  1.511134e+00
     8  2  2.221921e-01 -2.380838e+00
     8  3 -2.685020e+00  6.192556e+00
     8  4 -4.246605e+00 -4.780435e-01
     8  5 -4.045467e+00 -3.249158e+00
     8  6 -7.562323e-02  5.168574e-01
     8  7  1.538717e-02 -9.554084e-01
     8  8 -2.481421e+00 -8.086513e+00
     9  0 -3.949223e+00 -0.000000e+00
     9  1  2.974027e+00  1.811880e+00
     9  2  6.008902e-01 -2.680270e+00
     9  3 -2.062498e+00  4.066578e+00
     9  4 -2.420358e+00  1.341415e-01
     9  5 -3.847836e+00 -3.271987e+00
     9  6 -6.109061e-01  2.295716e+00
     9  7 -1.875997e-01  9.656956e-02
     9  8 -3.983214e-02  2.949785e-01
     9  9 -2.066213e+00 -7.264301e-01
    10  0 -1.763509e+00 -0.000000e+00
    10  1  1.978645e+00  1.432210e+00
    10  2  5.990977e-01 -2.234698e+00
    10  3 -1.040761e+00  2.329514e+00
    10  4 -1.564479e+00  3.507865e-01
    10  5 -2.515272e+00 -2.281602e+00
    10  6 -5.835435e-01  2.118450e+00
    10  7  3.389815e-02 -1.307463e+00
    10  8  6.913563e-02 -1.443734e-01
    10  9  6.380791e-02 -4.639132e-02
    10 10 -3.134353e+00  8.083991e+00
    11  0 -5.119340e-01 -0.000000e+00
    11  1  9.330824e-01  9.361736e-01
    11  2  3.058517e-01 -1.441665e+00
    11  3 -4.919742e-01  1.182883e+00
    11  4 -9.974277e-01  2.947610e-01
    11  5 -1.841402e+00 -1.718963e+00
    11  6 -4.317830e-01  1.278337e+00
    11  7 -2.780149e-02 -1.175224e+00
    11  8 -7.649715e-02  3.671114e-01
    11  9 -7.069549e-02 -3.198992e-02
    11 10  4.406556e-02 -9.697633e-02
    11 11  9.535512e-01 -2.530411e+00
    12  0 -9.558629e-02 -0.000000e+00
    12  1  4.045497e-01  5.448543e-01
    12  2  9.039834e-02 -8.540744e-01
    12  3 -2.380341e-01  6.713051e-01
    12  4 -5.976335e-01  1.640376e-01
    12  5 -1.023755e+00 -9.271310e-01
    12  6 -2.969821e-01  9.177847e-01
    12  7  8.577111e-02 -8.132634e-01
    12  8 -9.527040e-02  3.346839e-01
    12  9  3.840400e-02 -2.600763e-03
    12 10 -6.938931e-02  3.801845e-01
    12 11 -5.115906e-02  3.941720e-02
    12 12 -4.194235e-01  2.711638e+00
    13  0  9.430197e-02 -0.000000e+00
    13  1  1.345898e-01  2.753575e-01
    13  2  4.383172e-03 -4.362310e-01
    13  3 -6.165713e-02  2.604154e-01
    13  4 -3.753227e-01  9.433924e-02
    13  5 -4.950001e-01 -4.968799e-01
    13  6 -1.531320e-01  4.799921e-01
    13  7  4.233755e-02 -6.360073e-01
    13  8 -5.872720e-02  1.641928e-01
    13  9  2.032718e-02 -1.378994e-02
    13 10  1.638587e-02 -1.803930e-01
    13 11  1.254381e-01 -1.728835e-01
    13 12  9.572678e-03 -4.344134e-03
    13 13  1.075786e+00 -4.939663e-01
    14  0  1.030922e-01 -0.000000e+00
    14  1  5.062303e-03  1.178633e-01
    14  2 -3.178471e-02 -1.725417e-01
    14  3 -5.302699e-02  1.080169e-01
    14  4 -2.027504e-01  5.964145e-02
    14  5 -2.857530e-01 -2.878180e-01
    14  6 -4.882448e-02  2.494846e-01
    14  7  3.330560e-02 -3.111223e-01
    14  8 -3.334758e-02  1.425763e-01
    14  9 -3.907772e-03  1.000878e-02
    14 10 -3.810902e-03 -1.222862e-01
    14 11 -4.135283e-02  6.660011e-02
    14 12 -6.847163e-02  1.351935e-01
    14 13 -4.888549e-03 -8.221283e-03
    14 14  1.348806e-01  1.829371e-01
    15  0  4.114810e-02 -0.000000e+00
    15  1 -2.309842e-04  4.557741e-02
    15  2 -2.786215e-02 -7.830197e-02
    15  3 -3.585924e-02  4.941171e-02
    15  4 -1.204564e-01  1.536423e-02
    15  5 -1.034937e-01 -1.049657e-01
    15  6 -4.379225e-03  1.377163e-01
    15  7 -1.178939e-03 -1.681324e-01
    15  8 -8.219460e-03  4.677316e-02
    15  9  1.348706e-02 -3.465637e-03
    15 10 -2.610104e-02 -7.931983e-02
    15 11 -2.224075e-02  3.412788e-02
    15 12  1.593992e-02 -2.497673e-02
    15 13  8.775593e-02 -2.816875e-02
    15 14  2.003318e-03  6.953326e-04
    15 15 -7.955286e-02  1.882029e-01''', sep=' \n')

    magnetogram = [d.reshape((-1, 4)) for d in (data_0, data_1, data_2)]

    coefficients = [shc.Coefficients() for _in in range(len(magnetogram))]

    for _coeffs, _magn in zip(coefficients, magnetogram):
        for _id in range(_magn.shape[0]):
            _deg_l = int(_magn[_id, 0])
            _ord_m = int(_magn[_id, 1])
            _cf = _magn[_id, 2] + 1.0j * _magn[_id, 3]
            _coeffs.append(_deg_l, _ord_m, _cf)

    # import pdb; pdb.set_trace()
    return shc.hstack(coefficients)


def dipole():
    """A dipole"""
    coefficients = shc.Coefficients(np.zeros(3, dtype=complex))
    coefficients.append(1, 0, np.array([1.0+0.0j, 0.0+0.0j, 0.0+0.0j]))
    return coefficients


def quadrupole():
    """A dipole"""
    coefficients = shc.Coefficients(np.zeros(3, dtype=complex))
    coefficients.append(2, 0, np.array([1.0+0.0j, 0.0+0.0j, 0.0+0.0j]))
    coefficients.append(2, 1, np.array([0.0+1.0j, 0.0+0.0j, 0.0+0.0j]))
    return coefficients


_magnetograms = {"dipole": dipole, "quadrupole": quadrupole, "mengel": data_mengel}


def get_radial(name):
    full = _magnetograms[name]()
    return shc.hsplit(full)[0]


def get_all(name):
    return _magnetograms[name]()
