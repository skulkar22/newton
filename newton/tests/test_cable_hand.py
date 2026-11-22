# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests of the cable-driven hand helpers.

Covered: the spec, the cable route and the hand calculation (no simulation),
the skeleton and both cable models on one digit and on the hand, the control
blocks, the drawing helpers, grasping and the choreography (short simulations).
"""

import itertools
import math
import os
import tempfile
import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cable import cable_hand as ch
from newton.examples.cable import cable_hand_visuals as chv
from newton.tests.unittest_utils import get_selected_cuda_test_devices, get_test_devices

# Devices come from the test utilities, so the device selection of the test runner applies. The classes
# share models built once in setUpClass, so each test is tied to one device: the model-building and
# tendon tests run on the CPU, and the rod hand, which is far too slow there, on the first selected CUDA
# device (those tests are skipped without one).
CPU = next((device for device in get_test_devices() if device.is_cpu), None)
CUDA = next(iter(get_selected_cuda_test_devices()), None)


def setUpModule():
    """Skip the module when the test runner selects no CPU device, instead of running on the default device."""
    if CPU is None:
        raise unittest.SkipTest("the cable hand tests build their models on the CPU device, which is not selected")


# -----------------------------------------------------------------------------
# Reference numbers of the frozen design (SI units); only the entries of the one-cable digits (ring, little,
# thumb) of the tables below are compared
# -----------------------------------------------------------------------------

# fmt: off
REF_GEOMETRY ={"index": {"cable_length": 0.149202367832565,
           "segments": 75,
           "segment_length": 0.0019893649044342,
           "free_spans": [0.012, 0.00880284020207398, 0.0285860091844292, 0.00833006609171097,
                          0.00833006579686055, 0.00973025159900945, 0.00644127316607516],
           "wrap_extension": [0.0, 1.53588979495067, 2.07168095256061, 1.72261505161964, 2.23402147386062,
                              1.95138966952069, 1.96756040227429, 0.0],
           "wrap_full_flexion": [0.0, 1.53588979495067, 0.50088462556061, 1.72261505161964, 0.488692221860619,
                                 1.95138966952069, 0.74582992627429, 0.0],
           "total_wrap_extension": 11.4831573447865,
           "total_wrap_full_flexion": 6.94530128978652,
           "takeup_full_flexion": 0.030019663134,
           "rod_barrel_vertex": [["G0", 0.00443386576961929, 0.00503313268386296, 0.0062],
                                 ["S1", 0.00745873273455468, 0.00802064959469124, 0.0092],
                                 ["G1", 0.00443386576961929, 0.00503313268386296, 0.0062],
                                 ["S2", 0.00594918250013848, 0.00652543853705183, 0.0077],
                                 ["G2", 0.00443386576961929, 0.00503313268386296, 0.0062],
                                 ["S3", 0.00443386576961929, 0.00503313268386296, 0.0062]],
           "tendon_compliance": [0.0, 1.38069474595657e-05, 1.01283626804157e-05, 3.28904605740065e-05,
                                 9.5843987385803e-06, 9.58439839933161e-06, 1.11954227163233e-05,
                                 7.4111933480592e-06]},
 "middle": {"cable_length": 0.155680082593217,
            "segments": 78,
            "segment_length": 0.00199589849478483,
            "free_spans": [0.012, 0.00880284020207398, 0.0339754146566642, 0.00833006609171098,
                           0.00833006579686056, 0.0136723021536253, 0.00644127316607516],
            "wrap_extension": [0.0, 1.53588979495067, 1.99536208454041, 1.64629618359944, 2.23402147386062,
                               1.76523007132709, 1.7814008040807, 0.0],
            "wrap_full_flexion": [0.0, 1.53588979495067, 0.424565757540412, 1.64629618359944, 0.488692221860619,
                                  1.76523007132709, 0.559670328080697, 0.0],
            "total_wrap_extension": 10.9582004123589,
            "total_wrap_full_flexion": 6.42034435735894,
            "takeup_full_flexion": 0.030019663134,
            "rod_barrel_vertex": [["G0", 0.00443342949106358, 0.00503335169083852, 0.0062],
                                  ["S1", 0.00745846094241691, 0.00802078570168451, 0.0092],
                                  ["G1", 0.00443342949106358, 0.00503335169083852, 0.0062],
                                  ["S2", 0.00594884762936357, 0.00652560636624056, 0.0077],
                                  ["G2", 0.00443342949106358, 0.00503335169083852, 0.0062],
                                  ["S3", 0.00443342949106358, 0.00503335169083852, 0.0062]],
            "tendon_compliance": [0.0, 1.28241980734389e-05, 9.40744719668563e-06, 3.63089539320234e-05,
                                  8.90220146041155e-06, 8.90220114530987e-06, 1.4611359244833e-05,
                                  6.88368024390622e-06]},
 "ring": {"cable_length": 0.150714079818392,
          "segments": 75,
          "segment_length": 0.00200952106424523,
          "free_spans": [0.012, 0.00880284020207398, 0.0296747990236206, 0.00833006609171098,
                         0.00833006579686055, 0.0111098670640123, 0.00644127316607516],
          "wrap_extension": [0.0, 1.53588979495067, 2.05441089214539, 1.70534499120442, 2.23402147386062,
                             1.87817141622376, 1.89434214897737, 0.0],
          "wrap_full_flexion": [0.0, 1.53588979495067, 0.483614565145393, 1.70534499120442, 0.488692221860619,
                                1.87817141622376, 0.672611672977366, 0.0],
          "total_wrap_extension": 11.3021807173622,
          "total_wrap_full_flexion": 6.76432466236224,
          "takeup_full_flexion": 0.030019663134,
          "rod_barrel_vertex": [["G0", 0.00443251519792701, 0.00503381067429957, 0.0062],
                                ["S1", 0.00745789137630644, 0.00802107093124128, 0.0092],
                                ["G1", 0.00443251519792701, 0.00503381067429957, 0.0062],
                                ["S2", 0.0059481458683793, 0.00652595808037474, 0.0077],
                                ["G2", 0.00443251519792701, 0.00503381067429957, 0.0062],
                                ["S3", 0.00443251519792701, 0.00503381067429957, 0.0062]],
          "tendon_compliance": [0.0, 1.35116210699365e-05, 9.91172009580224e-06, 3.34128866444736e-05,
                                9.37939137656046e-06, 9.37939104456819e-06, 1.25093594921919e-05,
                                7.25267018566314e-06]},
 "little": {"cable_length": 0.141405142848718,
            "segments": 71,
            "segment_length": 0.00199162173026363,
            "free_spans": [0.012, 0.00880284020207398, 0.0183372827074317, 0.00833006609171097,
                           0.00833006579686055, 0.0082417501891301, 0.00644127316607518],
            "wrap_extension": [0.0, 1.53588979495067, 2.30513073481775, 1.95606483387678, 2.23402147386062,
                               2.04190524288943, 2.05807597564303, 0.0],
            "wrap_full_flexion": [0.0, 1.53588979495067, 0.734334407817754, 1.95606483387678, 0.488692221860619,
                                  2.04190524288943, 0.836345499643029, 0.0],
            "total_wrap_extension": 12.1310880560383,
            "total_wrap_full_flexion": 7.59323200103828,
            "takeup_full_flexion": 0.030019663134,
            "rod_barrel_vertex": [["G0", 0.00443371523418821, 0.00503320825032855, 0.0062],
                                  ["S1", 0.00745863895357016, 0.00802069655780587, 0.0092],
                                  ["G1", 0.00443371523418821, 0.00503320825032855, 0.0062],
                                  ["S2", 0.00594906695434312, 0.00652549644553692, 0.0077],
                                  ["G2", 0.00443371523418821, 0.00503320825032855, 0.0062],
                                  ["S3", 0.00443371523418821, 0.00503320825032855, 0.0062]],
            "tendon_compliance": [0.0, 1.54423983334779e-05, 1.13280804055316e-05, 2.3597635326813e-05,
                                  1.07196832276999e-05, 1.07196828482667e-05, 1.0606032448797e-05,
                                  8.28905883377294e-06]},
 "thumb": {"cable_length": 0.136034389885546,
           "segments": 68,
           "segment_length": 0.00200050573361097,
           "free_spans": [0.0149999995387572, 0.0252756521793657, 0.00833006609171098, 0.00833006579686055,
                          0.0148890293901257, 0.00644127316607518],
           "wrap_extension": [0.0, 2.13139414981942, 1.78232829863513, 2.23402147386062, 1.72035157870576,
                              1.73652231145936, 0.0],
           "wrap_full_flexion": [0.0, 0.56059782281942, 1.78232829863513, 0.488692221860619, 1.72035157870575,
                                 0.514791835459357, 0.0],
           "total_wrap_extension": 9.60461781248028,
           "total_wrap_full_flexion": 5.06676175748028,
           "takeup_full_flexion": 0.030019663134,
           "rod_barrel_vertex": [["S1", 0.00745826874695359, 0.00802088194940499, 0.0092],
                                 ["G1", 0.00443312097483176, 0.00503350656599131, 0.0062],
                                 ["S2", 0.00594861082696711, 0.00652572504747404, 0.0077],
                                 ["G2", 0.00443312097483176, 0.00503350656599131, 0.0062],
                                 ["S3", 0.00443312097483176, 0.00503350656599131, 0.0062]],
           "tendon_compliance": [0.0, 1.68688374271897e-05, 2.84247253727074e-05, 9.36790233197967e-06,
                                 9.36790200039405e-06, 1.67440415969165e-05, 7.2437862135857e-06]}}
REF_STAIRCASE = [2.0, 8.0, 12.0, 16.0, 20.0, 26.0, 32.0, 60.0]
REF_FRICTIONLESS_Q = [[0.0, 0.0, 0.0], [0.392699081699, 0.0, 0.0], [1.1780972451, 0.290888208666, 0.0],
 [1.570796327, 0.872664625997, 0.0], [1.570796327, 1.45444104333, 0.349065850399],
 [1.570796327, 1.745329252, 0.872664625997], [1.570796327, 1.745329252, 1.221730476],
 [1.570796327, 1.745329252, 1.221730476]]
REF_FRICTIONLESS_TAKEUP = [0.0, 0.00314159265359, 0.0113155513171, 0.018238690685, 0.0237655666496, 0.028274333884,
 0.030019663134, 0.030019663134]
REF_FRICTIONLESS_TRAVEL = {"index": [0.0, 0.0038891997571, 0.0125615631563, 0.0199831072598, 0.0260083879601,
           0.031264762298, 0.0337576986515, 0.0372465318011],
 "middle": [0.0, 0.0039086329014, 0.0125939517301, 0.0200284512631, 0.026066687393, 0.0313424948751,
            0.0338548643729, 0.0374343855292],
 "ring": [0.0, 0.003893734893, 0.0125691217162, 0.0199936892437, 0.026021993368, 0.0312829028418,
          0.0337803743313, 0.0372903714487],
 "little": [0.0, 0.0038658080821, 0.0125225770313, 0.0199285266849, 0.0259382129353, 0.0311711955982,
            0.0336407402767, 0.0370204122766],
 "thumb": [0.0, 0.0038496958232, 0.0124957232665, 0.0198909314142, 0.0258898761586, 0.0311067465626,
           0.0335601789823, 0.0368646604407]}
# fmt: on

# -----------------------------------------------------------------------------
# Reference numbers of the frozen design (two-cable index and middle finger, palm layout).
# A station is [body, x, y, radius, sense]; a table row is
# [[curl tension, knuckle tension], joint angles, [curl travel, knuckle travel]].
# -----------------------------------------------------------------------------

# fmt: off
REF2_CABLES = {
    "index": {
        "curl": {"role": "curl", "slider_body": "slider", "plane_z": 0.0015, "moment_arms": [0.005, 0.0065, 0.005],
                 "stations": {"A": ["slider", -0.0253, -0.01367056, 0.0, 0],
                              "G0": ["base", -0.009279623, -0.00867056, 0.005, 1],
                              "S1": ["base", 0.0, 0.0, 0.005, -1],
                              "G1": ["link1", 0.031216888, -0.003415527, 0.005, 1],
                              "S2": ["link1", 0.045, 0.0, 0.0065, -1],
                              "G2": ["link2", 0.014124325, -0.001464049, 0.005, 1],
                              "S3": ["link2", 0.028, 0.0, 0.005, -1],
                              "T": ["link3", 0.0075, -0.0032, 0.0, 0]},
                 "rest_anchor": [-0.0253, -0.01367056], "travel_max": 0.037,
                 "cable_length": 0.147410519478, "segments": 147,
                 "free_spans": [0.016020377, 0.00782879388768, 0.0297684383382, 0.00833006609171,
                                0.00833006579686, 0.00973025159901, 0.00644127316608],
                 "wrap_extension": [1.65806274728, 2.09112441892, 1.61988556565, 2.23402147386, 1.95138966952,
                                    1.96756040227],
                 "wrap_full_flexion": [1.65806274728, 0.520328091918, 1.61988556565, 0.488692221861,
                                       1.95138966952, 0.745829926274],
                 "takeup_full_flexion": 0.025307274153,
                 "tendon_compliance": [0.0, 1.73650241286e-05, 8.48589235806e-06, 3.22670090731e-05,
                                       9.02923811816e-06, 9.02923779856e-06, 1.0546946167e-05,
                                       6.98191209533e-06],
                 "rod_barrel_radius": [0.00448322886851, 0.00448322886851, 0.00448322886851, 0.00598710266376,
                                       0.00448322886851, 0.00448322886851]},
        "knuckle": {"role": "knuckle", "slider_body": "slider_knuckle", "plane_z": -0.0015,
                    "moment_arms": [0.008, 0.0, 0.0],
                    "stations": {"Ak": ["slider_knuckle", -0.0853, -0.013343785, 0.0, 0],
                                 "G0k": ["base", -0.013299295, -0.008343785, 0.005, 1],
                                 "S1k": ["base", 0.0, 0.0, 0.008, -1],
                                 "Tk": ["link1", 0.03, -0.008, 0.0, 0]},
                    "rest_anchor": [-0.0853, -0.013343785], "travel_max": 0.022,
                    "cable_length": 0.134939750804, "segments": 135,
                    "free_spans": [0.072000705, 0.00880284020207, 0.03],
                    "wrap_extension": [1.53588979495, 2.05709457845],
                    "wrap_full_flexion": [1.53588979495, 0.486298251445],
                    "takeup_full_flexion": 0.012566370616,
                    "tendon_compliance": [0.0, 5.68383681563e-05, 6.94908573779e-06, 2.36824215081e-05],
                    "rod_barrel_radius": [0.0044833371045, 0.00748958992013]}},
    "middle": {
        "curl": {"stations": {"A": ["slider", -0.0253, -0.01367056, 0.0, 0],
                              "G0": ["base", -0.009279623, -0.00867056, 0.005, 1],
                              "S1": ["base", 0.0, 0.0, 0.005, -1],
                              "G1": ["link1", 0.036216888, -0.003415527, 0.005, 1],
                              "S2": ["link1", 0.05, 0.0, 0.0065, -1],
                              "G2": ["link2", 0.014124325, -0.001464049, 0.005, 1],
                              "S3": ["link2", 0.031, 0.0, 0.005, -1],
                              "T": ["link3", 0.0075, -0.0032, 0.0, 0]},
                 "cable_length": 0.154093131481, "segments": 154}}}
REF2_JOINTS = [
    {"name": "mcp", "sheave_radius": 0.005, "spring_preload": 0.12, "spring_stiffness": 0.08,
     "spring_damping": 0.0014},
    {"name": "pip", "sheave_radius": 0.0065, "spring_preload": 0.026, "spring_stiffness": 0.0297938053468,
     "spring_damping": 0.0004},
    {"name": "dip", "sheave_radius": 0.005, "spring_preload": 0.04, "spring_stiffness": 0.0409255567951,
     "spring_damping": 0.00015}]
REF2_LINK_MASSES = {"index": [0.0087, 0.005, 0.0033], "middle": [0.0094, 0.0054, 0.0032],
                    "ring": [0.0084, 0.0049, 0.0032], "little": [0.0061, 0.0042, 0.0027],
                    "thumb": [0.0091, 0.0062, 0.0042]}
REF2_LAYOUT = {
    "finger_origin": {"index": [0.092, 0.0, 0.034], "middle": [0.095, 0.0, 0.0105],
                      "ring": [0.09, 0.0015, -0.0105], "little": [0.0825, 0.0045, -0.0315]},
    "thumb_hinge_point": [0.022, 0.004, 0.056],
    "thumb_hinge_axis": [-0.945846529188, -0.104528463268, -0.307324166947],
    "thumb_hinge_upper": 2.268928028,
    "thumb_origin": [0.019556284, 0.003729937, 0.063612839],
    "palm_mass": 0.301,
    "thumb_column_mass": 0.02,
    "solid_boxes": [
        {"name": "index_column", "x": [-0.033, 0.092], "z": [0.0235, 0.043], "yaw": -0.087266,
         "pivot": [0.092, 0.0, 0.034], "y": [-0.017, 0.01]},
        {"name": "middle_column", "x": [-0.03, 0.095], "z": [0.0, 0.021], "yaw": 0.0,
         "pivot": [0.095, 0.0, 0.0105], "y": [-0.017, 0.01]},
        {"name": "ring_column", "x": [-0.000344, 0.09], "z": [-0.021, 0.0], "yaw": 0.087266,
         "pivot": [0.09, 0.0015, -0.0105], "y": [-0.0155, 0.0115]},
        {"name": "little_column", "x": [-0.001273, 0.0825], "z": [-0.0405, -0.021], "yaw": 0.174533,
         "pivot": [0.0825, 0.0045, -0.0315], "y": [-0.0125, 0.0145]},
        {"name": "forearm_stub", "x": [-0.09, 0.0], "z": [-0.0255, 0.0325]},
        {"name": "thenar_housing", "x": [-0.1075, -0.011], "y": [-0.0085, 0.0085], "z": [-0.0085, 0.0085],
         "pivot": [0.022, 0.004, 0.056],
         "axes": [[0.945847, 0.104528, 0.307324], [-0.099412, 0.994522, -0.032301],
                  [-0.309017, 0.0, 0.951057]]}],
    "opposition_motor": {"center": [-0.056126, -0.005146, 0.014843], "size": [0.036, 0.016, 0.016],
                         "yaw": -0.314159,
                         "axes": [[0.945847, 0.104528, 0.307324], [-0.099412, 0.994522, -0.032301],
                                  [-0.309017, -0.0, 0.951057]]}}
REF2_THRESHOLDS = {"curl": {"mcp": {"start": 20.8, "full": 45.932741232},
                            "pip": {"start": 4.0, "full": 12.0},
                            "dip": {"start": 8.0, "full": 17.9999999968}},
                   "knuckle": {"mcp": {"start": 13.75, "full": 29.45796327}}}
REF2_TWO_CABLE = {
    "index": {
        "knuckle_staircase": [[[2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0]],
                              [[2.0, 12.0], [0.0, 0.0, 0.0], [0.0001, 0.000974698754022]],
                              [[2.0, 16.0], [0.225, 0.0, 0.0], [0.001265, 0.00316457825563]],
                              [[2.0, 20.0], [0.625, 0.0, 0.0], [0.003305, 0.00675445775724]],
                              [[2.0, 24.0], [1.025, 0.0, 0.0], [0.005345, 0.0103443372588]],
                              [[2.0, 28.0], [1.425, 0.0, 0.0], [0.007385, 0.0139342167605]],
                              [[2.0, 32.0], [1.570796327, 0.0, 0.0], [0.008153981635, 0.0154904668781]],
                              [[2.0, 60.0], [1.570796327, 0.0, 0.0], [0.008433981635, 0.0182196233893]]],
        "curl_staircase": [[[2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0]],
                           [[6.0, 2.0], [0.0, 0.436332312999, 0.0], [0.00333098107345, 4e-05]],
                           [[9.0, 2.0], [0.0, 1.0908307825, 0.12217304764], [0.0085672021426, 7e-05]],
                           [[12.0, 2.0], [0.0, 1.74532925199, 0.488692190558], [0.0150251536881, 0.0001]],
                           [[15.0, 2.0], [0.0, 1.745329252, 0.855211333477], [0.017228865182, 0.00013]],
                           [[18.0, 2.0], [0.0, 1.745329252, 1.221730476], [0.0194325766738, 0.00016]],
                           [[26.0, 2.0], [0.325, 1.745329252, 1.221730476], [0.0220472187517, 0.00284]],
                           [[34.0, 2.0], [0.825, 1.745329252, 1.221730476], [0.0255368608296, 0.00692]],
                           [[60.0, 2.0], [1.570796327, 1.745329252, 1.221730476],
                            [0.0324821792179, 0.013146370616]]],
        "grid": [[[2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0]],
                 [[6.0, 2.0], [0.0, 0.436332312999, 0.0], [0.00333098107345, 4e-05]],
                 [[10.0, 2.0], [0.0, 1.308996939, 0.244346095279], [0.0107198526578, 8e-05]],
                 [[14.0, 2.0], [0.0, 1.745329252, 0.733038285838], [0.0164942946841, 0.00012]],
                 [[18.0, 2.0], [0.0, 1.745329252, 1.221730476], [0.0194325766738, 0.00016]],
                 [[30.0, 2.0], [0.575, 1.745329252, 1.221730476], [0.0237920397907, 0.00488]],
                 [[2.0, 16.0], [0.225, 0.0, 0.0], [0.001265, 0.00316457825563]],
                 [[6.0, 16.0], [0.475, 0.436332312999, 0.0], [0.00584598107345, 0.00520457825563]],
                 [[10.0, 16.0], [0.725, 1.308996939, 0.244346095279], [0.0144848526578, 0.00724457825563]],
                 [[14.0, 16.0], [0.975, 1.745329252, 0.733038285838], [0.0215092946841, 0.00928457825563]],
                 [[18.0, 16.0], [1.225, 1.745329252, 1.221730476], [0.0256975766738, 0.0113245782556]],
                 [[30.0, 16.0], [1.570796327, 1.745329252, 1.221730476], [0.0289110214257, 0.0142109488716]],
                 [[2.0, 22.0], [0.825, 0.0, 0.0], [0.004325, 0.00854939750804]],
                 [[6.0, 22.0], [1.075, 0.436332312999, 0.0], [0.00890598107345, 0.010589397508]],
                 [[10.0, 22.0], [1.325, 1.308996939, 0.244346095279], [0.0175448526578, 0.012629397508]],
                 [[14.0, 22.0], [1.570796327, 1.745329252, 0.733038285838], [0.0245482763191, 0.014635768124]],
                 [[18.0, 22.0], [1.570796327, 1.745329252, 1.221730476], [0.0274865583088, 0.014675768124]],
                 [[30.0, 22.0], [1.570796327, 1.745329252, 1.221730476], [0.0289710214257, 0.014795768124]],
                 [[2.0, 28.0], [1.425, 0.0, 0.0], [0.007385, 0.0139342167605]],
                 [[6.0, 28.0], [1.570796327, 0.436332312999, 0.0], [0.0114449627084, 0.0151405873765]],
                 [[10.0, 28.0], [1.570796327, 1.308996939, 0.244346095279], [0.0188338342928, 0.0151805873765]],
                 [[14.0, 28.0], [1.570796327, 1.745329252, 0.733038285838], [0.0246082763191, 0.0152205873765]],
                 [[18.0, 28.0], [1.570796327, 1.745329252, 1.221730476], [0.0275465583088, 0.0152605873765]],
                 [[30.0, 28.0], [1.570796327, 1.745329252, 1.221730476], [0.0290310214257, 0.0153805873765]],
                 [[2.0, 40.0], [1.570796327, 0.0, 0.0], [0.008233981635, 0.0162702258813]],
                 [[6.0, 40.0], [1.570796327, 0.436332312999, 0.0], [0.0115649627084, 0.0163102258813]],
                 [[10.0, 40.0], [1.570796327, 1.308996939, 0.244346095279], [0.0189538342928, 0.0163502258813]],
                 [[14.0, 40.0], [1.570796327, 1.745329252, 0.733038285838], [0.0247282763191, 0.0163902258813]],
                 [[18.0, 40.0], [1.570796327, 1.745329252, 1.221730476], [0.0276665583088, 0.0164302258813]],
                 [[30.0, 40.0], [1.570796327, 1.745329252, 1.221730476], [0.0291510214257, 0.0165502258813]]],
        "poses": {"straight": [[2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0]],
                  "knuckle_only": [[2.0, 24.221975512], [1.0471975512, 0.0, 0.0],
                                   [0.0054582075111, 0.0105435535939]],
                  "hook": [[17.9999999968, 2.0], [0.0, 1.745329252, 1.221730476],
                           [0.0194325766734, 0.000159999999968]],
                  "full_fist": [[40.0, 40.0], [1.570796327, 1.745329252, 1.221730476],
                                [0.0303880740231, 0.0166502258813]],
                  "pinch_ready": [[6.99999784798, 28.332964615], [1.570796327, 0.654498, 0.0],
                                  [0.0129900743136, 0.0151830413745]]}},
    "middle": {
        "poses": {"straight": [[2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0]],
                  "knuckle_only": [[2.0, 24.221975512], [1.0471975512, 0.0, 0.0],
                                   [0.0054582075111, 0.0105435535939]],
                  "hook": [[17.9999999968, 2.0], [0.0, 1.745329252, 1.221730476],
                           [0.0194860375694, 0.000159999999968]],
                  "full_fist": [[40.0, 40.0], [1.570796327, 1.745329252, 1.221730476],
                                [0.0305150436511, 0.0166502258813]],
                  "pinch_ready": [[6.99999784798, 28.332964615], [1.570796327, 0.654498, 0.0],
                                  [0.0130067808364, 0.0151830413745]]}}}
# fmt: on

DIGIT_NAMES = ("index", "middle", "ring", "little", "thumb")
ANGLE_TOL = 1.0e-6  # [rad]
LENGTH_TOL = 1.0e-8  # [m]


def _pose_grid(digit, steps):
    axes = [np.linspace(joint.lower, joint.upper, steps) for joint in digit.joints]
    return itertools.product(*axes)


def _point_segment_distance(a, b, p):
    e = b - a
    t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0.0, 1.0)
    return float(np.linalg.norm(a + t * e - p))


ONE_CABLE_DIGITS = ("ring", "little", "thumb")


class TestCableHandSpec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_shared_numbers_match_frozen_design(self):
        """Compare the cable, actuator, rod, tendon, palm, pad and object numbers with the frozen values."""
        spec = self.spec
        self.assertEqual(spec.cable.EA, 2000.0)
        self.assertEqual(spec.cable.radius, 0.0005)
        self.assertEqual(spec.cable.anchor_compliance, 2.0e-5)
        self.assertEqual(spec.cable.hinge_compliance, 1.0e-5)
        self.assertEqual((spec.cable.tension_rest, spec.cable.tension_hold, spec.cable.tension_max), (2.0, 40.0, 60.0))
        self.assertEqual(spec.station_hardware.flange_height_above_centerline, 0.0012)
        self.assertEqual(
            (spec.actuator.stroke, spec.actuator.speed_limit, spec.actuator.force_limit), (0.042, 0.03, 60.0)
        )
        self.assertEqual((spec.actuator.kp, spec.actuator.kd, spec.actuator.slider_mass), (20000.0, 20.0, 0.006))
        self.assertEqual((spec.rod.radius, spec.rod.segment_length_target, spec.rod.contact_ke), (0.0005, 0.001, 1.0e8))
        self.assertEqual((spec.tendon.orientation_guides, spec.tendon.orientation_sheaves), (1, -1))
        self.assertEqual(list(spec.staircase.tensions), REF_STAIRCASE)
        self.assertEqual((spec.palm.mass, spec.palm.thumb_column.mass), (0.301, 0.020))
        self.assertEqual((spec.actuator.stop_ke, spec.actuator.stop_kd), (20000.0, 20.0))
        self.assertEqual((spec.pads.mu, spec.pads.ke), (1.0, 1.0e5))
        self.assertEqual(spec.objects["can"]["mass"], 0.15)
        self.assertEqual(spec.objects["can"]["grasp_center_z"], 0.025)
        self.assertAlmostEqual(spec.objects["can"]["grasp_opposition"], math.radians(70.0), places=12)

    def test_design_numbers_have_one_source(self):
        """The spec is the only source of the joint stops, the structural stiffness and the can pose."""
        spec = ch.default_hand_spec()
        # structural joint stiffness: the hinge compliance of the spec and both solver settings
        self.assertEqual(ch.STRUCTURAL_JOINT_KE, 1.0e5)
        self.assertEqual(spec.cable.hinge_compliance, 1.0 / ch.STRUCTURAL_JOINT_KE)
        for settings in (ch.tendon_solver_settings(), ch.tendon_solver_settings(True), ch.rod_solver_settings(1.0e-3)):
            self.assertEqual(settings["rigid_joint_linear_ke"], ch.STRUCTURAL_JOINT_KE)
            self.assertEqual(settings["rigid_joint_angular_ke"], ch.STRUCTURAL_JOINT_KE)
        # joint stops, slider stops and pad stiffness of a changed spec reach both hand builders
        for digit in spec.digits.values():
            for joint in digit.joints:
                joint.limit_ke = 1234.0
        spec.actuator.stop_ke = 5678.0
        spec.pads.ke = 4.0e4
        plain, hand, _tendons = ch.build_tendon_hand_model(spec, device=CPU)
        grasp, grasp_hand, _tendons, scene = ch.build_tendon_hand_grasp_model(spec, "ball", device=CPU)
        for model, skeleton in ((plain, hand), (grasp, grasp_hand)):
            limit_ke = model.joint_limit_ke.numpy()
            for digit in skeleton.digits.values():
                np.testing.assert_allclose(limit_ke[list(digit.dof_flexion)], 1234.0)
                np.testing.assert_allclose(limit_ke[list(digit.dof_sliders)], 5678.0)
            self.assertEqual(float(limit_ke[skeleton.dof_opposition]), 1234.0)
        pad_shapes = sorted(scene.pads.shapes.values())
        np.testing.assert_allclose(grasp.shape_material_ke.numpy()[pad_shapes], 4.0e4)
        # the can pose of the grasp plan follows the spec
        default = ch.grasp_plan(ch.default_hand_spec(), "can")
        self.assertAlmostEqual(default.opposition, math.radians(70.0), places=12)
        spec.objects["can"]["grasp_center_z"] = 0.010
        spec.objects["can"]["grasp_opposition"] = math.radians(60.0)
        moved = ch.grasp_plan(spec, "can")
        self.assertAlmostEqual(moved.opposition, math.radians(60.0), places=12)
        # palm z is world z in the can grasp: the palm starts 15 mm higher for a can center 15 mm lower
        shift = np.array(moved.palm_grasp, dtype=float)[:3] - np.array(default.palm_grasp, dtype=float)[:3]
        np.testing.assert_allclose(shift, [0.0, 0.0, 0.015], atol=1.0e-9)

    def test_dict_round_trip(self):
        """Convert the spec to plain dicts and back and check that nothing is lost."""
        data = self.spec.to_dict()
        self.assertEqual(
            set(data),
            {
                "cable",
                "station_hardware",
                "digits",
                "palm",
                "actuator",
                "pads",
                "objects",
                "rod",
                "tendon",
                "staircase",
            },
        )
        stations = data["digits"]["ring"]["cables"][0]["stations"]
        self.assertNotIn("joint", stations[0])
        self.assertEqual(stations[2]["joint"], 0)
        again = ch.HandSpec.from_dict(data)
        self.assertEqual(again.to_dict(), data)
        self.assertIsInstance(again.digits["thumb"].mount, ch.ThumbMountSpec)
        self.assertIsInstance(again.digits["ring"].mount, ch.FingerMountSpec)
        self.assertIsInstance(again.digits["ring"].stations[1], ch.StationSpec)

    def test_default_spec_is_a_fresh_copy(self):
        """Change one default spec and check that another one is not affected."""
        other = ch.default_hand_spec()
        length = self.spec.digits["ring"].links[0].length
        other.digits["ring"].links[0].length = 1.0
        self.assertEqual(self.spec.digits["ring"].links[0].length, length)


class TestCableHandRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_geometry_matches_frozen_design(self):
        """Compare cable length, free spans, wraps and take-up of the one-cable digits with the frozen values."""
        for name in ONE_CABLE_DIGITS:
            ref = REF_GEOMETRY[name]
            geo = ch._digit_geometry(self.spec, name)
            for key in ("cable_length", "takeup_full_flexion"):
                self.assertAlmostEqual(geo[key], ref[key], delta=1.0e-12, msg=f"{name} {key}")
            for key in ("total_wrap_extension", "total_wrap_full_flexion"):
                self.assertAlmostEqual(geo[key], ref[key], delta=1.0e-10, msg=f"{name} {key}")
            for key in ("free_spans", "wrap_extension", "wrap_full_flexion"):
                np.testing.assert_allclose(geo[key], ref[key], rtol=0.0, atol=1.0e-10, err_msg=f"{name} {key}")

    def test_length_identity_spans_and_wraps_over_the_joint_range(self):
        """Check over a grid of poses that the length is the take-up law, spans are constant and wraps valid."""
        for name in DIGIT_NAMES:
            digit = self.spec.digits[name]
            for c, cable in enumerate(digit.cables):
                label = f"{name} {cable.name}"
                rest = ch.digit_route(digit, cable=c)
                contact = [k for k, station in enumerate(cable.stations) if station.radius > 0.0]
                for q in _pose_grid(digit, 5):
                    route = ch.digit_route(digit, q, cable=c)
                    takeup = ch.cable_takeup(digit, q, c)
                    self.assertAlmostEqual(route.length, rest.length - takeup, delta=1.0e-10, msg=f"{label} {q}")
                    np.testing.assert_allclose(route.spans, rest.spans, rtol=0.0, atol=1.0e-10, err_msg=f"{label} {q}")
                    self.assertTrue(np.all(route.wraps[contact] > 0.0), f"{label} {q}")
                    self.assertTrue(np.all(route.wraps[contact] < math.pi), f"{label} {q}")
                    # pulling the slider by the take-up restores the rest length
                    pulled = ch.digit_route(digit, q, travel=takeup, cable=c)
                    self.assertAlmostEqual(pulled.length, rest.length, delta=1.0e-10, msg=f"{label} {q}")
                # the wrap on a joint sheave drops by the joint angle, a guide keeps its wrap
                q_full = [joint.upper for joint in digit.joints]
                full = ch.digit_route(digit, q_full, cable=c)
                expected = rest.wraps.copy()
                for k, station in enumerate(cable.stations):
                    if station.joint is not None:
                        expected[k] -= q_full[station.joint]
                np.testing.assert_allclose(full.wraps, expected, rtol=0.0, atol=1.0e-10, err_msg=label)

    def test_route_samples_are_equally_spaced_on_the_center_line(self):
        """Sample a route and check that the points are equally spaced along the cable center line."""
        digit = self.spec.digits["ring"]
        route = ch.digit_route(digit, (0.4, 0.7, 0.3))
        points = route.sample(4001)
        np.testing.assert_allclose(points[0], route.centers[0], atol=1.0e-12)
        np.testing.assert_allclose(points[-1], route.centers[-1], atol=1.0e-12)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.assertAlmostEqual(float(np.sum(steps)), route.length, delta=2.0e-7)  # chords are shorter than arcs
        self.assertLess(float(np.max(steps) - np.min(steps)), 1.0e-9)
        # every sample is on a station's center-line circle or outside all of them
        for k in range(1, len(route.centers) - 1):
            dist = np.linalg.norm(points - route.centers[k], axis=1)
            self.assertGreater(float(np.min(dist)), route.radii[k] - 1.0e-9)

    def test_station_centers_follow_the_links(self):
        """Flex a digit and check that every station center moves with the body that carries it."""
        digit = self.spec.digits["ring"]
        q = (0.5, 0.25, 0.75)
        origins, angles = ch.link_frames(digit, q)
        np.testing.assert_allclose(angles, [0.0, 0.5, 0.75, 1.5], atol=1.0e-15)
        first = digit.links[0].length
        np.testing.assert_allclose(origins[2], first * np.array([math.cos(0.5), math.sin(0.5)]), atol=1.0e-15)
        centers = ch.station_centers(digit, q, travel=0.003)
        names = [station.name for station in digit.stations]
        np.testing.assert_allclose(centers[names.index("S2")], origins[2], atol=1.0e-15)
        np.testing.assert_allclose(centers[names.index("S3")], origins[3], atol=1.0e-15)
        anchor = np.array(digit.slider.rest_anchor) + 0.003 * np.array(digit.slider.pull_direction)
        np.testing.assert_allclose(centers[0], anchor, atol=1.0e-15)

    def test_finger_frames_in_palm(self):
        """Check origin, axes and splay direction of the four finger frames in the palm frame."""
        for name in ("index", "middle", "ring", "little"):
            mount = self.spec.digits[name].mount
            origin, rot = ch.digit_frame_in_palm(self.spec, name)
            np.testing.assert_allclose(origin, mount.frame_origin, atol=1.0e-15)
            np.testing.assert_allclose(rot[:, 0], mount.frame_x, atol=1.0e-9)
            np.testing.assert_allclose(rot[:, 1], mount.frame_y, atol=1.0e-9)
            np.testing.assert_allclose(rot[:, 2], mount.frame_z, atol=1.0e-9)
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1.0e-12)
            self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, delta=1.0e-12)
        # a positive splay turns the finger away from the thumb side (+z); the index is turned toward it
        self.assertGreater(ch.digit_frame_in_palm(self.spec, "index")[1][2, 0], 0.0)
        self.assertLess(ch.digit_frame_in_palm(self.spec, "little")[1][2, 0], 0.0)
        tip = ch.lift_points([0.095, 0.0], *ch.digit_frame_in_palm(self.spec, "index"))
        mount = self.spec.digits["index"].mount
        self.assertEqual(list(mount.frame_origin), REF2_LAYOUT["finger_origin"]["index"])
        np.testing.assert_allclose(tip, np.array(mount.frame_origin) + 0.095 * np.array(mount.frame_x), atol=1.0e-9)

    def test_thumb_frame_turns_about_the_hinge_line(self):
        """Turn the thumb frame by an opposition angle and check that the hinge line stays where it is."""
        thumb = self.spec.digits["thumb"]
        mount = thumb.mount
        origin0, rot0 = ch.digit_frame_in_palm(self.spec, "thumb")
        np.testing.assert_allclose(origin0, mount.frame_origin_at_phi0, atol=1.0e-15)
        np.testing.assert_allclose(rot0[:, 0], mount.frame_x_at_phi0, atol=1.0e-15)
        np.testing.assert_allclose(rot0[:, 2], mount.frame_z_at_phi0, atol=1.0e-15)
        # the digit frame is right-handed; at zero opposition the flexion axis is the palm normal, tilted by
        # the 6 deg of the thumb hinge
        np.testing.assert_allclose(np.cross(rot0[:, 0], rot0[:, 1]), rot0[:, 2], atol=1.0e-9)
        self.assertAlmostEqual(math.degrees(math.acos(float(rot0[1, 2]))), 6.0, delta=0.4)
        self.assertAlmostEqual(float(rot0[2, 2]), 0.0, delta=1.0e-9)
        for phi in (0.0, 0.7, mount.hinge_upper):
            origin, rot = ch.digit_frame_in_palm(self.spec, "thumb", phi)
            # the slider line is the hinge line: its tangent point and direction do not move with the opposition
            point = ch.lift_points(thumb.slider.line_tangent_point, origin, rot)
            np.testing.assert_allclose(point, mount.hinge_point, atol=2.0e-9)
            pull = rot[:, :2] @ np.array(thumb.slider.pull_direction)
            np.testing.assert_allclose(pull, -np.array(mount.slider_line_direction), atol=1.0e-9)
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1.0e-9)
        # positive opposition lifts the straight thumb to the palmar side (+y)
        tip = ch.lift_points([0.101, 0.0], *ch.digit_frame_in_palm(self.spec, "thumb", 0.5 * math.pi))
        self.assertGreater(tip[1], 0.08)
        # at zero opposition the straight thumb lies beside the palm, its tip within 5 mm of the hinge height
        self.assertAlmostEqual(ch.lift_points([0.101, 0.0], origin0, rot0)[1], mount.hinge_point[1], delta=5.0e-3)

    def test_frame_to_transform(self):
        """Convert an origin and a rotation matrix to a transform and check the points it maps."""
        origin, rot = ch.digit_frame_in_palm(self.spec, "thumb", 1.1)
        xform = ch.frame_to_transform(origin, rot)
        point = wp.transform_point(xform, wp.vec3(0.01, -0.02, 0.03))
        np.testing.assert_allclose(np.array(point), origin + rot @ np.array([0.01, -0.02, 0.03]), atol=1.0e-6)


class TestCableHandRodAndTendonData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_rod_station_radii(self):
        """Check barrel, vertex and flange radius of every rod station of every cable against the barrel rule."""
        rod_radius = self.spec.rod.radius
        for name in DIGIT_NAMES:
            digit = self.spec.digits[name]
            for c, cable in enumerate(digit.cables):
                stations = ch.rod_stations(self.spec, name, cable=c)
                contact = [station for station in cable.stations if station.radius > 0.0]
                self.assertEqual([s.station for s in stations], [station.name for station in contact], name)
                _count, seg = ch.rod_segments(self.spec, name, cable=c)
                for station, spec_station in zip(stations, contact, strict=True):
                    self.assertEqual(station.centerline_radius, spec_station.radius)
                    barrel, vertex = ch.rod_barrel_radii(spec_station.radius, seg, rod_radius)
                    self.assertAlmostEqual(station.collision_barrel_radius, barrel, delta=1.0e-12)
                    self.assertAlmostEqual(station.start_vertex_radius, vertex, delta=1.0e-12)
                    self.assertLess(station.collision_barrel_radius, station.centerline_radius - rod_radius)
                    self.assertGreater(station.start_vertex_radius, station.centerline_radius)
                    self.assertGreater(station.flange_radius, station.centerline_radius + rod_radius)

    def test_barrel_rule_takes_up_the_nominal_radius(self):
        """Check that a segment chain on the corrected barrel takes up the nominal radius per radian."""
        # a segment of length l wrapped with its mid-point at b subtends 2 atan(l / (2 b)) = l / R
        for radius in (0.008, 0.0065, 0.005):
            seg = 0.0019894
            barrel, vertex = ch.rod_barrel_radii(radius, seg, 0.0005)
            b = barrel + 0.0005
            self.assertAlmostEqual(seg / (2.0 * math.atan(seg / (2.0 * b))), radius, delta=1.0e-14)
            self.assertAlmostEqual(vertex, math.hypot(b, 0.5 * seg), delta=1.0e-15)

    def test_rod_start_pose(self):
        """Check segment count, spacing, end points and slider travel of the rod start pose of every digit."""
        rod_radius = self.spec.rod.radius
        cables = [(d, c) for d in DIGIT_NAMES for c in range(len(self.spec.digits[d].cables))]
        for digit_name, c in cables:
            digit = self.spec.digits[digit_name]
            cable = digit.cables[c]
            name = f"{digit_name} {cable.name}"
            count, seg = ch.rod_segments(self.spec, digit_name, cable=c)
            pose = ch.rod_start_pose(self.spec, digit_name, cable=c)
            points = pose.points
            self.assertEqual(points.shape, (count + 1, 2), name)
            self.assertEqual(pose.segment_length, seg)
            steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            np.testing.assert_allclose(steps, seg, rtol=0.0, atol=1.0e-11, err_msg=name)
            # ends: the end anchor, and the slider anchor at the returned (small) travel
            centers = ch.station_centers(digit, travel=pose.travel, cable=c)
            np.testing.assert_allclose(points[-1], centers[-1], atol=1.0e-12, err_msg=name)
            np.testing.assert_allclose(points[0], centers[0], atol=1.0e-11, err_msg=name)
            self.assertLess(abs(pose.travel), 5.0e-5, name)
            self.assertGreater(pose.travel, cable.slider.travel_min, name)
            # no vertex inside a collision barrel, and no segment cuts into one
            contact = [k for k, station in enumerate(cable.stations) if station.radius > 0.0]
            for k, station in zip(contact, ch.rod_stations(self.spec, digit_name, cable=c), strict=True):
                dist = np.linalg.norm(points - centers[k], axis=1)
                self.assertGreater(
                    float(np.min(dist)), station.collision_barrel_radius + rod_radius, f"{name} {station.station}"
                )
                closest = min(_point_segment_distance(points[i], points[i + 1], centers[k]) for i in range(count))
                self.assertGreater(
                    closest, station.collision_barrel_radius + rod_radius - 1.0e-9, f"{name} {station.station}"
                )
                # wrapped segments touch the barrel: mid-points sit at barrel radius + rod radius
                self.assertLess(
                    closest, station.collision_barrel_radius + rod_radius + 1.0e-9, f"{name} {station.station}"
                )

    def test_rod_start_pose_at_a_flexed_pose_and_in_3d(self):
        """Check the rod start pose of a flexed digit and its placement by a 3D digit frame."""
        q = (0.6, 0.9, 0.4)
        count, seg = ch.rod_segments(self.spec, "ring")
        origin, rot = ch.digit_frame_in_palm(self.spec, "ring")
        pose = ch.rod_start_pose_3d(self.spec, "ring", origin, rot, q)
        self.assertEqual(pose.points.shape, (count + 1, 3))
        steps = np.linalg.norm(np.diff(pose.points, axis=0), axis=1)
        np.testing.assert_allclose(steps, seg, rtol=0.0, atol=1.0e-11)
        # the slider has taken up the cable that the flexed joints released
        self.assertAlmostEqual(pose.travel, ch.cable_takeup(self.spec.digits["ring"], q), delta=5.0e-5)
        # all vertices lie in the digit mid-plane
        np.testing.assert_allclose((pose.points - origin) @ rot[:, 2], 0.0, atol=1.0e-12)
        # the first vertex is the slider anchor at the returned travel, the last one the tip anchor
        centers = ch.lift_points(ch.station_centers(self.spec.digits["ring"], q, pose.travel), origin, rot)
        np.testing.assert_allclose(pose.points[0], centers[0], atol=1.0e-11)
        np.testing.assert_allclose(pose.points[-1], centers[-1], atol=1.0e-12)
        # with a given travel the rod keeps its spacing and its first vertex stays on the slider line
        given = ch.rod_start_pose(self.spec, "ring", q, travel=pose.travel + 0.001)
        steps = np.linalg.norm(np.diff(given.points, axis=0), axis=1)
        np.testing.assert_allclose(steps, seg, rtol=0.0, atol=1.0e-11)
        anchor = ch.station_centers(self.spec.digits["ring"], q, given.travel)[0]
        self.assertAlmostEqual(float(np.linalg.norm(given.points[0] - anchor)), 0.001, delta=1.0e-6)

    def test_tendon_links(self):
        """Compare type, body, radius, orientation and compliance of every tendon link with the frozen values."""
        cable = self.spec.cable
        for name in ONE_CABLE_DIGITS:
            digit = self.spec.digits[name]
            links = ch.tendon_links(self.spec, name)
            self.assertEqual(len(links), len(digit.stations))
            self.assertEqual(links[0].type, "ATTACHMENT")
            self.assertEqual(links[-1].type, "ATTACHMENT")
            self.assertEqual(links[0].body, "slider")
            self.assertEqual(links[-1].body, "link3")
            self.assertEqual(links[0].compliance, 0.0)
            rolling = [link for link in links if link.type == "ROLLING"]
            self.assertEqual(len(rolling), 5 if name == "thumb" else 6)
            for link, station in zip(links, digit.stations, strict=True):
                self.assertEqual(link.offset, (station.pos[0], station.pos[1], 0.0))
                if station.kind == "guide":
                    self.assertEqual(link.orientation, 1)
                elif station.kind == "sheave":
                    self.assertEqual(link.orientation, -1)
                self.assertEqual(link.radius, station.radius)
            total = REF_GEOMETRY[name]["cable_length"] / cable.EA + cable.anchor_compliance
            self.assertAlmostEqual(sum(link.compliance for link in links), total, delta=1.0e-15)
            np.testing.assert_allclose(
                [link.compliance for link in links], REF_GEOMETRY[name]["tendon_compliance"], rtol=0.0, atol=1.0e-15
            )


class TestCableHandStatics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_frictionless_table_all_digits(self):
        """Compare joint angles, take-up and travel of the one-cable digits with the frozen values."""
        for name in ONE_CABLE_DIGITS:
            digit = self.spec.digits[name]
            for i, tension in enumerate(REF_STAIRCASE):
                row = ch.frictionless_row(self.spec, name, tension)
                np.testing.assert_allclose(row["q"], REF_FRICTIONLESS_Q[i], rtol=0.0, atol=ANGLE_TOL)
                np.testing.assert_allclose(ch.frictionless_angles(digit, tension), row["q"], rtol=0.0, atol=0.0)
                self.assertAlmostEqual(row["takeup"], REF_FRICTIONLESS_TAKEUP[i], delta=LENGTH_TOL)
                self.assertAlmostEqual(row["travel"], REF_FRICTIONLESS_TRAVEL[name][i], delta=LENGTH_TOL)

    def test_spring_thresholds(self):
        """Check the tensions at which each joint starts to move and reaches its upper stop."""
        # joints start to move at 6 / 10 / 16 N and are fully flexed at 14 / 22 / 30 N
        digit = self.spec.digits["ring"]
        for j, (start, full) in enumerate(((6.0, 14.0), (10.0, 22.0), (16.0, 30.0))):
            self.assertEqual(ch.frictionless_angles(digit, start - 1.0e-6)[j], 0.0)
            self.assertGreater(ch.frictionless_angles(digit, start + 1.0e-3)[j], 0.0)
            self.assertLess(ch.frictionless_angles(digit, full - 1.0e-3)[j], digit.joints[j].upper)
            self.assertAlmostEqual(ch.frictionless_angles(digit, full + 1.0e-6)[j], digit.joints[j].upper, delta=1.0e-9)

    def test_friction_tables_of_a_one_cable_digit(self):
        """Check the loading and unloading capstan tables of the ring finger: capstan law, lag and hysteresis."""
        digit = self.spec.digits["ring"]
        rest = ch.digit_route(digit)
        sheaves = ch.sheave_station_indices(digit)
        for mu in (0.02, 0.05):
            tables = ch.friction_tables(self.spec, "ring", REF_STAIRCASE, mu)
            self.assertEqual([row["T"] for row in tables["loading"]], REF_STAIRCASE)
            self.assertEqual([row["T"] for row in tables["unloading"]], REF_STAIRCASE[::-1])
            for row in tables["loading"]:
                # capstan law over the whole route: the wrap of a joint sheave is its straight value less q
                wrap = float(np.sum(rest.wraps)) - sum(row["q"])
                self.assertAlmostEqual(row["tip_tension"], row["T"] * math.exp(-mu * wrap), delta=1.0e-9)
                # with friction the finger lags the frictionless table while loading
                free = ch.frictionless_angles(digit, row["T"])
                self.assertTrue(all(a <= b + 1.0e-12 for a, b in zip(row["q"], free, strict=True)))
                self.assertEqual(len(row["joint_span_tension"]), len(sheaves))
            # the limit tension closes the finger with either friction
            np.testing.assert_allclose(tables["loading"][-1]["q"], [j.upper for j in digit.joints], atol=1.0e-9)
            # hysteresis: at the same tension the finger is at least as closed on the way down
            for up, down in zip(tables["loading"], reversed(tables["unloading"]), strict=True):
                self.assertTrue(all(b >= a - 1.0e-12 for a, b in zip(up["q"], down["q"], strict=True)))
            self.assertGreater(tables["unloading"][3]["q"][2], tables["loading"][4]["q"][2] + 0.1)

    def test_zero_friction_reduces_to_the_frictionless_table(self):
        """Check that the capstan statics with zero friction give the frictionless table."""
        tables = ch.friction_tables(self.spec, "ring", REF_STAIRCASE, 0.0)
        for i, tension in enumerate(REF_STAIRCASE):
            ref = ch.frictionless_row(self.spec, "ring", tension)
            for row in (tables["loading"][i], tables["unloading"][-1 - i]):
                np.testing.assert_allclose(row["q"], ref["q"], rtol=0.0, atol=1.0e-9)
                self.assertAlmostEqual(row["travel"], ref["travel"], delta=1.0e-10)
                self.assertAlmostEqual(row["tip_tension"], tension, delta=1.0e-12)

    def test_hinge_opening_is_part_of_the_travel(self):
        """Check that the slider travel includes the opening of the three hinges under the cable tension."""
        # every flexion hinge opens by the hinge compliance times the cable tension at its sheave
        cable = self.spec.cable
        for name in ONE_CABLE_DIGITS:
            length = REF_GEOMETRY[name]["cable_length"]
            for tension in REF_STAIRCASE:
                row = ch.frictionless_row(self.spec, name, tension)
                load = tension - cable.tension_rest
                self.assertAlmostEqual(row["hinge_stretch"], 3.0 * load * 1.0e-5, delta=1.0e-12)
                expected = row["takeup"] + load * (
                    length / cable.EA + cable.anchor_compliance + 3.0 * cable.hinge_compliance
                )
                self.assertAlmostEqual(row["travel"], expected, delta=LENGTH_TOL)
        self.assertAlmostEqual(ch.frictionless_row(self.spec, "ring", cable.tension_rest)["travel"], 0.0, delta=1.0e-15)
        # with friction each hinge carries the tension of the span that leaves its sheave
        cap = ch.CapstanStatics(self.spec, "ring")
        q = cap.solve(26.0, 0.05, cap.LOADING)
        tensions = cap.tensions(26.0, 0.05, cap.LOADING, q)
        stiff = ch.default_hand_spec()
        stiff.cable.hinge_compliance = 0.0
        rigid = ch.CapstanStatics(stiff, "ring")
        hinge = cap.stretch(tensions, q) - rigid.stretch(tensions, q)
        self.assertAlmostEqual(hinge, 1.0e-5 * sum(tensions[k] for k in cap.sheaves), delta=1.0e-12)
        self.assertLess(hinge, 3.0e-5 * 26.0)
        row = ch.friction_tables(self.spec, "ring", REF_STAIRCASE, 0.05)["loading"][5]
        self.assertAlmostEqual(sum(row["joint_span_tension"]), sum(tensions[k] for k in cap.sheaves), delta=1.0e-9)

    def test_joint_torque_uses_the_span_leaving_the_sheave(self):
        """Check with friction that each joint balances the tension of the span that leaves its sheave."""
        # equilibrium of a moving joint: r_j * (tension of the span leaving S_j) = preload_j + k_j q_j
        cap = ch.CapstanStatics(self.spec, "ring")
        digit = self.spec.digits["ring"]
        tension, mu = 12.0, 0.05
        q = cap.solve(tension, mu, cap.LOADING)
        spans = cap.tensions(tension, mu, cap.LOADING, q)
        joint = digit.joints[0]
        self.assertGreater(q[0], 0.0)
        self.assertLess(q[0], joint.upper)
        k = cap.sheaves[0]
        self.assertAlmostEqual(
            joint.sheave_radius * spans[k], joint.spring_preload + joint.spring_stiffness * q[0], delta=1.0e-12
        )
        self.assertGreater(
            abs(joint.sheave_radius * spans[k - 1] - joint.spring_preload - joint.spring_stiffness * q[0]), 1.0e-3
        )

    def test_tip_tension(self):
        """Check the tension that reaches the tip anchor with and without friction."""
        digit = self.spec.digits["ring"]
        q_full = [joint.upper for joint in digit.joints]
        for mu in (0.02, 0.05):
            straight = ch.tip_tension(self.spec, "ring", 1.0, mu, q=(0.0, 0.0, 0.0))
            flexed = ch.tip_tension(self.spec, "ring", 1.0, mu, q=q_full)
            self.assertAlmostEqual(
                straight, math.exp(-mu * REF_GEOMETRY["ring"]["total_wrap_extension"]), delta=1.0e-12
            )
            self.assertAlmostEqual(
                flexed, math.exp(-mu * REF_GEOMETRY["ring"]["total_wrap_full_flexion"]), delta=1.0e-12
            )
        # at the tension limit the finger is closed: the tip sees the limit less the capstan loss when flexed
        self.assertAlmostEqual(
            ch.tip_tension(self.spec, "ring", 60.0, 0.02),
            60.0 * math.exp(-0.02 * REF_GEOMETRY["ring"]["total_wrap_full_flexion"]),
            delta=1.0e-9,
        )
        self.assertEqual(ch.tip_tension(self.spec, "ring", 20.0, 0.0), 20.0)

    def test_gravity_angles_of_a_one_cable_digit(self):
        """Check the joint angles of the ring finger under gravity: direction of the shift and its size."""
        digit = self.spec.digits["ring"]
        # gravity across the finger in its flexion plane: toward the palm it helps the cable close a free
        # joint, away from the palm it holds the joint back; the shift is the gravity torque over the spring
        for tension in (10.0, 12.0):
            free = ch.frictionless_angles(digit, tension)
            down = ch.gravity_angles(digit, tension, (0.0, 1.0))
            up = ch.gravity_angles(digit, tension, (0.0, -1.0))
            self.assertGreater(down[0], free[0])
            self.assertLess(up[0], free[0])
            self.assertLess(down[0] - up[0], math.radians(30.0))
        # a finger at rest tension stays on its lower stops against gravity from either side
        for direction in ((0.0, 1.0), (0.0, -1.0)):
            np.testing.assert_allclose(ch.gravity_angles(digit, 2.0, direction), 0.0, atol=1.0e-12)
        # without gravity the frictionless angles are returned
        np.testing.assert_allclose(
            ch.gravity_angles(digit, 12.0, (0.0, 1.0), gravity=0.0), ch.frictionless_angles(digit, 12.0), atol=1.0e-12
        )

    def test_gravity_table_includes_the_carriage_weight(self):
        """Check that the weight of the actuator carriage along the pull direction enters the gravity angles."""
        # hanging digit (gravity along digit +x, against the pull direction): the carriage weight comes off
        # the cable tension, so every free joint is lower by weight * moment arm / spring stiffness
        digit = self.spec.digits["ring"]
        weight = digit.slider.mass * ch.GRAVITY
        self.assertAlmostEqual(weight, 0.006 * 9.81, delta=1.0e-12)
        self.assertEqual(tuple(digit.slider.pull_direction), (-1.0, 0.0))
        massless = ch.HandSpec.from_dict(self.spec.to_dict()).digits["ring"]
        massless.slider.mass = 0.0
        for tension in (8.0, 12.0, 20.0):
            q = ch.gravity_angles(digit, tension, (1.0, 0.0))
            # the same digit without carriage mass at the reduced tension
            np.testing.assert_allclose(q, ch.gravity_angles(massless, tension - weight, (1.0, 0.0)), atol=1.0e-12)
            # a free first joint (below 14 N) drops by a little less than weight * r / k = 0.66 deg (the
            # hanging links add their own restoring torque to the spring)
            if tension < 14.0:
                drop = ch.gravity_angles(massless, tension, (1.0, 0.0))[0] - q[0]
                joint = digit.joints[0]
                spring_only = weight * joint.sheave_radius / joint.spring_stiffness
                self.assertGreater(drop, 0.75 * spring_only)
                self.assertLess(drop, spring_only)
        # pointing up, the carriage weight adds to the tension
        up = ch.gravity_angles(digit, 12.0, (-1.0, 0.0))
        np.testing.assert_allclose(up, ch.gravity_angles(massless, 12.0 + weight, (-1.0, 0.0)), atol=1.0e-12)

    def test_staircase(self):
        """Check the levels of the tension staircase and the rows and branches of its reference table."""
        levels = ch.staircase_levels(self.spec)
        self.assertEqual(levels, [2.0, 8.0, 12.0, 16.0, 20.0, 26.0, 32.0, 60.0, 32.0, 26.0, 20.0, 16.0, 12.0, 8.0, 2.0])
        for mu in (0.0, 0.02):
            table = ch.staircase_table(self.spec, "ring", mu)
            self.assertEqual([row["T"] for row in table], levels)
            self.assertEqual([row["branch"] for row in table], ["loading"] * 8 + ["unloading"] * 7)
        table = ch.staircase_table(self.spec, "ring", 0.02)
        unloading = ch.friction_tables(self.spec, "ring", REF_STAIRCASE, 0.02)["unloading"]
        self.assertAlmostEqual(table[9]["travel"], unloading[2]["travel"], delta=LENGTH_TOL)
        table = ch.staircase_table(self.spec, "thumb")
        self.assertAlmostEqual(table[3]["travel"], REF_FRICTIONLESS_TRAVEL["thumb"][3], delta=LENGTH_TOL)
        self.assertEqual(table[3]["tip_tension"], 16.0)


class TestCableHandSkeleton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def _check_single_finger(self, name, base):
        digit = self.spec.digits[name]
        q = (0.3, 0.5, 0.2)
        travel = 0.004
        base_xform = wp.transform(
            wp.vec3(0.1, 0.2, 0.3), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
        )
        model, skeleton = ch.build_single_finger_model(
            self.spec, name, base=base, base_xform=base_xform, q=q, travel=travel, device=CPU
        )
        fixed = base == "fixed"
        self.assertEqual(model.body_count, 5)
        self.assertEqual(model.joint_count, 5 if fixed else 4)
        self.assertEqual(model.joint_dof_count, 4)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(set(skeleton.bodies), {"base", "link1", "link2", "link3", "slider"})
        self.assertEqual(len(set(skeleton.bodies.values())), 5)

        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        parents = ("base", "link1", "link2")
        for i, joint in enumerate(skeleton.joint_flexion):
            self.assertEqual(joint_type[joint], newton.JointType.REVOLUTE)
            self.assertEqual(joint_parent[joint], skeleton.bodies[parents[i]])
            self.assertEqual(joint_child[joint], skeleton.bodies[f"link{i + 1}"])
        self.assertEqual(joint_type[skeleton.joint_slider], newton.JointType.PRISMATIC)
        self.assertEqual(joint_parent[skeleton.joint_slider], skeleton.bodies["base"])
        self.assertEqual(joint_child[skeleton.joint_slider], skeleton.bodies["slider"])
        if fixed:
            self.assertEqual(joint_type[skeleton.base_joint], newton.JointType.FIXED)
        else:
            self.assertIsNone(skeleton.base_joint)

        # limits, spring drive and slider drive
        dofs = [*skeleton.dof_flexion, skeleton.dof_slider]
        lower = model.joint_limit_lower.numpy()[dofs]
        upper = model.joint_limit_upper.numpy()[dofs]
        np.testing.assert_allclose(lower, [0.0, 0.0, 0.0, -0.002], atol=1.0e-9)
        np.testing.assert_allclose(upper, [0.5 * math.pi, math.radians(100.0), math.radians(70.0), 0.042], rtol=1.0e-6)
        np.testing.assert_allclose(model.joint_limit_ke.numpy()[dofs[:3]], 300.0)
        np.testing.assert_allclose(model.joint_limit_kd.numpy()[dofs[:3]], 0.01)
        stiffness = [joint.spring_stiffness for joint in digit.joints]
        damping = [joint.spring_damping for joint in digit.joints]
        np.testing.assert_allclose(model.joint_target_ke.numpy()[dofs], [*stiffness, 2.0e4], rtol=1.0e-6)
        np.testing.assert_allclose(model.joint_target_kd.numpy()[dofs], [*damping, 20.0], rtol=1.0e-6)
        coords = [*skeleton.q_flexion, skeleton.q_slider]
        np.testing.assert_allclose(model.joint_target_q.numpy()[coords], 0.0, atol=0.0)
        np.testing.assert_allclose(model.joint_q.numpy()[coords], [*q, travel], rtol=1.0e-6)

        # masses and centers of mass
        mass = model.body_mass.numpy()
        com = model.body_com.numpy()
        for i, link in enumerate(digit.links):
            body = skeleton.bodies[f"link{i + 1}"]
            self.assertAlmostEqual(float(mass[body]), link.mass, delta=1.0e-8)
            np.testing.assert_allclose(com[body], [0.5 * link.length, 0.0, 0.0], atol=1.0e-9)
        self.assertAlmostEqual(float(mass[skeleton.bodies["slider"]]), 0.006, delta=1.0e-8)
        self.assertAlmostEqual(
            float(mass[skeleton.bodies["base"]]), (0.020 if name == "thumb" else 0.301) if fixed else 0.0, delta=1.0e-8
        )
        inertia = model.body_inertia.numpy()[skeleton.bodies["link1"]]
        link = digit.links[0]
        height = link.y_palmar - link.y_dorsal
        np.testing.assert_allclose(inertia[2, 2], link.mass / 12.0 * (link.length**2 + height**2), rtol=1.0e-5)

        # controller record
        np.testing.assert_allclose(skeleton.spring_preload, [0.048, 0.065, 0.080], rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(skeleton.joint_f_bias, [-0.048, -0.065, -0.080], rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(skeleton.spring_stiffness, stiffness, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(skeleton.spring_damping, damping, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(skeleton.spring_target_q, 0.0, atol=0.0)
        np.testing.assert_allclose(skeleton.sheave_radius, [0.008, 0.0065, 0.005], rtol=0.0, atol=1.0e-15)
        self.assertEqual((skeleton.slider_travel_min, skeleton.slider_travel_max), (-0.002, 0.042))
        self.assertEqual((skeleton.slider_kp, skeleton.slider_kd), (2.0e4, 20.0))
        control = model.control()
        joint_f = ch.write_spring_preload(control.joint_f.numpy(), skeleton)
        self.assertEqual(joint_f.shape, (model.joint_dof_count,))
        np.testing.assert_allclose(joint_f[list(skeleton.dof_flexion)], [-0.048, -0.065, -0.080], rtol=1.0e-6)
        self.assertEqual(joint_f[skeleton.dof_slider], 0.0)

        # body poses: forward kinematics of the joints reproduces the authored poses, and they
        # agree with the planar kinematics lifted through the base pose
        body_q = model.body_q.numpy().copy()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        np.testing.assert_allclose(state.body_q.numpy(), body_q, atol=1.0e-6)
        origins, _angles = ch.link_frames(digit, q)
        for i in range(3):
            expected = wp.transform_point(base_xform, wp.vec3(origins[i + 1][0], origins[i + 1][1], 0.0))
            np.testing.assert_allclose(body_q[skeleton.bodies[f"link{i + 1}"]][:3], np.array(expected), atol=1.0e-6)
        # the tip anchor and the slider anchor sit where the route puts them
        centers = ch.station_centers(digit, q, travel)
        for key, station, center in (
            ("link3", digit.stations[-1], centers[-1]),
            ("slider", digit.stations[0], centers[0]),
        ):
            pose = wp.transform(*body_q[skeleton.bodies[key]].tolist())
            got = wp.transform_point(pose, wp.vec3(station.pos[0], station.pos[1], 0.0))
            expected = wp.transform_point(base_xform, wp.vec3(center[0], center[1], 0.0))
            np.testing.assert_allclose(np.array(got), np.array(expected), atol=1.0e-6)

    def test_single_finger_kinematic_base(self):
        """Build a one-cable finger on a kinematic base and check bodies, joints, springs and start pose."""
        self._check_single_finger("ring", "kinematic")

    def test_single_finger_fixed_base(self):
        """Build a one-cable finger on a fixed base and check bodies, joints, springs and start pose."""
        self._check_single_finger("ring", "fixed")

    def test_single_thumb(self):
        """Build the thumb alone and check bodies, joints, springs and start pose."""
        self._check_single_finger("thumb", "kinematic")

    def test_single_finger_default_pose(self):
        """Build a single finger at its default pose, check its pose, and reject an unknown base kind."""
        model, skeleton = ch.build_single_finger_model(name="ring", device=CPU)
        self.assertEqual(skeleton.name, "ring")
        body_q = model.body_q.numpy()
        # straight finger along world +x, flexion axis along world +z
        links = self.spec.digits["ring"].links
        reach = links[0].length + links[1].length
        np.testing.assert_allclose(body_q[skeleton.bodies["link3"]][:3], [reach, 0.0, 0.0], atol=1.0e-7)
        np.testing.assert_allclose(body_q[skeleton.bodies["link3"]][3:], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7)
        np.testing.assert_allclose(model.joint_q.numpy(), 0.0, atol=0.0)
        with self.assertRaises(ValueError):
            ch.build_single_finger_model(base="floating", device=CPU)

    def test_two_digits_on_one_palm(self):
        """Add two digit skeletons to one base body and check that each sits in its own frame."""
        builder = newton.ModelBuilder()
        palm = builder.add_link(is_kinematic=True, label="palm")
        skeletons = []
        for name in ("ring", "little"):
            xform = ch.frame_to_transform(*ch.digit_frame_in_palm(self.spec, name))
            skeletons.append(ch.add_digit_skeleton(builder, self.spec, name, palm, base_xform_digit=xform))
        builder.add_articulation([joint for skeleton in skeletons for joint in skeleton.joints])
        model = builder.finalize(device=CPU)
        self.assertEqual(model.body_count, 9)
        self.assertEqual(model.joint_count, 8)
        self.assertEqual(skeletons[1].dof_flexion, (4, 5, 6))
        joint_f = ch.write_spring_preload(np.zeros(model.joint_dof_count), skeletons)
        np.testing.assert_allclose(joint_f, [-0.048, -0.065, -0.08, 0.0, -0.048, -0.065, -0.08, 0.0])
        # the little finger's MCP sits at its mount and the finger points along its splayed x axis
        body_q = model.body_q.numpy()
        origin, rot = ch.digit_frame_in_palm(self.spec, "little")
        np.testing.assert_allclose(body_q[skeletons[1].bodies["link1"]][:3], origin, atol=1.0e-7)
        np.testing.assert_allclose(body_q[skeletons[1].bodies["link2"]][:3], origin + 0.036 * rot[:, 0], atol=1.0e-7)


class TestCableHandRodFinger(unittest.TestCase):
    """Rod cable block and the shared control block (governor, staircase, preload)."""

    @classmethod
    def setUpClass(cls):
        # a one-cable finger with a rod cable (1 mm segments)
        cls.spec = ch.default_hand_spec()
        cls.model, cls.skeleton, cls.cable = ch.build_single_rod_finger_model(cls.spec, "ring", device=CPU)
        cls.count, cls.segment = ch.rod_segments(cls.spec, "ring")

    def test_counts(self):
        """Check body, joint, shape and articulation counts of the rod ring finger and the rod thumb."""
        model, cable, count = self.model, self.cable, self.count
        # 150.7 mm of cable in 1 mm segments
        self.assertEqual(count, round(REF_GEOMETRY["ring"]["cable_length"] / self.spec.rod.segment_length_target))
        self.assertEqual(model.body_count, 5 + count)  # base, three links, slider and the rod segments
        self.assertEqual(len(cable.bodies), count)
        self.assertEqual(model.joint_count, 4 + (count - 1) + 2)
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(model.shape_count, 6 * 3 + count)
        thumb, _, thumb_cable = ch.build_single_rod_finger_model(self.spec, "thumb", device=CPU)
        self.assertEqual(thumb.body_count, 5 + len(thumb_cable.bodies))
        self.assertEqual(len(thumb_cable.bodies), ch.rod_segments(self.spec, "thumb")[0])
        self.assertEqual(len(thumb_cable.station_shapes), 5)
        self.assertEqual(thumb.articulation_count, 1)

    def test_fixed_base_and_flexed_start(self):
        """Build the rod finger flexed on a fixed base and check start travel and closed anchors."""
        q = (0.4, 0.6, 0.3)
        model, skeleton, cable = ch.build_single_rod_finger_model(self.spec, "ring", base="fixed", q=q, device=CPU)
        self.assertEqual(model.joint_count, 5 + (self.count - 1) + 2)
        self.assertEqual(model.articulation_count, 1)
        # the start travel of a flexed pose is the cable take-up of that pose
        self.assertAlmostEqual(cable.start_travel, ch.cable_takeup(self.spec.digits["ring"], q), delta=5.0e-5)
        probe = ch.RodTensionProbe(model, skeleton, cable)
        self.assertLess(np.abs(probe.anchor_gaps(model.body_q.numpy())).max(), 1.0e-6)

    def test_station_shapes(self):
        """Check body, position, radius and height of the barrel and flange shapes of every station."""
        model, skeleton, cable = self.model, self.skeleton, self.cable
        hardware = self.spec.station_hardware
        shape_body = model.shape_body.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_xform = model.shape_transform.numpy()
        stations = {s.name: s for s in self.spec.digits["ring"].stations}
        rod_stations = {s.station: s for s in ch.rod_stations(self.spec, "ring")}
        self.assertEqual(list(cable.station_shapes), ["G0", "S1", "G1", "S2", "G2", "S3"])
        for name, (barrel, flange_neg, flange_pos) in cable.station_shapes.items():
            station = stations[name]
            for shape in (barrel, flange_neg, flange_pos):
                self.assertEqual(shape_body[shape], skeleton.bodies[station.body])
                np.testing.assert_allclose(shape_xform[shape][:2], station.pos, atol=1.0e-7)
            self.assertAlmostEqual(shape_scale[barrel][0], rod_stations[name].collision_barrel_radius, delta=1.0e-8)
            self.assertAlmostEqual(shape_scale[barrel][1], 0.0014, delta=1.0e-8)
            for shape, sign in ((flange_neg, -1.0), (flange_pos, 1.0)):
                self.assertAlmostEqual(shape_scale[shape][0], station.radius + 0.0012, delta=1.0e-8)
                self.assertAlmostEqual(shape_scale[shape][1], 0.5 * hardware.flange_thickness, delta=1.0e-8)
                self.assertAlmostEqual(shape_xform[shape][2], sign * 0.0011, delta=1.0e-8)
            # side clearance of the rod between the flange faces
            face = shape_xform[flange_pos][2] - shape_scale[flange_pos][1]
            self.assertAlmostEqual(face - self.spec.rod.radius, 0.0003, delta=1.0e-8)

    def test_collision_pairs(self):
        """Check that every contact pair is one rod capsule against one station shape of the digit."""
        model, cable = self.model, self.cable
        rod = set(cable.rod_shapes)
        stations = {shape for triple in cable.station_shapes.values() for shape in triple}
        self.assertEqual(len(rod), self.count)
        groups = model.shape_collision_group.numpy()
        self.assertTrue(np.all(groups[sorted(rod | stations)] == cable.collision_group))
        self.assertNotEqual(cable.collision_group, 1)  # 1 is the builder default of every other shape
        pairs = model.shape_contact_pairs.numpy()
        self.assertEqual(len(pairs), self.count * 18)
        for a, b in pairs:
            # no rod-rod and no station-station pair: every pair is one rod capsule against one station shape
            self.assertEqual((int(a) in rod) + (int(b) in rod), 1)
            self.assertEqual((int(a) in stations) + (int(b) in stations), 1)

    def test_rod_properties(self):
        """Check mass, inertia, stiffness and anchors of the rod and that its start pose is unstretched."""
        model, skeleton, cable = self.model, self.skeleton, self.cable
        spec = self.spec
        count = self.count
        self.assertAlmostEqual(cable.segment_length, REF_GEOMETRY["ring"]["cable_length"] / count, delta=1.0e-12)
        self.assertAlmostEqual(cable.segment_length, 0.001, delta=5.0e-6)
        mass = model.body_mass.numpy()[cable.bodies]
        self.assertAlmostEqual(mass.sum(), spec.cable.linear_density * count * cable.segment_length, delta=1.0e-8)
        # physical segment inertia, not shifted by finalize()
        inertia = model.body_inertia.numpy()[cable.bodies]
        self.assertLess(np.trace(inertia, axis1=1, axis2=2).max(), 1.0e-11)
        self.assertGreater(np.linalg.eigvalsh(inertia).min(), 1.0e-14)
        probe = ch.RodTensionProbe(model, skeleton, cable)
        np.testing.assert_allclose(probe.stretch_ke, spec.cable.EA / cable.segment_length, rtol=1.0e-5)
        np.testing.assert_allclose(probe.shear_ke, probe.stretch_ke)
        dof = model.joint_qd_start.numpy()[cable.rod_joints[0]]
        np.testing.assert_allclose(
            model.joint_target_ke.numpy()[dof + 3 : dof + 6], spec.cable.bend_rigidity / cable.segment_length, rtol=1e-5
        )
        # anchors: ball joints on the slider and on the distal link
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        for joint, parent, child in zip(
            cable.anchor_joints, ("slider", "link3"), (cable.bodies[0], cable.bodies[-1]), strict=True
        ):
            self.assertEqual(joint_type[joint], newton.JointType.BALL)
            self.assertEqual(joint_parent[joint], skeleton.bodies[parent])
            self.assertEqual(joint_child[joint], child)
        # the start pose is unstretched, closed at both anchors and lies in the digit mid-plane
        body_q = model.body_q.numpy()
        self.assertLess(np.abs(probe.tension(body_q)).max(), 0.05)
        self.assertLess(probe.anchor_gaps(body_q).max(), 1.0e-7)
        self.assertLess(np.abs(probe.lateral_offset(body_q)).max(), 1.0e-7)
        self.assertEqual(probe.lateral_offset(body_q).shape, (count + 1,))
        self.assertAlmostEqual(model.joint_q.numpy()[skeleton.q_slider], cable.start_travel, delta=1.0e-9)

    def test_groove_offset_is_measured_on_the_body_that_carries_the_groove(self):
        """A rod moved sideways with its link reads zero in the groove; moved against it, the displacement."""
        model, skeleton, cable = self.model, self.skeleton, self.cable
        probe = ch.RodTensionProbe(model, skeleton, cable)
        start = model.body_q.numpy().astype(np.float64)
        vertices = self.count + 1
        self.assertEqual(probe.groove_offset(start).shape, (vertices,))
        self.assertLess(np.abs(probe.groove_offset(start)).max(), 1.0e-7)
        d = 2.0e-4
        side = np.array([0.0, 0.0, d])  # the flexion axis of the single finger is world z

        # the rod alone moves: every vertex is off the digit plane, and every vertex in a groove is off it
        against = start.copy()
        against[cable.bodies, :3] += side
        np.testing.assert_allclose(probe.lateral_offset(against), d, atol=1.0e-9)
        offset = probe.groove_offset(against)
        held = offset > 0.5 * d
        np.testing.assert_allclose(offset[held], d, atol=1.0e-9)
        np.testing.assert_allclose(offset[~held], 0.0, atol=1.0e-12)  # free spans
        self.assertGreater(held.sum(), vertices // 3)  # six stations hold more than a third of the vertices
        self.assertLess(held.sum(), vertices)

        # the middle link moves with the rod: its two stations read zero, the others still the displacement
        with_link = against.copy()
        with_link[skeleton.bodies["link2"], :3] += side
        np.testing.assert_allclose(probe.lateral_offset(with_link), d, atol=1.0e-9)
        offset = probe.groove_offset(with_link)
        on_link = held & (np.abs(offset) < 1.0e-9)
        shape_body = model.shape_body.numpy()
        link2 = skeleton.bodies["link2"]
        carried = [name for name, shapes in cable.station_shapes.items() if shape_body[shapes[0]] == link2]
        self.assertEqual(carried, ["G2", "S3"])
        self.assertGreater(on_link.sum(), 5)
        np.testing.assert_allclose(offset[held & ~on_link], d, atol=1.0e-9)

        # every body that carries a station moves with the rod: no vertex is off its groove
        together = against.copy()
        for key in ("base", "link1", "link2", "link3"):
            together[skeleton.bodies[key], :3] += side
        np.testing.assert_allclose(probe.groove_offset(together), 0.0, atol=1.0e-9)

    def test_weight_as_body_forces(self):
        """Check that the rod finger model has no gravity and that the weights are applied as body forces."""
        model, cable = self.model, self.cable
        # the model has no gravity and the caller applies the weights as body forces
        self.assertEqual(cable.weight_gravity, (0.0, 0.0, -ch.GRAVITY))
        np.testing.assert_allclose(model.gravity.numpy(), 0.0)
        state = model.state()
        state.clear_forces()
        ch.create_body_weight(model, cable.weight_gravity).apply(state)
        body_f = state.body_f.numpy()
        np.testing.assert_allclose(body_f[:, 2], -ch.GRAVITY * model.body_mass.numpy(), rtol=1.0e-6)
        np.testing.assert_allclose(np.delete(body_f, 2, axis=1), 0.0)

    def test_start_travel_mismatch_is_rejected(self):
        """Reject a rod cable on a skeleton whose slider is not at the travel of the rod start pose."""
        builder = newton.ModelBuilder()
        skeleton = ch.add_single_finger(builder, self.spec, "ring", travel=0.002)
        with self.assertRaises(ValueError):
            ch.add_rod_cable(builder, self.spec, skeleton)

    def test_governor_rule(self):
        """Step the actuator governor and check its speed limit, force limit, speed guard and both modes."""
        spec, model, skeleton = self.spec, self.model, self.skeleton
        act = spec.actuator
        state = model.state()
        control = model.control()
        dt = 1.0 / 2400.0
        s = self.cable.start_travel
        target = int(model.joint_target_q_start.numpy()[skeleton.joint_slider])

        # carriage speed guard: the reference keeps a margin below the speed limit, and a carriage that moves
        # faster than the reference (here 5 mm/s, faked through the last measured travel) slows it by that much
        gov = ch.create_actuator_governor(model, spec, skeleton, command=act.force_limit)
        self.assertEqual((gov.speed_filter, gov.speed_margin), (ch.GOVERNOR_SPEED_FILTER, ch.GOVERNOR_SPEED_MARGIN))
        gov.step(state, control, dt)
        rate = act.speed_limit - ch.GOVERNOR_SPEED_MARGIN
        self.assertAlmostEqual(gov.ref_rate.numpy()[0], rate, delta=1.0e-4)
        gov.travel.assign(np.array([s - (rate + 0.005) * dt], dtype=np.float32))
        gov.step(state, control, dt)
        self.assertAlmostEqual(gov.ref_rate.numpy()[0], rate - 0.005, delta=2.0e-4)
        self.assertAlmostEqual(gov.speed_excess.numpy()[0][0], 0.005, delta=2.0e-4)

        # the plain rule (guard off). Force mode at the limit: the reference rises at the speed limit and
        # stops at s + force_limit / kp
        gov = ch.create_actuator_governor(
            model, spec, skeleton, command=act.force_limit, speed_filter=0.0, speed_margin=0.0
        )
        gov.step(state, control, dt)
        self.assertAlmostEqual(gov.travel.numpy()[0], s, delta=1.0e-7)
        self.assertAlmostEqual(gov.s_ref.numpy()[0] - s, act.speed_limit * dt, delta=1.0e-8)
        self.assertAlmostEqual(control.joint_target_q.numpy()[target], gov.s_ref.numpy()[0], delta=1.0e-9)
        for _ in range(400):
            gov.step(state, control, dt)
        self.assertAlmostEqual(gov.s_ref.numpy()[0] - s, act.force_limit / act.kp, delta=1.0e-7)
        self.assertAlmostEqual(gov.drive_force.numpy()[0], act.force_limit, delta=2.0e-3)
        # a lower tension command brings the reference down at the same rate
        gov.set_command(20.0)
        before = gov.s_ref.numpy()[0]
        gov.step(state, control, dt)
        self.assertAlmostEqual(before - gov.s_ref.numpy()[0], act.speed_limit * dt, delta=1.0e-8)
        for _ in range(400):
            gov.step(state, control, dt)
        self.assertAlmostEqual(gov.drive_force.numpy()[0], 20.0, delta=2.0e-3)

        # position mode: the goal is clamped between the rest tension and the force limit
        gov = ch.create_actuator_governor(
            model, spec, skeleton, mode=ch.GOVERNOR_POSITION, command=0.010, speed_filter=0.0, speed_margin=0.0
        )
        for _ in range(400):
            gov.step(state, control, dt)
        self.assertAlmostEqual(gov.drive_force.numpy()[0], act.force_limit, delta=2.0e-3)
        gov.set_command(-0.005)
        for _ in range(400):
            gov.step(state, control, dt)
        self.assertAlmostEqual(gov.drive_force.numpy()[0], act.force_min, delta=2.0e-3)
        gov.set_command(s + 0.001)
        for _ in range(400):
            gov.step(state, control, dt)
        self.assertAlmostEqual(gov.s_ref.numpy()[0], s + 0.001, delta=1.0e-7)

    def test_staircase_and_preload(self):
        """Step the staircase command through its levels and check the spring preload written to the control."""
        spec, model, skeleton = self.spec, self.model, self.skeleton
        stair = ch.create_staircase_command(spec, device=model.device, blend=0.0)
        self.assertEqual(stair.levels.shape[0], 15)
        self.assertAlmostEqual(stair.period, 1.0)
        self.assertAlmostEqual(stair.duration, 15.0)
        self.assertEqual(stair.hold_window(2), (2.5, 3.0))
        command = wp.zeros(1, dtype=float, device=model.device)
        seen = []
        for _ in range(16 * 4):
            stair.step(command, 0.25)
            seen.append(float(command.numpy()[0]))
        # without a blend the command steps from level to level and keeps the last one
        self.assertEqual(seen[::4], [*ch.staircase_levels(spec), 2.0])
        self.assertEqual(int(stair.level_index.numpy()[0]), 14)

        # Default: the command moves to a new level over the blend time with the minimum-jerk curve of the
        # keyframe scripts. It starts at the old level and ends at the new one, with zero rate at both ends,
        # and is monotonic in between; the hold then sits on the level.
        levels = [[2.0, 2.0], [18.0, 2.0], [6.0, 28.0]]
        blended = ch.create_staircase_command(spec, levels=levels, hold=0.5, transition=0.75, device=model.device)
        self.assertEqual((blended.blend, blended.cyclic), (ch.STAIRCASE_BLEND, False))
        self.assertEqual(ch.STAIRCASE_BLEND, 0.6)
        self.assertEqual(ch.create_staircase_command(spec, device=model.device).blend, 0.5)  # capped by the move
        dt = 0.0025
        pair = wp.zeros(2, dtype=float, device=model.device)
        trace = []
        for _ in range(round(3.0 * blended.period / dt)):
            blended.step(pair, dt)
            trace.append(pair.numpy().copy())
        trace = np.array(trace)
        steps = round(blended.period / dt)
        np.testing.assert_allclose(trace[:steps], [levels[0]] * steps, atol=1.0e-4)  # no blend into the first level
        for k in (1, 2):
            move = trace[k * steps : k * steps + round(blended.blend / dt) + 1]
            np.testing.assert_allclose(move[0], levels[k - 1], atol=1.0e-4)
            np.testing.assert_allclose(move[-1], levels[k], atol=1.0e-4)
            np.testing.assert_allclose(trace[(k + 1) * steps - 1], levels[k], atol=1.0e-4)
            for column in range(2):
                rise = levels[k][column] - levels[k - 1][column]
                rate = np.diff(move[:, column]) / dt
                if rise == 0.0:
                    np.testing.assert_allclose(rate, 0.0, atol=1.0e-3)
                    continue
                self.assertTrue(np.all(np.sign(rise) * rate >= -1.0e-4))  # monotonic
                # zero rate at both ends, and the minimum-jerk peak of 1.875 times the mean rate in the middle
                self.assertLess(abs(rate[0]), 0.01 * abs(rise) / blended.blend)
                self.assertLess(abs(rate[-1]), 0.01 * abs(rise) / blended.blend)
                self.assertAlmostEqual(np.abs(rate).max(), 1.875 * abs(rise) / blended.blend, delta=0.01 * abs(rise))
            # the kernel and the host evaluation agree
            t_mid = k * blended.period + 0.5 * blended.blend
            np.testing.assert_allclose(trace[round(t_mid / dt)], blended.evaluate(t_mid), atol=1.0e-4)
            np.testing.assert_allclose(
                blended.evaluate(t_mid), 0.5 * (np.array(levels[k - 1]) + levels[k]), atol=1.0e-5
            )
        # a cyclic staircase starts again and blends from its last level to its first
        loop = ch.create_staircase_command(
            spec, levels=levels, hold=0.5, transition=0.75, device=model.device, cyclic=True
        )
        np.testing.assert_allclose(loop.evaluate(3.0 * loop.period), levels[2], atol=1.0e-5)
        np.testing.assert_allclose(loop.evaluate(3.0 * loop.period + loop.blend), levels[0], atol=1.0e-5)
        np.testing.assert_allclose(blended.evaluate(3.0 * blended.period + blended.blend), levels[2], atol=1.0e-5)
        with self.assertRaises(ValueError):
            ch.create_staircase_command(spec, hold=0.5, transition=0.3, blend=0.5)

        control = model.control()
        ch.create_spring_preload(model, skeleton).apply(control)
        expected = ch.write_spring_preload(np.zeros(model.joint_dof_count), skeleton)
        np.testing.assert_allclose(control.joint_f.numpy(), expected, rtol=1.0e-6)

    def test_rest_tension_cuda(self):
        """Simulate the rod finger for 0.2 s at the rest tension on CUDA and check tension, angles and grooves."""
        if CUDA is None:
            self.skipTest("needs CUDA")
        spec = self.spec
        model, skeleton, cable = ch.build_single_rod_finger_model(spec, "ring", device=CUDA)
        dt = 1.0 / 2400.0
        pipeline = ch.create_rod_collision_pipeline(model)
        solver = ch.create_rod_solver(
            model,
            dt,
            inertia_reference=ch.rod_inertia_reference(spec),
            rigid_body_contact_buffer_size=ch.rod_contact_buffer(spec, "ring"),
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = pipeline.contacts()
        gov = ch.create_actuator_governor(model, spec, skeleton)
        ch.create_spring_preload(model, skeleton).apply(control)
        weight = ch.create_body_weight(model, cable.weight_gravity)
        for _ in range(12 * 40):
            state_0.clear_forces()
            weight.apply(state_0)
            gov.step(state_0, control, dt)
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        self.assertTrue(np.isfinite(body_q).all())
        probe = ch.RodTensionProbe(model, skeleton, cable)
        tension = probe.tension(body_q)
        np.testing.assert_allclose(tension, spec.cable.tension_rest, atol=0.2)
        joint_q = wp.clone(model.joint_q)
        joint_qd = wp.clone(model.joint_qd)
        newton.eval_ik(model, state_0, joint_q, joint_qd)
        np.testing.assert_allclose(joint_q.numpy()[list(skeleton.q_flexion)], 0.0, atol=math.radians(0.05))
        self.assertLess(np.abs(probe.groove_offset(body_q)).max(), 0.0003)
        self.assertEqual(int(solver.rigid_articulation_pivot_clamped.numpy()[0]), 0)


class TestCableHandTendonFinger(unittest.TestCase):
    """Single one-cable digit with the massless tendon: model, route in the solver, rest-tension hold (CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def _build(self, name, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, skeleton, tendon = ch.build_tendon_finger_model(self.spec, name, device=CPU, **kwargs)
            solver = ch.make_tendon_solver(model)
        self.assertEqual([str(w.message) for w in caught], [])
        return model, skeleton, tendon, solver

    def test_links_match_the_frozen_list(self):
        """Check the tendon links, compliances and joint stops of every one-cable digit's model against the spec."""
        rolling, attachment = int(newton.TendonLinkType.ROLLING), int(newton.TendonLinkType.ATTACHMENT)
        cable = self.spec.cable
        for name in ONE_CABLE_DIGITS:
            model, skeleton, tendon, _solver = self._build(name)
            stations = ["A", "G0", "S1", "G1", "S2", "G2", "S3", "T"]
            if name == "thumb":
                stations.remove("G0")
            self.assertEqual(tendon.stations, stations)
            self.assertEqual(model.tendon_count, 1)
            self.assertEqual(model.tendon_link_count, len(stations))
            self.assertEqual(model.tendon_segment_count, len(stations) - 1)
            types = model.tendon_link_type.numpy()
            bodies = model.tendon_link_body.numpy()
            orientation = model.tendon_link_orientation.numpy()
            radius = model.tendon_link_radius.numpy()
            offset = model.tendon_link_offset.numpy()
            axis = model.tendon_link_axis.numpy()
            for k, link in enumerate(ch.tendon_links(self.spec, name)):
                station = link.station
                self.assertEqual(int(types[k]), attachment if station in ("A", "T") else rolling)
                self.assertEqual(int(bodies[k]), skeleton.bodies[link.body])
                np.testing.assert_allclose(offset[k], link.offset, rtol=0.0, atol=1.0e-8)
                np.testing.assert_allclose(axis[k], (0.0, 0.0, 1.0), rtol=0.0, atol=0.0)
                self.assertAlmostEqual(float(radius[k]), link.radius, delta=1.0e-9)
                if station.startswith("G"):
                    self.assertEqual(int(orientation[k]), 1)
                elif station.startswith("S"):
                    self.assertEqual(int(orientation[k]), -1)
            self.assertEqual(model.tendon_link_mu.numpy().max(), 0.0)
            compliance = model.tendon_seg_compliance.numpy()
            total = REF_GEOMETRY[name]["cable_length"] / cable.EA + cable.anchor_compliance
            self.assertAlmostEqual(float(compliance.sum()), total, delta=1.0e-10)
            np.testing.assert_allclose(compliance, REF_GEOMETRY[name]["tendon_compliance"][1:], rtol=1.0e-6)
            np.testing.assert_allclose(tendon.compliance, compliance, rtol=1.0e-6)
            np.testing.assert_allclose(model.tendon_seg_damping.numpy(), self.spec.tendon.damping)
            # the joint stops are those of the spec: soft, because the local solve stalls on stiff ones
            dofs = list(skeleton.dof_flexion)
            np.testing.assert_allclose(model.joint_limit_ke.numpy()[dofs], 300.0)
            np.testing.assert_allclose(model.joint_limit_kd.numpy()[dofs], 0.01)
        # other stops in the spec reach the model; the option limit_gains replaces them
        spec = ch.default_hand_spec()
        for joint in spec.digits["ring"].joints:
            joint.limit_ke = 1234.0
        model, skeleton, _tendon = ch.build_tendon_finger_model(spec, "ring", device=CPU)
        np.testing.assert_allclose(model.joint_limit_ke.numpy()[list(skeleton.dof_flexion)], 1234.0)
        model, skeleton, _tendon = ch.build_tendon_finger_model(spec, "ring", device=CPU, limit_gains=(500.0, 0.02))
        np.testing.assert_allclose(model.joint_limit_ke.numpy()[list(skeleton.dof_flexion)], 500.0)

    def test_friction_is_set_on_the_rolling_links_only(self):
        """Build a tendon finger with friction and check that only the rolling links carry it."""
        model, _skeleton, tendon, solver = self._build("ring", mu=0.05)
        mu = model.tendon_link_mu.numpy()
        np.testing.assert_allclose(mu, [0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0])
        self.assertEqual(tendon.mu, 0.05)
        self.assertTrue(solver.tendon_material_direct)
        self.assertTrue(ch.tendon_solver_settings(True)["tendon_material_direct"])
        self.assertNotIn("tendon_material_direct", ch.tendon_solver_settings(False))
        # the cable is built in the loading state of the friction table at the rest tension
        # (the actuator span is also cut shorter by the opening of the three hinges under that tension)
        spans = np.array(ch.digit_route(self.spec.digits["ring"]).spans)
        cap = ch.CapstanStatics(self.spec, "ring")
        loading = cap.tensions(self.spec.cable.tension_rest, 0.05, cap.LOADING, (0.0, 0.0, 0.0))
        spans[0] -= self.spec.cable.hinge_compliance * sum(loading[k] for k in cap.sheaves)
        tension = (spans - model.tendon_seg_rest_length.numpy()[tendon.segments]) / tendon.compliance
        np.testing.assert_allclose(tendon.rest_length, model.tendon_seg_rest_length.numpy(), rtol=1.0e-6)
        self.assertAlmostEqual(float(tension[0]), self.spec.cable.tension_rest, delta=0.01)
        # at rest the tip sees the rest tension less the capstan loss over the wrap of the straight finger
        tip = self.spec.cable.tension_rest * math.exp(-0.05 * REF_GEOMETRY["ring"]["total_wrap_extension"])
        self.assertAlmostEqual(float(tension[-1]), tip, delta=0.01)
        self.assertTrue(np.all(np.diff(tension) < 0.0))

    def test_solver_route_is_the_planar_route(self):
        """Compare tangent points and cable length of the solver's tendon route with the planar route."""
        # orientation signs: the solver's tangent points land on the tangent-line and arc route
        for name in ONE_CABLE_DIGITS:
            digit = self.spec.digits[name]
            for q in ((0.0, 0.0, 0.0), tuple(joint.upper for joint in digit.joints)):
                travel = ch.cable_takeup(digit, q)
                model, _skeleton, tendon, solver = self._build(name, q=q, travel=travel)
                route = ch.digit_route(digit, q, travel)
                left, right = ch.tendon_tangent_points(solver, model, model.state(), tendon)
                np.testing.assert_allclose(left[:, :2], route.p_out[:-1], rtol=0.0, atol=1.0e-7)
                np.testing.assert_allclose(right[:, :2], route.p_in[1:], rtol=0.0, atol=1.0e-7)
                np.testing.assert_allclose(left[:, 2], 0.0, atol=1.0e-9)
                length = ch._tendon_cable_length(solver, model, model.state(), tendon)
                self.assertAlmostEqual(length, route.length, delta=2.0e-7)
                # the cable is built with the rest tension and cut shorter by the opening of the three hinges
                # under it: unstretched length = route - tension * (compliance + 3 * hinge compliance)
                stretch = self.spec.cable.tension_rest * (
                    float(tendon.compliance.sum()) + 3.0 * self.spec.cable.hinge_compliance
                )
                self.assertAlmostEqual(
                    ch._tendon_unstretched_length(solver, tendon), route.length - stretch, delta=2.0e-7
                )
                starts, ends = ch._tendon_cable_lines(solver, model, model.state())
                self.assertEqual(len(starts), len(ends))
                self.assertGreater(len(starts), tendon.seg_count)

    def test_rest_tension_hold(self):
        """Simulate the tendon finger at the rest tension and check that it stays straight at that tension."""
        # 0.25 s at the rest tension in plain force mode: after the start transient (the proximal joint
        # lifts off its stop by under 1 deg for about 0.15 s) the finger is straight on its lower stops
        model, skeleton, tendon, solver = self._build("ring")
        ke = model.joint_target_ke.numpy()
        ke[skeleton.dof_slider] = 0.0
        model.joint_target_ke.assign(ke)
        state_0, state_1, control = model.state(), model.state(), model.control()
        joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
        ch.write_spring_preload(joint_f, skeleton)
        joint_f[skeleton.dof_slider] = self.spec.cable.tension_rest
        control.joint_f.assign(joint_f)
        dt = 1.0 / (ch.TENDON_FPS * ch.TENDON_SUBSTEPS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(15 * ch.TENDON_SUBSTEPS):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0
        self.assertEqual([str(w.message) for w in caught], [])
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
        joint_q = state_0.joint_q.numpy()
        # the stops give by (preload - tension * moment arm) / limit stiffness, at most 0.013 deg
        np.testing.assert_allclose(joint_q[list(skeleton.q_flexion)], 0.0, atol=math.radians(0.02))
        self.assertAlmostEqual(float(joint_q[skeleton.q_slider]), 0.0, delta=2.0e-5)
        np.testing.assert_allclose(ch.tendon_span_tensions(solver, tendon), self.spec.cable.tension_rest, rtol=0.01)
        first, last = ch.tendon_end_tensions(solver, tendon)
        self.assertAlmostEqual(first, last, delta=0.02)
        length = ch._tendon_cable_length(solver, model, state_0, tendon)
        stretch = length - ch._tendon_unstretched_length(solver, tendon)
        self.assertAlmostEqual(stretch, ch._tendon_stretch(solver, tendon), delta=1.0e-6)


class TestCableHandDrawing(unittest.TestCase):
    """Drawing helpers: tension colors, cable pieces, visual meshes of a single digit (CPU, no stepping)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_tension_colors(self):
        """Map tensions to colors: the hue is the cable's role, the brightness rises with the tension."""
        colors = ch.tension_colors([0.0, 2.0, 31.0, 60.0, 80.0], 60.0)
        self.assertEqual(colors.shape, (5, 3))
        self.assertEqual(colors.dtype, np.float32)
        np.testing.assert_allclose(colors[0], colors[1])  # clamped below the rest tension
        np.testing.assert_allclose(colors[3], colors[4])  # clamped above
        # a cable at rest is dim (0.45 of the base color), halfway up the scale it has 0.725, and at the top
        # the base color of its role
        base = np.array(ch.CABLE_COLORS["curl"])
        np.testing.assert_allclose(base, np.array([0x1A, 0xBF, 0xFF]) / 255.0)
        np.testing.assert_allclose(np.array(ch.CABLE_COLORS["knuckle"]), np.array([0xFF, 0x29, 0x8F]) / 255.0)
        np.testing.assert_allclose(colors[1], 0.45 * base, atol=1.0e-6)
        np.testing.assert_allclose(colors[2], 0.725 * base, atol=1.0e-6)
        np.testing.assert_allclose(colors[3], base, atol=1.0e-6)
        self.assertEqual(ch.tension_colors(np.zeros((2, 4)), 60.0).shape, (2, 4, 3))
        tension = np.linspace(2.0, 60.0, 7)
        curl = ch.tension_colors(tension, 60.0, "curl")
        knuckle = ch.tension_colors(tension, 60.0, "knuckle")
        # brightness rises with the tension on both cables, and the hue does not change with it
        for colors in (curl, knuckle):
            self.assertTrue(np.all(np.diff(colors.sum(axis=1)) > 0.0))
            scale = colors[:, 0] / colors[-1, 0]
            np.testing.assert_allclose(colors, np.outer(scale, colors[-1]), rtol=0.0, atol=1.0e-5)
        # the two cables of a finger differ in hue at every tension: the curl cable is blue, the knuckle
        # cable red; the single cable of a one-cable digit takes the curl hue
        self.assertTrue(np.all(curl[:, 2] > curl[:, 0]) and np.all(knuckle[:, 0] > knuckle[:, 2]))
        self.assertGreater(float(np.abs(curl - knuckle).max(axis=1).min()), 0.2)
        np.testing.assert_array_equal(ch.tension_colors(tension, 60.0, "flexor"), curl)

    def test_rod_segment_tensions(self):
        """Stretch one joint of a rod and check the tension per segment that the probe's read-back gives."""
        model, skeleton, cable = ch.build_single_rod_finger_model(self.spec, "ring", device=CPU)
        probe = ch.RodTensionProbe(model, skeleton, cable)
        body_q = model.body_q.numpy().astype(np.float64)
        # pull the segments after segment 39 away from it by 2.5 um along its axis: joint 39 alone is stretched
        k, stretch = 39, 2.5e-6
        axis = ch._quat_rotate_np(body_q[cable.bodies[k], 3:], np.array([0.0, 0.0, 1.0]))
        body_q[cable.bodies[k + 1 :], :3] += stretch * axis
        joint = probe.tension(body_q)
        self.assertEqual(joint.shape, (len(cable.bodies) - 1,))
        peak = self.spec.cable.EA / cable.segment_length * stretch  # about 5 N on a 1 mm segment
        self.assertAlmostEqual(float(joint[k]), peak, delta=0.01 * peak)
        self.assertLess(float(np.abs(np.delete(joint, k)).max()), 0.05)
        segment = ch.rod_segment_tensions(probe, body_q)
        self.assertEqual(segment.shape, (len(cable.bodies),))
        # a segment carries the mean of its two joints; the end segments have one joint each
        np.testing.assert_allclose(segment[1:-1], 0.5 * (joint[:-1] + joint[1:]), rtol=0.0, atol=1.0e-12)
        self.assertEqual((segment[0], segment[-1]), (joint[0], joint[-1]))
        np.testing.assert_allclose(segment[[k, k + 1]], 0.5 * peak, rtol=0.02)

    def test_cable_piece_transforms(self):
        """Check that the transform of each cable piece maps the unit cylinder onto its two end points."""
        rng = np.random.default_rng(3)
        starts = rng.normal(size=(12, 3))
        direction = rng.normal(size=(12, 3))
        direction[0] = (0.0, 0.0, 1.0)
        direction[1] = (0.0, 0.0, -1.0)
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        length = rng.uniform(0.001, 0.05, size=12)
        ends = starts + direction * length[:, None]
        xforms, half = ch.cable_piece_transforms(starts, ends)
        self.assertEqual(xforms.shape, (12, 7))
        np.testing.assert_allclose(half, 0.5 * length, rtol=1.0e-9)
        np.testing.assert_allclose(xforms[:, :3], 0.5 * (starts + ends), atol=1.0e-6)
        np.testing.assert_allclose(np.linalg.norm(xforms[:, 3:], axis=1), 1.0, atol=1.0e-6)
        z = np.tile([0.0, 0.0, 1.0], (12, 1))
        np.testing.assert_allclose(ch._quat_rotate_np(xforms[:, 3:].astype(float), z), direction, atol=1.0e-6)

    def test_tendon_cable_pieces(self):
        """Check that the drawn pieces of a tendon form one connected line that carries the span tensions."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, _skeleton, tendon = ch.build_tendon_finger_model(self.spec, "ring", device=CPU)
            solver = ch.make_tendon_solver(model)
        self.assertEqual([str(w.message) for w in caught], [])
        state = model.state()
        starts, ends, tension = ch.tendon_cable_pieces(solver, model, state, tendon, arc_step=0.05)
        self.assertEqual(len(starts), len(tension))
        np.testing.assert_allclose(starts[1:], ends[:-1], atol=1.0e-9)  # one continuous path
        route = ch.digit_route(self.spec.digits["ring"])
        np.testing.assert_allclose(starts[0][:2], route.p_out[0], atol=1.0e-6)
        np.testing.assert_allclose(ends[-1][:2], route.p_in[-1], atol=1.0e-6)
        # chords of at most 0.05 rad shorten an arc by less than 0.02 %
        length = float(np.linalg.norm(ends - starts, axis=1).sum())
        self.assertAlmostEqual(length, route.length, delta=2.0e-4 * route.length)
        self.assertLess(float(np.abs(starts[:, 2]).max()), 1.0e-6)  # the path lies in the flexion plane
        # every station with wrap contributes at least one arc piece
        self.assertGreater(len(starts), tendon.seg_count)

    def test_digit_visuals(self):
        """Attach the visual meshes to single digits and check their count, mass and the x-ray opacity."""
        counts = {}
        for name in ("ring", "thumb"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                plain, _, _ = ch.build_tendon_finger_model(self.spec, name, device=CPU)
                model, skeleton, _ = ch.build_tendon_finger_model(
                    self.spec, name, device=CPU, visual_materials=ch.visual_materials(xray=True)
                )
            self.assertEqual([str(w.message) for w in caught], [])
            counts[name] = model.shape_count - plain.shape_count
            shape_body = model.shape_body.numpy()
            for key, body in skeleton.bodies.items():
                self.assertGreater(int((shape_body == body).sum()), 0, f"{name} {key}")
            # visual shapes carry no mass and the x-ray look makes some of them translucent
            np.testing.assert_allclose(model.body_mass.numpy(), plain.body_mass.numpy())
            opacity = model.shape_opacity.numpy()
            self.assertTrue((opacity < 0.5).any())
            self.assertTrue((opacity > 0.99).any())
        # links 6 + 7 + 4 parts, carriage, and on the base: actuator, guide, first sheave and the bracket
        self.assertEqual(counts["ring"], 22)
        looks = ch.visual_materials()
        self.assertTrue(all("opacity" not in look for look in looks.values()))
        self.assertEqual(set(ch.XRAY_SHELL_MATERIALS) - set(looks), set())

    def test_xray_look_and_hidden_stations_of_a_finalized_model(self):
        """Make the shells of a finalized hand translucent and hide the station shapes of a rod finger."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, _hand, _tendons = ch.build_tendon_hand_model(self.spec, device=CPU, visuals=True)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue((model.shape_opacity.numpy() > 0.99).all())
        changed = ch.set_xray_look(model, opacity=0.25)
        opacity = model.shape_opacity.numpy()
        self.assertGreater(changed, 0)
        self.assertEqual(int((opacity < 0.5).sum()), changed)
        np.testing.assert_allclose(opacity[opacity < 0.5], 0.25)
        self.assertTrue((opacity > 0.99).any())  # pulleys, pins and carriages stay opaque
        self.assertEqual(len(ch.load_hand_visual_parts(self.spec)), 133)

        rod_model, _skeleton, cable = ch.build_single_rod_finger_model(self.spec, "ring", device=CPU)
        visible = int(newton.ShapeFlags.VISIBLE)
        stations = [shape for group in cable.station_shapes.values() for shape in group]
        self.assertTrue(all(rod_model.shape_flags.numpy()[stations] & visible))
        self.assertEqual(ch.hide_rod_stations(rod_model, cable), len(stations))
        flags = rod_model.shape_flags.numpy()
        self.assertFalse(any(flags[stations] & visible))
        self.assertTrue(all(flags[cable.rod_shapes] & visible))

    def test_rod_finger_visuals_hide_the_station_shapes(self):
        """Build a rod finger with visual meshes and check that its station shapes are not drawn."""
        model, _skeleton, cable = ch.build_single_rod_finger_model(
            self.spec, "ring", device=CPU, visual_materials=ch.visual_materials()
        )
        flags = model.shape_flags.numpy()
        visible = int(newton.ShapeFlags.VISIBLE)
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for shapes in cable.station_shapes.values():
            for shape in shapes:
                self.assertEqual(flags[shape] & visible, 0)
                self.assertEqual(flags[shape] & collide, collide)
        for shape in cable.rod_shapes:
            self.assertEqual(flags[shape] & visible, visible)


class TestCableHandVisualAsset(unittest.TestCase):
    """The shipped visual asset, the generator that writes it, and the meshes of a changed spec (no model)."""

    # largest size accepted for a visual asset of this example [bytes]
    ASSET_SIZE_LIMIT = 500_000

    def _written_size(self, parts):
        """Write the parts to a temporary USD file and return its size [bytes]."""
        try:
            from pxr import Usd  # noqa: F401
        except ImportError:
            self.skipTest("writing the visual asset needs pxr")
        with tempfile.TemporaryDirectory() as folder:
            path = chv.write_visuals_usd(parts, os.path.join(folder, "visuals.usd"))
            return os.path.getsize(path)

    def test_generator_reproduces_the_shipped_asset(self):
        """Generate the meshes of the default spec and compare them with the parts loaded from the asset."""
        spec = ch.default_hand_spec().to_dict()
        parts = chv.build_hand_visuals(spec)
        self.assertEqual(len(parts), 133)
        # every part sits on a body of the hand; a two-cable finger has one carriage per cable
        keys = chv.body_keys(spec)
        self.assertEqual({part.body for part in parts}, set(keys))
        self.assertEqual(len(keys), 24)
        self.assertIn("index/slider_knuckle", keys)
        self.assertIn("middle/slider_knuckle", keys)
        self.assertLess(self._written_size(parts), self.ASSET_SIZE_LIMIT)
        self.assertLess(os.path.getsize(newton.examples.get_asset(chv.ASSET_NAME)), self.ASSET_SIZE_LIMIT)
        # the shipped asset holds the same parts: names, bodies, materials, poses, vertices (to 1 um) and
        # triangles. The comparison is by content, not by bytes: the file layout belongs to the USD library.
        shipped = {(part.body, part.name): part for part in chv.load_visuals_usd()}
        self.assertEqual(len(shipped), 133)
        self.assertEqual(set(shipped), {(part.body, part.name) for part in parts})
        for part in parts:
            other, label = shipped[(part.body, part.name)], f"{part.body}/{part.name}"
            self.assertEqual(other.material, part.material, label)
            np.testing.assert_allclose(other.pos, part.pos, rtol=0.0, atol=1.0e-6, err_msg=label)
            np.testing.assert_allclose(np.abs(np.dot(other.quat, part.quat)), 1.0, rtol=0.0, atol=1.0e-6, err_msg=label)
            np.testing.assert_allclose(other.vertices, part.vertices, rtol=0.0, atol=1.0e-6, err_msg=label)
            np.testing.assert_array_equal(other.indices, part.indices, err_msg=label)
        self.assertEqual(len(ch.load_hand_visual_parts()), 133)

    def test_changed_spec_changes_the_meshes(self):
        """Load the parts of a spec with a wider ring-finger link and find the wider frame, with or without pxr."""

        def frame_width(parts):
            (frame,) = [part for part in parts if part.body == "ring/link1" and part.name == "frame"]
            return float(np.ptp(np.asarray(frame.vertices)[:, 2]))

        spec = ch.default_hand_spec()
        default_width = frame_width(ch.load_hand_visual_parts())
        self.assertAlmostEqual(default_width, spec.digits["ring"].links[0].width, delta=1.0e-6)
        # a spec equal to the default takes the same parts as no spec
        self.assertAlmostEqual(frame_width(ch.load_hand_visual_parts(spec)), default_width, delta=1.0e-9)
        spec.digits["ring"].links[0].width += 0.004
        self.assertAlmostEqual(frame_width(ch.load_hand_visual_parts(spec)), default_width + 0.004, delta=1.0e-6)
        # the same through a builder that attaches the visuals
        model, _skeleton, _tendon = ch.build_tendon_finger_model(
            spec, "ring", device=CPU, visual_materials=ch.visual_materials()
        )
        shape = model.shape_label.index("visual/link1/frame")
        mesh = model.shape_source[shape]
        self.assertAlmostEqual(float(np.ptp(np.asarray(mesh.vertices)[:, 2])), default_width + 0.004, delta=1.0e-6)


class TestCableHandHandSkeleton(unittest.TestCase):
    """Whole-hand skeleton: counts, body keys, frames, joint order, visuals, wrist and opposition drivers."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def _build(self, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, hand = ch._build_hand_skeleton_model(self.spec, device=CPU, **kwargs)
        self.assertEqual([str(w.message) for w in caught], [])
        return model, hand

    def test_counts_and_body_keys(self):
        """Check body, joint and articulation counts and the body keys of the hand skeleton."""
        from newton.examples.cable import cable_hand_visuals as chv  # noqa: PLC0415

        # 24 rigid bodies: palm, thumb column, 15 links and 7 carriages (two each on index and middle)
        for wrist, dof, coord in (("fixed", 23, 23), ("kinematic", 29, 30)):
            model, hand = self._build(wrist=wrist)
            self.assertEqual(model.body_count, 24)
            self.assertEqual(model.joint_count, 24)
            self.assertEqual(model.joint_dof_count, dof)
            self.assertEqual(model.joint_coord_count, coord)
            self.assertEqual(model.articulation_count, 1)
            joint_type = model.joint_type.numpy()
            self.assertEqual(int((joint_type == newton.JointType.REVOLUTE).sum()), 16)
            self.assertEqual(int((joint_type == newton.JointType.PRISMATIC).sum()), 7)
            # one linear actuator per cable; the key of a digit's first cable is the digit name
            self.assertEqual(
                hand.actuators, ["index", "index_knuckle", "middle", "middle_knuckle", "ring", "little", "thumb"]
            )
            self.assertEqual(hand.actuator_cables[1], ("index", "knuckle"))
            self.assertEqual(hand.actuator_index("middle", "knuckle"), 3)
            self.assertEqual(hand.actuator_index("thumb"), 6)
            self.assertIn("index/slider_knuckle", hand.bodies)
            root = newton.JointType.FREE if wrist == "kinematic" else newton.JointType.FIXED
            self.assertEqual(joint_type[hand.root_joint], root)
            kinematic = bool(model.body_flags.numpy()[hand.palm] & int(newton.BodyFlags.KINEMATIC))
            self.assertEqual(kinematic, wrist == "kinematic")
            # body keys are those the visual parts refer to, one body each
            self.assertEqual(sorted(hand.bodies), sorted(chv.body_keys(self.spec.to_dict())))
            self.assertEqual(sorted(hand.bodies.values()), list(range(24)))
            self.assertEqual(hand.names, list(DIGIT_NAMES))
            # masses: palm 301 g, thumb column 20 g, links 84 g, seven carriages 42 g
            mass = model.body_mass.numpy()
            self.assertAlmostEqual(float(mass.sum()), 0.447, delta=1.0e-6)
            self.assertAlmostEqual(float(mass[hand.palm]), 0.301, delta=1.0e-7)
            self.assertAlmostEqual(float(mass[hand.thumb_column]), 0.020, delta=1.0e-7)
            inertia = model.body_inertia.numpy()
            for body in (hand.palm, hand.thumb_column):
                self.assertTrue(np.all(np.linalg.eigvalsh(inertia[body]) > 0.0))
        with self.assertRaises(ValueError):
            ch._build_hand_skeleton_model(self.spec, wrist="floating", device=CPU)

    def test_joint_order_is_one_contiguous_block_per_digit(self):
        """Check that the joints and coordinates of every digit form one contiguous block of the model."""
        model, hand = self._build()
        self.assertEqual(hand.joints, list(range(model.joint_count)))
        self.assertEqual(hand.root_joint, 0)
        expected = ["wrist"]
        for name in DIGIT_NAMES:
            joints = hand.digit_joints[name]
            self.assertEqual(joints, list(range(joints[0], joints[0] + len(joints))))
            if name == "thumb":
                self.assertEqual(joints[0], hand.joint_opposition)
                expected.append("thumb_opposition")
            skeleton = hand.digits[name]
            self.assertEqual(joints[-len(skeleton.joints) :], skeleton.joints)
            expected += [f"{name}_mcp", f"{name}_pip", f"{name}_dip", f"{name}_slider_axis"]
            if name in TWO_CABLE_DIGITS:
                expected.append(f"{name}_slider_knuckle_axis")
        self.assertEqual(list(model.joint_label), expected)
        self.assertEqual(list(model.articulation_start.numpy()[:1]), [0])
        # the thumb and its slider hang on the thumb column, the fingers and their sliders on the palm
        parent = model.joint_parent.numpy()
        child = model.joint_child.numpy()
        self.assertEqual((parent[hand.joint_opposition], child[hand.joint_opposition]), (hand.palm, hand.thumb_column))
        for name in DIGIT_NAMES:
            skeleton = hand.digits[name]
            base = hand.thumb_column if name == "thumb" else hand.palm
            self.assertEqual(skeleton.bodies["base"], base)
            self.assertEqual(parent[skeleton.joint_flexion[0]], base)
            for joint in skeleton.joint_sliders:
                self.assertEqual(parent[joint], base)
        # a cable model may add joints after each digit; the articulation then takes all of them
        builder = newton.ModelBuilder()
        added = []

        def after_digit(builder, hand, skeleton):
            body = builder.add_link(mass=0.001, inertia=wp.mat33(np.eye(3) * 1.0e-9), label=f"{skeleton.name}_extra")
            added.append(builder.add_joint_ball(parent=skeleton.bodies["link3"], child=body))

        hand = ch.add_hand_skeleton(builder, self.spec, after_digit=after_digit)
        joints = ch.add_hand_articulation(builder, hand)
        self.assertEqual(joints, list(range(29)))
        self.assertEqual(added, [6, 12, 17, 22, 28])
        with self.assertRaises(ValueError):
            ch.add_hand_digit(builder, self.spec, hand, "index")

    def test_digit_frames_and_opposition_hinge(self):
        """Check the digit frames of the built hand and the axis, limits and drive of the opposition hinge."""
        palm_xform = wp.transform(
            wp.vec3(0.1, -0.2, 0.3), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
        )
        for opposition in (0.0, math.radians(118.0)):
            model, hand = self._build(palm_xform=palm_xform, opposition=opposition)
            body_q = model.body_q.numpy()
            for name in DIGIT_NAMES:
                origin, rot = ch.digit_frame_in_palm(self.spec, name, opposition)
                skeleton = hand.digits[name]
                length = self.spec.digits[name].links[0].length
                link1 = np.array(wp.transform_point(palm_xform, wp.vec3(*origin)))
                link2 = np.array(wp.transform_point(palm_xform, wp.vec3(*(origin + length * rot[:, 0]))))
                np.testing.assert_allclose(body_q[skeleton.bodies["link1"]][:3], link1, atol=2.0e-7)
                np.testing.assert_allclose(body_q[skeleton.bodies["link2"]][:3], link2, atol=2.0e-7)
                # flexion axis: link z in the world
                axis = wp.quat_rotate(wp.quat(*body_q[skeleton.bodies["link1"]][3:]), wp.vec3(0.0, 0.0, 1.0))
                expected = wp.transform_vector(palm_xform, wp.vec3(*rot[:, 2]))
                np.testing.assert_allclose(np.array(axis), np.array(expected), atol=1.0e-6)
            # the thumb column frame is the thumb digit frame
            origin, _rot = ch.digit_frame_in_palm(self.spec, "thumb", opposition)
            expected = np.array(wp.transform_point(palm_xform, wp.vec3(*origin)))
            np.testing.assert_allclose(body_q[hand.thumb_column][:3], expected, atol=2.0e-7)
            # forward kinematics of the joint coordinates reproduces the authored poses
            self.assertAlmostEqual(float(model.joint_q.numpy()[hand.q_opposition]), opposition, delta=1.0e-7)
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            np.testing.assert_allclose(state.body_q.numpy(), body_q, atol=1.0e-6)
        mount = self.spec.digits["thumb"].mount
        dof = hand.dof_opposition
        self.assertAlmostEqual(float(model.joint_limit_lower.numpy()[dof]), 0.0, delta=1.0e-9)
        self.assertAlmostEqual(float(model.joint_limit_upper.numpy()[dof]), math.radians(130.0), delta=1.0e-6)
        self.assertAlmostEqual(float(model.joint_target_ke.numpy()[dof]), 20.0, delta=1.0e-6)
        self.assertAlmostEqual(float(model.joint_target_kd.numpy()[dof]), 0.2, delta=1.0e-6)
        np.testing.assert_allclose(model.joint_axis.numpy()[dof], mount.hinge_axis, atol=1.0e-6)
        self.assertEqual((hand.opposition_torque_limit, hand.opposition_speed_limit), (0.8, 3.0))

    def test_actuator_gains_override(self):
        """Build the hand with other slider gains and check that they reach the model."""
        builder = newton.ModelBuilder()
        hand = ch.add_hand_skeleton(builder, self.spec, actuator_gains=(2.0e4, 200.0))
        ch.add_hand_articulation(builder, hand)
        model = builder.finalize(device=CPU)
        dofs = [dof for skeleton in hand.skeletons for dof in skeleton.dof_sliders]
        self.assertEqual(len(dofs), 7)
        np.testing.assert_allclose(model.joint_target_kd.numpy()[dofs], 200.0)
        np.testing.assert_allclose(model.joint_target_ke.numpy()[dofs], 2.0e4)
        self.assertEqual({skeleton.slider_kd for skeleton in hand.skeletons}, {200.0})

    def test_visuals_attach_to_every_body(self):
        """Attach the shipped visual meshes and check that every body has parts and no mass is added."""
        try:
            from pxr import Usd  # noqa: F401
        except ImportError:
            self.skipTest("the visual asset needs pxr")
        model, hand = self._build(visuals=True)
        self.assertEqual(model.shape_count, 133)
        shape_body = model.shape_body.numpy()
        for key, body in hand.bodies.items():
            self.assertGreater(int((shape_body == body).sum()), 0, key)
        # visual shapes carry no mass
        self.assertAlmostEqual(float(model.body_mass.numpy().sum()), 0.447, delta=1.0e-6)

    def test_wrist_driver_and_opposition_reference(self):
        """Step the hand control and check the wrist motion, the weights and the opposition reference."""
        model, hand = self._build(wrist="kinematic")
        control = ch.create_hand_control(model, self.spec, hand, gravity=(0.0, -ch.GRAVITY, 0.0))
        self.assertIsNotNone(control.wrist)
        self.assertEqual(control.governor.count, 7)
        self.assertEqual([key for key, _ in control.governor.actuators], [name for name, _ in hand.actuator_cables])
        state, ctrl = model.state(), model.control()
        control.apply_preload(ctrl)
        joint_f = ctrl.joint_f.numpy()
        for name, skeleton in hand.digits.items():
            preload = [-0.12, -0.026, -0.04] if name in TWO_CABLE_DIGITS else [-0.048, -0.065, -0.080]
            np.testing.assert_allclose(joint_f[list(skeleton.dof_flexion)], preload, rtol=1.0e-6)
        dt = 1.0e-3
        turn = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.4)
        control.set_wrist_target((0.02, 0.0, -0.04, *turn), steps=4)
        control.set_tension([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        com = model.body_com.numpy()[hand.palm]
        for k in range(1, 6):
            control.step(state, ctrl, dt)
            pose = state.body_q.numpy()[hand.palm]
            share = min(k, 4) / 4.0
            np.testing.assert_allclose(pose[:3], share * np.array([0.02, 0.0, -0.04]), atol=1.0e-6)
            angle = 2.0 * math.acos(min(abs(float(pose[6])), 1.0))
            self.assertAlmostEqual(angle, share * 0.4, delta=1.0e-4)
            velocity = state.body_qd.numpy()[hand.palm]
            if k <= 4:
                self.assertAlmostEqual(float(np.linalg.norm(velocity[3:])), 0.1 / dt, delta=0.05)
            else:
                np.testing.assert_allclose(velocity, 0.0, atol=1.0e-4)
        self.assertGreater(float(np.linalg.norm(com)), 0.0)
        # body forces were cleared and the weights written
        body_f = state.body_f.numpy()
        np.testing.assert_allclose(body_f[:, 1], -ch.GRAVITY * model.body_mass.numpy(), rtol=1.0e-5)
        # governor commands per actuator, force mode
        np.testing.assert_allclose(control.governor.command.numpy(), [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
        # commands per digit: a tension for all cables of a digit, a tension per cable, a two-cable pose
        # name, a first-joint angle at a curl tension, or one actuator key; other cables keep their command
        control.set_tension({"ring": 12.0, "index": "hook", "middle": {"curl": 9.0, "knuckle": 20.0}})
        np.testing.assert_allclose(
            control.governor.command.numpy(), [18.0, 2.0, 9.0, 20.0, 12.0, 8.0, 10.0], rtol=1.0e-6
        )
        control.set_tension({"index": 14.0, "index_knuckle": 3.0, "middle": {"curl": 6.0, "knuckle_angle": 0.5}})
        np.testing.assert_allclose(
            control.governor.command.numpy(),
            [14.0, 3.0, 6.0, (0.12 + 0.04 - 0.03) / 0.008, 12.0, 8.0, 10.0],
            rtol=1.0e-6,
        )
        tensions = ch.hand_cable_tensions(self.spec, {"thumb": 7.0, "middle": "full_fist"})
        self.assertEqual(tensions, {"thumb": 7.0, "middle": 40.0, "middle_knuckle": 40.0})

        # opposition reference on a fixed wrist (the bodies do not move without a solver step): rate limit
        # 3 rad/s, and no further than torque limit / stiffness from the measured angle
        model, hand = self._build(opposition=0.5)
        control = ch.create_hand_control(model, self.spec, hand)
        self.assertIsNone(control.wrist)
        with self.assertRaises(ValueError):
            ch.create_wrist_driver(model, hand)
        with self.assertRaises(ValueError):
            control.set_wrist_target((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        state, ctrl = model.state(), model.control()
        control.set_opposition(1.0)
        for _ in range(5):
            control.step(state, ctrl, dt)
        self.assertAlmostEqual(float(control.opposition_reference.numpy()[0]), 0.5 + 5 * 3.0 * dt, delta=1.0e-6)
        self.assertAlmostEqual(float(control.opposition_angle.numpy()[0]), 0.5, delta=1.0e-5)
        target_index = int(model.joint_target_q_start.numpy()[hand.joint_opposition])
        self.assertAlmostEqual(float(ctrl.joint_target_q.numpy()[target_index]), 0.5 + 5 * 3.0 * dt, delta=1.0e-6)
        for _ in range(40):
            control.step(state, ctrl, dt)
        self.assertAlmostEqual(float(control.opposition_reference.numpy()[0]), 0.5 + 0.8 / 20.0, delta=1.0e-5)
        control.set_opposition(-1.0)
        for _ in range(60):
            control.step(state, ctrl, dt)
        self.assertAlmostEqual(float(control.opposition_reference.numpy()[0]), 0.5 - 0.8 / 20.0, delta=1.0e-5)


class TestCableHandTendonHand(unittest.TestCase):
    """Hand with seven massless tendons: routes on the palm, rest hold and one two-cable digit closing (CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def _build(self, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, hand, tendons = ch.build_tendon_hand_model(self.spec, device=CPU, **kwargs)
            solver = ch.make_tendon_hand_solver(model)
        self.assertEqual([str(w.message) for w in caught], [])
        return model, hand, tendons, solver

    def test_model_and_routes(self):
        """Check counts, cable routes and joint stops of the tendon hand with a fixed and a kinematic wrist."""
        for kwargs in ({}, {"wrist": "kinematic", "opposition": math.radians(60.0)}):
            model, hand, tendons, solver = self._build(**kwargs)
            # the counts of the design: 24 rigid bodies, 7 cables, 47 tendon links
            self.assertEqual(model.body_count, 24)
            self.assertEqual(model.tendon_count, 7)
            self.assertEqual(model.tendon_link_count, 47)
            self.assertEqual(model.tendon_segment_count, 40)
            self.assertEqual(model.articulation_count, 1)
            np.testing.assert_allclose(model.gravity.numpy(), 0.0)
            self.assertEqual(list(tendons), hand.actuators)
            state = model.state()
            routes = ch._hand_cable_routes(self.spec, hand, model.body_q.numpy())
            self.assertEqual(list(routes), hand.actuators)
            for key, (name, cable) in zip(hand.actuators, hand.actuator_cables, strict=True):
                skeleton, tendon = hand.digits[name], tendons[key]
                self.assertEqual(tendon.cable, cable)
                planar = ch.digit_route(self.spec.digits[name], cable=cable)
                # the stations carried by the palm are placed through the digit frame
                np.testing.assert_allclose(routes[key].p_out, planar.p_out, atol=1.0e-6)
                np.testing.assert_allclose(routes[key].wraps, planar.wraps, atol=1.0e-4)
                length = ch._tendon_cable_length(solver, model, state, tendon)
                self.assertAlmostEqual(length, planar.length, delta=5.0e-7)
                dofs = list(skeleton.dof_flexion)
                np.testing.assert_allclose(model.joint_limit_ke.numpy()[dofs], 300.0)
            self.assertAlmostEqual(float(model.joint_limit_ke.numpy()[hand.dof_opposition]), 300.0)
            starts, ends = ch._tendon_cable_lines(solver, model, state)
            self.assertEqual(len(starts), len(ends))
            self.assertGreater(len(starts), 40)

    def test_flexed_start_pose(self):
        """Build the tendon hand with flexed digits and check joint angles, travel and cable lengths."""
        q = {"index": (0.6, 0.4, 0.2), "thumb": (0.3, 0.0, 0.0)}
        model, hand, tendons, solver = self._build(q=q, opposition=math.radians(100.0))
        state = model.state()
        takeup = ch.hand_takeup(self.spec, q)
        for name, angles in q.items():
            digit = self.spec.digits[name]
            for c, cable in enumerate(digit.cables):
                planar = ch.digit_route(digit, angles, takeup[name][c], cable=cable.name)
                tendon = tendons[ch.actuator_key(name, c, cable.name)]
                length = ch._tendon_cable_length(solver, model, state, tendon)
                self.assertAlmostEqual(length, planar.length, delta=5.0e-7)
        joint_state = ch.hand_joint_state(model, state, hand)
        np.testing.assert_allclose(joint_state["q"][0], q["index"], atol=1.0e-5)
        np.testing.assert_allclose(joint_state["q"][1], 0.0, atol=1.0e-6)
        self.assertAlmostEqual(joint_state["opposition"], math.radians(100.0), delta=1.0e-5)
        # one travel per actuator: the curl carriage took up 5 / 6.5 / 5 mm per radian, the knuckle carriage 8 mm
        self.assertEqual(joint_state["travel"].shape, (7,))
        np.testing.assert_allclose(
            joint_state["travel"][:2], [0.005 * 0.6 + 0.0065 * 0.4 + 0.005 * 0.2, 0.008 * 0.6], atol=1.0e-6
        )
        self.assertAlmostEqual(float(joint_state["travel"][6]), 0.008 * 0.3, delta=1.0e-6)

    def test_rest_hold_then_index_closes(self):
        """Simulate the tendon hand for a second: all cables rest, then the two index cables pull, the others rest."""
        model, hand, tendons, solver = self._build()
        control = ch.create_hand_control(model, self.spec, hand)
        state_0, state_1, ctrl = model.state(), model.state(), model.control()
        control.apply_preload(ctrl)
        dt = 1.0 / (ch.TENDON_FPS * ch.TENDON_SUBSTEPS)
        rest = self.spec.cable.tension_rest

        def run(frames):
            nonlocal state_0, state_1
            for _ in range(frames * ch.TENDON_SUBSTEPS):
                control.step(state_0, ctrl, dt)
                solver.step(state_0, state_1, ctrl, None, dt)
                state_0, state_1 = state_1, state_0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # 0.25 s at the rest tension: every digit straight on its lower stops, every span at 2 N
            run(15)
            self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
            joint_state = ch.hand_joint_state(model, state_0, hand)
            np.testing.assert_allclose(joint_state["q"], 0.0, atol=math.radians(0.1))
            np.testing.assert_allclose(joint_state["travel"], 0.0, atol=5.0e-5)
            self.assertAlmostEqual(joint_state["opposition"], 0.0, delta=math.radians(0.1))
            for tension in ch.hand_span_tensions(solver, tendons).values():
                np.testing.assert_allclose(tension, rest, rtol=0.05)
            # 0.75 s with 9 N on the index curl cable and 20 N on its knuckle cable: the index closes to
            # 60.9 / 62.5 / 7.0 deg (closed form), the other digits stay straight at the rest tension
            command = {"index": {"curl": 9.0, "knuckle": 20.0}}
            control.set_tension(command)
            run(54)
        self.assertEqual([str(w.message) for w in caught], [])
        body_q = state_0.body_q.numpy()
        self.assertTrue(np.isfinite(body_q).all())
        joint_state = ch.hand_joint_state(model, state_0, hand)
        row = ch.frictionless_row(self.spec, "index", command["index"])
        # the curl reference moves at 24 mm/s, the knuckle reference at 28 mm/s, and the joints settle within
        # 0.2 s after that
        np.testing.assert_allclose(joint_state["q"][0], row["q"], atol=math.radians(1.0))
        np.testing.assert_allclose(joint_state["travel"][:2], row["travel"], atol=2.0e-4)
        np.testing.assert_allclose(joint_state["q"][1:], 0.0, atol=math.radians(0.5))
        np.testing.assert_allclose(joint_state["travel"][2:], 0.0, atol=5.0e-5)
        tensions = ch.hand_span_tensions(solver, tendons)
        self.assertEqual(list(tensions), hand.actuators)
        for key in hand.actuators[2:]:
            np.testing.assert_allclose(tensions[key], rest, rtol=0.1)
        np.testing.assert_allclose(tensions["index"], 9.0, rtol=0.05)
        np.testing.assert_allclose(tensions["index_knuckle"], 20.0, rtol=0.05)
        for route in ch._hand_cable_routes(self.spec, hand, body_q).values():
            wraps = route.wraps[1:-1]
            self.assertTrue(np.all(wraps > 0.0) and np.all(wraps < math.pi))


@wp.kernel
def _sum_kernel(value: wp.array[wp.int32], total: wp.array[wp.int32]):
    total[0] = total[0] + value[0]


_GRASP_TEST_PHASES = {
    "lead_in": 0.05,
    "approach": 0.15,
    "close": 0.7,
    "oppose": 0.8,
    "thumb": 0.5,
    "settle": 0.8,
    "lift": 0.3,
    "hold": 0.2,
}
"""Phase durations of the shortened grasp plan of ``TestCableHandGrasp`` [s]."""


class TestCableHandGrasp(unittest.TestCase):
    """Grasping: pads and their collision filters, the keyframe script, the plans, and the ball grip (CPU).

    The lifted grasps are simulated by the registered tests of the hand example (can and block).
    """

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()
        # the ball scene with pedestal and seat, for the tests of the model, the pads and the script
        cls.plan = ch.grasp_plan(cls.spec, "ball", lift=0.03, approach=0.01, phases=_GRASP_TEST_PHASES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.model, cls.hand, cls.tendons, cls.scene = ch.build_tendon_hand_grasp_model(
                cls.spec,
                "ball",
                object_xform=cls.plan.object_xform,
                palm_xform=cls.plan.palm_start,
                pedestal=cls.plan.pedestal,
                device=CPU,
            )
        cls.build_warnings = [str(w.message) for w in caught]

    def test_pads_keys_shapes_and_filters(self):
        """Check keys, shapes, contact material and collision pairs of the pads in a grasp scene."""
        spec, model, hand, scene = self.spec, self.model, self.hand, self.scene
        self.assertEqual(self.build_warnings, [])
        pads = scene.pads
        expected = [f"{name}/link{i}" for name in spec.digits for i in (1, 2, 3)]
        expected += [f"palm/{pad.name}" for pad in spec.palm.pads]
        self.assertEqual(pads.keys, expected)
        self.assertEqual(len(pads.keys), 17)
        self.assertEqual(model.body_count, 25)  # 24 bodies of the hand and the object
        shape_body = model.shape_body.numpy()
        scale = model.shape_scale.numpy()
        for i, key in enumerate(pads.keys):
            shape = pads.shapes[key]
            self.assertEqual(model.shape_label[shape], f"pad/{key}")
            self.assertEqual(int(shape_body[shape]), hand.palm if key.startswith("palm/") else hand.bodies[key])
            self.assertEqual(int(model.shape_type.numpy()[shape]), int(newton.GeoType.CAPSULE))
            self.assertAlmostEqual(float(scale[shape][0]), pads.radius[i], places=7)
            self.assertAlmostEqual(float(scale[shape][1]), 0.5 * np.linalg.norm(pads.p1[i] - pads.p0[i]), places=7)
        digit = spec.digits["index"]
        np.testing.assert_allclose(pads.radius[:3], [pad.radius for pad in digit.pads])
        np.testing.assert_allclose(pads.p0[0], [*digit.pads[0].p0, 0.0])
        np.testing.assert_allclose(pads.p1[-1], spec.palm.pads[-1].p1)
        # contact material of the spec; pair friction is the geometric mean
        pad_shapes = sorted(pads.shapes.values())
        mu = model.shape_material_mu.numpy()
        np.testing.assert_allclose(mu[pad_shapes], spec.pads.mu)
        np.testing.assert_allclose(model.shape_material_ke.numpy(), spec.pads.ke)
        np.testing.assert_allclose(model.shape_gap.numpy()[pad_shapes], spec.pads.gap)
        self.assertAlmostEqual(math.sqrt(mu[scene.object_shape] * mu[pad_shapes[0]]), spec.pads.mu, places=6)
        self.assertAlmostEqual(math.sqrt(mu[scene.object_shape] * mu[scene.table_shape]), spec.pads.table_mu, places=6)
        self.assertAlmostEqual(
            math.sqrt(mu[scene.object_shape] * mu[scene.pedestal_shape]), spec.pads.table_mu, places=6
        )
        # filters: a pad meets the object, the table and the pedestal with its seat, never another pad
        self.assertEqual(set(model.shape_collision_group.numpy().tolist()), {ch.GRASP_COLLISION_GROUP})
        self.assertNotEqual(ch.GRASP_COLLISION_GROUP, 0)
        self.assertLess(ch.GRASP_COLLISION_GROUP, ch.ROD_FIRST_COLLISION_GROUP)
        self.assertEqual(len(scene.seat_shapes), 3)
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in model.shape_contact_pairs.numpy()}
        others = (scene.object_shape, scene.table_shape, scene.pedestal_shape, *scene.seat_shapes)
        wanted = {tuple(sorted((pad, other))) for pad in pad_shapes for other in others}
        wanted |= {tuple(sorted((scene.object_shape, other))) for other in others[1:]}
        self.assertEqual(pairs, wanted)

    def test_pad_self_collision_option(self):
        """Add pads with self collision and check that only pads of different digits may collide."""
        spec = self.spec
        builder = newton.ModelBuilder()
        hand = ch.add_hand_skeleton(builder, spec)
        ch.add_hand_articulation(builder, hand)
        pads = ch.add_hand_pads(builder, spec, hand, self_collision=True)
        self.assertTrue(pads.self_collision)
        model = builder.finalize(device=CPU)
        owner = {shape: key.split("/")[0] for key, shape in pads.shapes.items()}
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in model.shape_contact_pairs.numpy()}
        self.assertGreater(len(pairs), 0)
        for a, b in pairs:
            self.assertNotEqual(owner[a], owner[b])
        self.assertIn(tuple(sorted((pads.shapes["thumb/link3"], pads.shapes["index/link3"]))), pairs)
        self.assertIn(tuple(sorted((pads.shapes["middle/link3"], pads.shapes["palm/heel"]))), pairs)

    def test_object_properties(self):
        """Check size, mass and inertia of the three objects and the signed distance to their surfaces."""
        spec, model, scene = self.spec, self.model, self.scene
        kind, size, mass, inertia = ch.grasp_object_properties(spec, "can")
        self.assertEqual((kind, size, mass), ("cylinder", (0.032, 0.055), 0.150))
        np.testing.assert_allclose(inertia[2], 0.5 * 0.150 * 0.032**2)
        np.testing.assert_allclose(inertia[0], 0.150 / 12.0 * (3.0 * 0.032**2 + 0.110**2))
        kind, size, mass, inertia = ch.grasp_object_properties(spec, "block")
        self.assertEqual((kind, size, mass), ("box", (0.015, 0.015, 0.030), 0.040))
        np.testing.assert_allclose(inertia, 0.040 / 12.0 * np.array([0.0045, 0.0045, 0.0018]))
        self.assertEqual((scene.object_kind, scene.object_size), ("sphere", (0.028,)))
        self.assertAlmostEqual(float(model.body_mass.numpy()[scene.object_body]), 0.080, places=7)
        np.testing.assert_allclose(
            np.diag(model.body_inertia.numpy()[scene.object_body]), 0.4 * 0.080 * 0.028**2, rtol=1e-5
        )
        # signed distances to the three kinds of object
        np.testing.assert_allclose(ch.object_surface_distance("sphere", (0.028,), [[0.0, 0.03, 0.0]]), [0.002])
        points = [[0.04, 0.0, 0.0], [0.0, 0.0, 0.06], [0.01, 0.0, 0.0], [0.035, 0.0, 0.059]]
        np.testing.assert_allclose(
            ch.object_surface_distance("cylinder", (0.032, 0.055), points), [0.008, 0.005, -0.022, 0.005], atol=1e-12
        )
        np.testing.assert_allclose(
            ch.object_surface_distance("box", (0.015, 0.015, 0.03), [[0.02, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            [0.005, -0.015],
            atol=1e-12,
        )

    def _keyframes(self):
        quarter = np.array(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * math.pi), dtype=float)
        return [
            ch.HandKeyframe(0.0, wrist=(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0), opposition=0.0, tension=2.0),
            ch.HandKeyframe(1.0, wrist=(0.2, 0.0, 0.1, *quarter)),
            ch.HandKeyframe(1.5, opposition=1.0, tension={"index": 30.0, "thumb": 10.0}),
            ch.HandKeyframe(2.0, travel={"middle": 0.012}),
            ch.HandKeyframe(3.0, wrist=(0.2, 0.0, 0.3, *(-quarter)), travel={"middle": 0.02}, tension=4.0),
        ]

    def test_script_interpolation(self):
        """Resolve keyframes and check the interpolated wrist, opposition and commands and the wrist peaks."""
        # one command column per actuator: index, index_knuckle, middle, middle_knuckle, ring, little, thumb.
        # Without a spec a keyframe entry addresses one actuator by its key.
        names = self.hand.actuators
        self.assertEqual(names, ["index", "index_knuckle", "middle", "middle_knuckle", "ring", "little", "thumb"])
        index, middle, thumb = names.index("index"), names.index("middle"), names.index("thumb")
        keys = ch.resolve_hand_keyframes(self._keyframes(), names, start_tension=2.0)
        self.assertEqual(keys["command"].shape, (5, 7))
        # open fields keep the previous value; a travel command switches its actuator to position mode
        np.testing.assert_allclose(keys["wrist"][2], keys["wrist"][1])
        np.testing.assert_allclose(keys["command"][2], [30.0, 2.0, 2.0, 2.0, 2.0, 2.0, 10.0])
        self.assertEqual(keys["mode"][3].tolist(), [0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(keys["command"][3], [30.0, 2.0, 0.012, 2.0, 2.0, 2.0, 10.0])
        # in one keyframe the travel entries are applied after the tension entry
        self.assertEqual(keys["mode"][4].tolist(), [0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_allclose(keys["command"][4], [4.0, 4.0, 0.02, 4.0, 4.0, 4.0, 4.0])
        # with the spec a digit command covers every cable of the digit, and a two-cable finger takes a pair
        # of tensions or a pose name
        pair = [ch.HandKeyframe(0.0, tension=2.0), ch.HandKeyframe(1.0, tension={"index": 30.0, "middle": "hook"})]
        resolved = ch.resolve_hand_keyframes(pair, names, start_tension=2.0, spec=self.spec)
        hook = ch.two_cable_poses(self.spec, "middle")["hook"]
        np.testing.assert_allclose(
            resolved["command"][1], [30.0, 30.0, hook["curl"], hook["knuckle"], 2.0, 2.0, 2.0], rtol=1e-6
        )
        # the last wrist quaternion was given with the opposite sign: same hemisphere after resolving
        self.assertGreater(float(keys["wrist"][4, 3:] @ keys["wrist"][3, 3:]), 0.99)
        # values at the keyframes, held before the first and after the last
        for k, t in enumerate(keys["times"]):
            value = ch.evaluate_hand_keyframes(keys, float(t))
            np.testing.assert_allclose(value["wrist"], keys["wrist"][k], atol=1e-12)
            self.assertAlmostEqual(value["opposition"], keys["opposition"][k], places=12)
            np.testing.assert_allclose(value["command"], keys["command"][k], atol=1e-12)
        np.testing.assert_allclose(ch.evaluate_hand_keyframes(keys, -1.0)["wrist"], keys["wrist"][0])
        np.testing.assert_allclose(ch.evaluate_hand_keyframes(keys, 9.0)["command"], keys["command"][4])
        # minimum-jerk blend: 0.5 at mid-time, 0.103515625 at a quarter
        self.assertAlmostEqual(float(ch.minimum_jerk(0.25)), 0.103515625, places=12)
        mid = ch.evaluate_hand_keyframes(keys, 0.5)
        np.testing.assert_allclose(mid["wrist"][:3], [0.1, 0.0, 0.1], atol=1e-12)
        eighth = np.array(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.25 * math.pi), dtype=float)
        np.testing.assert_allclose(mid["wrist"][3:], eighth, atol=1e-7)
        self.assertAlmostEqual(ch.evaluate_hand_keyframes(keys, 1.125)["opposition"], 0.103515625, places=12)
        np.testing.assert_allclose(ch.evaluate_hand_keyframes(keys, 1.25)["command"][[index, thumb]], [16.0, 6.0])
        # an actuator that changes mode keeps the earlier mode and value until the later keyframe
        before = ch.evaluate_hand_keyframes(keys, 1.99)
        self.assertEqual(before["mode"].tolist(), [0] * 7)
        self.assertAlmostEqual(float(before["command"][middle]), 2.0, places=12)
        switched = ch.evaluate_hand_keyframes(keys, 2.0)
        self.assertEqual(switched["mode"].tolist(), [0, 0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(float(switched["command"][middle]), 0.012, places=12)
        self.assertAlmostEqual(float(ch.evaluate_hand_keyframes(keys, 2.5)["command"][middle]), 0.016, places=12)
        # wrist speed and acceleration: zero at the keyframes, peaks as predicted
        t = np.linspace(0.0, 1.0, 4001)
        x = np.array([ch.evaluate_hand_keyframes(keys, float(v))["wrist"][0] for v in t])
        speed = np.gradient(x, t)
        acceleration = np.gradient(speed, t)
        peaks = ch.hand_keyframe_wrist_peaks(keys)
        self.assertAlmostEqual(peaks["speed"][0], ch.MINIMUM_JERK_PEAK_RATE * 0.2, places=12)
        self.assertAlmostEqual(float(speed.max()), peaks["speed"][0], delta=1e-4)
        self.assertAlmostEqual(float(np.abs(acceleration[2:-2]).max()), peaks["acceleration"][0], delta=2e-3)
        self.assertLess(abs(float(speed[1])), 1e-4)
        self.assertLess(abs(float(speed[-2])), 1e-4)
        self.assertAlmostEqual(peaks["angular_speed"][0], ch.MINIMUM_JERK_PEAK_RATE * 0.5 * math.pi, places=6)
        np.testing.assert_allclose(peaks["speed"][[1, 2]], 0.0, atol=1e-12)
        with self.assertRaises(ValueError):
            ch.resolve_hand_keyframes(self._keyframes()[:1], names)
        with self.assertRaises(ValueError):
            ch.resolve_hand_keyframes([ch.HandKeyframe(1.0), ch.HandKeyframe(1.0)], names)

    def test_script_kernel_matches_the_host(self):
        """Step the script kernel and compare what it writes with the host evaluation of the keyframes."""
        spec, model, hand = self.spec, self.model, self.hand
        control = ch.create_hand_control(model, spec, hand)
        script = ch.create_hand_script(model, spec, hand, self._keyframes())
        self.assertAlmostEqual(script.duration, 3.0)
        dt = 1.0 / 400.0
        for step in range(1, 1301):
            script.step(control, dt)
            if step % 50 and step != 1:
                continue
            want = script.evaluate(step * dt)
            np.testing.assert_allclose(control.wrist.target.numpy()[0], want["wrist"], atol=2e-6)
            self.assertEqual(int(control.wrist.steps_left.numpy()[0]), 1)
            self.assertAlmostEqual(float(control.opposition_target.numpy()[0]), want["opposition"], places=5)
            np.testing.assert_allclose(control.governor.command.numpy(), want["command"], rtol=1e-5, atol=1e-7)
            self.assertEqual(control.governor.mode.numpy().tolist(), want["mode"].tolist())
        self.assertEqual(int(script.step_count.numpy()[0]), 1300)
        script.reset()
        self.assertEqual(int(script.step_count.numpy()[0]), 0)
        # the wrist limits are checked when the script is created
        fast = [ch.HandKeyframe(0.0, wrist=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)), ch.HandKeyframe(0.1)]
        fast[1].wrist = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            ch.create_hand_script(model, spec, hand, fast)

    def test_grasp_plan(self):
        """Check phases, wrist limits, loop closure and palm stand-off of the can and ball grasp plans."""
        spec = self.spec
        for name in ("can", "ball"):
            plan = ch.grasp_plan(spec, name)
            self.assertEqual(list(plan.phases), list(ch.GRASP_PHASES))
            self.assertAlmostEqual(plan.keyframes[-1].time, sum(ch.GRASP_PHASES.values()))
            self.assertAlmostEqual(plan.hold_tension, spec.cable.tension_hold)
            keys = ch.resolve_hand_keyframes(
                plan.keyframes, self.hand.actuators, start_tension=spec.cable.tension_rest, spec=spec
            )
            peaks = ch.hand_keyframe_wrist_peaks(keys)
            self.assertLess(float(peaks["speed"].max()), ch.HAND_SCRIPT_WRIST_SPEED)
            self.assertLess(float(peaks["acceleration"].max()), ch.HAND_SCRIPT_WRIST_ACCELERATION)
            # the hand ends where it started, open
            np.testing.assert_allclose(keys["wrist"][-1], keys["wrist"][0], atol=1e-12)
            np.testing.assert_allclose(keys["command"][-1], spec.cable.tension_rest)
            self.assertEqual(keys["opposition"][-1], 0.0)
            # at the grasp the object is one standoff away from resting on both palm pads
            palm = np.array(plan.palm_grasp, dtype=float)
            center = ch._quat_rotate_inv_np(palm[3:], np.array(plan.object_xform, dtype=float)[:3] - palm[:3])
            radius = spec.objects[name]["radius"]
            for pad in spec.palm.pads:
                gap = np.linalg.norm(center[:2] - np.array(pad.p0[:2])) - radius - pad.radius
                self.assertAlmostEqual(gap, 0.001, delta=2e-4)
        with self.assertRaises(ValueError):
            ch.grasp_plan(spec, "brick")

    def test_ball_is_enveloped_by_the_tendon_hand(self):
        """Simulate the ball grasp up to the settled grip: pads on the ball, thumb opposed, cables at tension."""
        # fingers close, the thumb turns onto the ball, the thumb cable is pulled; no lift (the lifted
        # grasps with release are run by the registered example tests)
        spec, model, hand, tendons, scene, plan = self.spec, self.model, self.hand, self.tendons, self.scene, self.plan
        dt = 1.0 / (ch.TENDON_FPS * ch.TENDON_SUBSTEPS)
        end = plan.phases["settle"][0] + 0.3
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = ch.create_grasp_collision_pipeline(model)
            solver = ch.make_tendon_hand_grasp_solver(model)
            control = ch.create_hand_control(model, spec, hand, gravity=(0.0, 0.0, -ch.GRAVITY))
            script = ch.create_hand_script(model, spec, hand, plan.keyframes)
            states = [model.state(), model.state()]
            ctrl = model.control()
            contacts = pipeline.contacts()
            control.apply_preload(ctrl)
            start = ch.grasp_state(model, states[0], hand, scene)
            lowest = np.inf
            for step in range(1, round(end / dt) + 1):
                script.step(control, dt)
                control.step(states[0], ctrl, dt)
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], ctrl, contacts, dt)
                states[0], states[1] = states[1], states[0]
                if step % ch.TENDON_SUBSTEPS == 0 and step * dt > 0.2:
                    tension = np.concatenate(list(ch.hand_span_tensions(solver, tendons).values()))
                    lowest = min(lowest, float(tension.min()))
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
        grip = ch.grasp_state(model, states[0], hand, scene)
        # the ball stays on its seat, within a few millimeters of where it lay, and is at rest
        self.assertLess(float(np.linalg.norm(grip["object_pose"][:3] - start["object_pose"][:3])), 0.006)
        self.assertLess(float(np.linalg.norm(grip["object_velocity"])), 0.01)
        touching = {key for key, d in zip(scene.pads.keys, grip["pad_distance"], strict=True) if d < 2.0e-4}
        self.assertTrue({"index/link3", "ring/link3"} <= touching, touching)
        self.assertTrue(any(key.startswith("thumb/") for key in touching), touching)
        self.assertTrue(any(key.startswith("palm/") for key in touching), touching)
        self.assertGreater(float(grip["pad_distance"].min()), -0.5e-3)
        # the thumb has turned onto the ball: well past the angle at which the straight thumb meets it
        joint_state = ch.hand_joint_state(model, states[0], hand)
        self.assertGreater(joint_state["opposition"], math.radians(115.0))
        self.assertLess(joint_state["opposition"], plan.opposition + math.radians(0.5))
        # every cable carries its command: the fingers the hold tension, the thumb its own grasp tension;
        # no cable went slack on the way
        span = ch.hand_span_tensions(solver, tendons)
        for key in hand.actuators:
            wanted = ch.GRASP_POSES["ball"]["thumb_tension"] if key == "thumb" else plan.hold_tension
            np.testing.assert_allclose(span[key], wanted, atol=0.5)
        self.assertGreater(lowest, 0.5)

    def test_block_pinch_plan(self):
        """Check order, commands and geometry of the pinch plan: thumb beside the block, fingers opposite."""
        # three-digit pinch: thumb turned first and kept straight beside the block, index and middle close
        spec = self.spec
        plan = ch.grasp_plan(spec, "block")
        pose = ch.GRASP_POSES["block"]
        self.assertEqual(sorted(plan.phases), sorted(ch.GRASP_PHASES))
        order = list(plan.phases)
        self.assertLess(order.index("oppose"), order.index("approach"))
        self.assertLess(order.index("retreat"), order.index("unoppose"))
        self.assertIsNone(plan.pedestal)
        self.assertAlmostEqual(plan.hold_tension, pose["hold_tension"])
        self.assertAlmostEqual(plan.tilt, pose["tilt"])
        self.assertAlmostEqual(plan.opposition, pose["opposition"])
        keys = ch.resolve_hand_keyframes(
            plan.keyframes, self.hand.actuators, start_tension=spec.cable.tension_rest, spec=spec
        )
        hold = int(np.searchsorted(keys["times"], plan.phases["hold"][1]))
        rest = spec.cable.tension_rest
        # the curl cables hold the shape of the fingers (9 N), the knuckle cables press (40 N)
        self.assertEqual((pose["curl_tension"], pose["hold_tension"]), (9.0, 40.0))
        np.testing.assert_allclose(keys["command"][hold], [9.0, 40.0, 9.0, 40.0, rest, rest, rest])
        self.assertAlmostEqual(keys["opposition"][hold], plan.opposition)
        np.testing.assert_allclose(keys["wrist"][-1], keys["wrist"][0], atol=1e-12)
        np.testing.assert_allclose(keys["command"][-1], rest)
        self.assertEqual(keys["opposition"][-1], 0.0)
        # the block stands on the table; in the palm frame its center is where the pose puts it
        self.assertAlmostEqual(float(plan.object_xform[2]), 0.030, places=6)
        palm = np.array(plan.palm_grasp, dtype=float)
        center = ch._quat_rotate_inv_np(palm[3:], np.array(plan.object_xform, dtype=float)[:3] - palm[:3])
        np.testing.assert_allclose(center, pose["center"], atol=1e-6)
        # the straight thumb at the pinch angle stands 0.5 to 2 mm from the side face of the block, and the
        # planes of the index and middle finger cut the opposite face at least 3 mm from its edges
        kind, size, _, _ = ch.grasp_object_properties(spec, "block")
        origin, rot = ch.digit_frame_in_palm(spec, "thumb", plan.opposition)
        origins, _ = ch.link_frames(spec.digits["thumb"], (0.0, 0.0, 0.0))
        pad = spec.digits["thumb"].pads[2]
        samples = np.linspace(0.0, 1.0, 17)[:, None]
        planar = origins[3][None, :] + np.array(pad.p0)[None, :] + samples * (np.array(pad.p1) - np.array(pad.p0))
        points = ch.lift_points(planar, origin, rot) - center
        # block axes in the palm frame: the long axis is the palm normal
        in_block = np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)
        gap = float(ch.object_surface_distance(kind, size, in_block).min()) - pad.radius
        self.assertGreater(gap, 0.5e-3)
        self.assertLess(gap, 2.0e-3)
        for name in ("index", "middle"):
            z = ch.digit_frame_in_palm(spec, name)[0][2]
            self.assertLess(abs(z - center[2]), size[1] - 0.003)


class TestCableHandChoreography(unittest.TestCase):
    """Pose library and choreography script of the two-cable hand (CPU, no solver steps)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()
        cls.actuators = ["index", "index_knuckle", "middle", "middle_knuckle", "ring", "little", "thumb"]

    def test_pose_library_follows_the_closed_form(self):
        """Check the poses of the two-cable hand: tensions per cable, angles of the closed form, pinch geometry."""
        spec = self.spec
        poses = ch.hand_poses(spec)
        expected = ["rest", "one", "two", "three", "four", "five", "hook_knuckle"]
        expected += ["pinch_index", "pinch_middle", "fist"]
        self.assertEqual(list(poses), expected)
        rest = spec.cable.tension_rest
        for pose in poses.values():
            # one tension per cable, in the order of the actuators; the angles are the two-input closed form
            self.assertEqual(list(pose.tension), self.actuators)
            for name, digit in spec.digits.items():
                tension = [pose.tension[ch.actuator_key(name, c, cable)] for c, cable in enumerate(digit.cable_names)]
                np.testing.assert_allclose(
                    pose.angles[name], ch.frictionless_angles(digit, tension), rtol=0.0, atol=1.0e-9
                )
            self.assertGreaterEqual(min(pose.tension.values()), rest)
            self.assertLessEqual(max(pose.tension.values()), spec.actuator.force_limit)
            self.assertGreaterEqual(pose.opposition, 0.0)
            self.assertLessEqual(pose.opposition, spec.digits["thumb"].mount.hinge_upper)
        fingers = DIGIT_NAMES[:4]
        for count, label in enumerate(("one", "two", "three", "four"), start=1):
            pose = poses[label]
            straight = [name for name in fingers if max(pose.angles[name]) == 0.0]
            self.assertEqual(straight, list(fingers[:count]))
            for name in fingers[count:]:  # the closed fingers sit on their upper stops
                np.testing.assert_allclose(pose.angles[name], [j.upper for j in spec.digits[name].joints])
            self.assertEqual((pose.opposition, pose.tension["thumb"]), ch.CHOREOGRAPHY_FOLDED_THUMB)
        for name in DIGIT_NAMES:
            self.assertEqual(max(poses["five"].angles[name]), 0.0)
            self.assertEqual(max(poses["rest"].angles[name]), 0.0)
        np.testing.assert_allclose(
            poses["fist"].angles["index"], [math.radians(90.0), math.radians(100.0), math.radians(70.0)]
        )
        self.assertEqual(poses["one"].tension["thumb"], 20.0)
        # the pose only two-cable fingers can take: the index finger hooks (outer joints on their stops, first
        # joint straight) while the middle finger bends its first joint to 60 deg with straight outer joints
        hook = poses["hook_knuckle"]
        self.assertEqual((hook.tension["index_knuckle"], hook.tension["middle"]), (rest, rest))
        np.testing.assert_allclose(hook.angles["index"], [0.0, math.radians(100.0), math.radians(70.0)], atol=1.0e-6)
        np.testing.assert_allclose(hook.angles["middle"], [math.radians(60.0), 0.0, 0.0], atol=1.0e-6)
        for name in ("ring", "little", "thumb"):
            self.assertEqual(max(hook.angles[name]), 0.0)
        # pinch poses: the layout's opposition angles, the finger in its pinch pose, pads the chosen gap apart
        for finger, angle in (("index", 118.0), ("middle", 128.0)):
            pose = poses[f"pinch_{finger}"]
            self.assertAlmostEqual(pose.opposition, math.radians(angle), delta=1.0e-12)
            ready = ch.two_cable_poses(spec, finger)["pinch_ready"]
            self.assertAlmostEqual(pose.tension[finger], ready["curl"], places=9)
            self.assertAlmostEqual(pose.tension[f"{finger}_knuckle"], ready["knuckle"], places=9)
            gap = ch.pinch_pad_gap(spec, finger, pose.opposition, pose.angles["thumb"], pose.angles[finger])
            self.assertAlmostEqual(gap, ch.CHOREOGRAPHY_PINCH_GAP, delta=2.0e-6)
            self.assertGreater(pose.tension["thumb"], rest)
        # gravity enters the thumb tension of a pinch, and the pads keep their gap
        heavy = ch.hand_poses(spec, gravity=(-ch.GRAVITY, 0.0, 0.0))["pinch_index"]
        self.assertNotAlmostEqual(heavy.tension["thumb"], poses["pinch_index"].tension["thumb"], delta=0.05)
        gap = ch.pinch_pad_gap(spec, "index", heavy.opposition, heavy.angles["thumb"], heavy.angles["index"])
        self.assertAlmostEqual(gap, ch.CHOREOGRAPHY_PINCH_GAP, delta=2.0e-6)

    def test_keyframes_are_loopable_and_within_the_drive_limits(self):
        """Check that the choreography keyframes close into a loop, respect the limits and match the schedule."""
        spec = self.spec
        home = (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0)
        keys, schedule, poses = ch.hand_choreography_keyframes(spec, home_wrist=home)
        resolved = ch.resolve_hand_keyframes(keys, self.actuators, start_tension=spec.cable.tension_rest, spec=spec)
        times = resolved["times"]
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(float(times[-1]), 16.6, delta=1.0e-9)
        # loopable: the last keyframe is the first
        for name in ("wrist", "opposition", "command", "mode"):
            np.testing.assert_allclose(resolved[name][-1], resolved[name][0], atol=1.0e-12)
        np.testing.assert_allclose(resolved["command"][0], spec.cable.tension_rest)
        self.assertTrue(np.all(resolved["mode"] == ch.GOVERNOR_FORCE))
        self.assertLessEqual(float(resolved["command"].max()), spec.actuator.force_limit)
        self.assertGreaterEqual(float(resolved["command"].min()), spec.cable.tension_rest)
        # the wave turns the palm about its normal without moving the palm origin
        np.testing.assert_allclose(resolved["wrist"][:, :3], np.tile(home[:3], (len(times), 1)), atol=1.0e-9)
        peaks = ch.hand_keyframe_wrist_peaks(resolved)
        self.assertLessEqual(float(peaks["angular_speed"].max()), ch.HAND_SCRIPT_WRIST_ANGULAR_SPEED)
        self.assertGreater(float(peaks["angular_speed"].max()), 1.0)
        # schedule: the named holds in order, each ending on a keyframe, the pose commanded during the hold
        order = ["one", "two", "three", "four", "five", "hook_knuckle", "pinch_index", "pinch_middle", "fist", "rest"]
        self.assertEqual([entry["pose"] for entry in schedule], order)
        for entry in schedule:
            self.assertGreaterEqual(entry["end"] - entry["start"], 0.05 - 1.0e-9)
            value = ch.evaluate_hand_keyframes(resolved, entry["end"])
            pose = poses[entry["pose"]]
            np.testing.assert_allclose(value["command"], [pose.tension[key] for key in self.actuators], atol=1.0e-9)
            self.assertAlmostEqual(value["opposition"], pose.opposition, delta=1.0e-9)
            earlier = ch.evaluate_hand_keyframes(resolved, entry["start"])
            np.testing.assert_allclose(earlier["command"], value["command"], atol=1.0e-9)
        # without a home pose the keyframes carry no wrist motion
        fixed, _schedule, _poses = ch.hand_choreography_keyframes(spec)
        self.assertTrue(all(key.wrist is None for key in fixed))
        self.assertEqual(len(fixed), len(keys))
        # seating stroke: rest -> 16 N -> rest with the thumb opposed while the cables are loaded
        seat = ch.resolve_hand_keyframes(
            ch.hand_seating_keyframes(spec), self.actuators, start_tension=spec.cable.tension_rest, spec=spec
        )
        self.assertEqual(float(seat["command"].max()), ch.CHOREOGRAPHY_SEATING[1])
        np.testing.assert_allclose(seat["command"][[0, -1]], spec.cable.tension_rest)
        loaded = seat["command"].max(axis=1) > spec.cable.tension_rest
        np.testing.assert_allclose(seat["opposition"][loaded], ch.CHOREOGRAPHY_SEATING[0])
        self.assertEqual(float(seat["opposition"][-1]), 0.0)

    def test_choreography_script_on_the_tendon_hand(self):
        """Create the choreography script for a hand model and check the seating stroke and the first wave swing."""
        # The seating stroke, the wave and the first counted pose are simulated by the registered test of
        # the hand example; here the script is created and stepped on its own, without the solver.
        spec = self.spec
        palm_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.1), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.5 * math.pi)
        )
        gravity = (0.0, 0.0, -ch.GRAVITY)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, hand, _tendons = ch.build_tendon_hand_model(
                spec, wrist="kinematic", palm_xform=palm_xform, device=CPU
            )
            control = ch.create_hand_control(model, spec, hand, gravity=gravity)
            script, schedule, poses = ch.hand_choreography(model, spec, hand, gravity=gravity)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertTrue(schedule[0]["seating"])
        self.assertAlmostEqual(script.duration, schedule[0]["end"] + 16.6, delta=1.0e-6)
        self.assertEqual(schedule[-1]["pose"], "rest")
        self.assertAlmostEqual(schedule[-1]["end"], script.duration, delta=1.0e-9)
        self.assertIn("pinch_middle", poses)
        # middle of the seating stroke: every cable at the seating tension, the thumb opposed
        seated = script.evaluate(1.35)
        np.testing.assert_allclose(seated["command"], ch.CHOREOGRAPHY_SEATING[1], atol=1.0e-6)
        self.assertAlmostEqual(seated["opposition"], ch.CHOREOGRAPHY_SEATING[0], places=6)
        # end of the seating stroke: rest tension, no opposition, the palm at its home pose
        rest = script.evaluate(schedule[0]["end"])
        np.testing.assert_allclose(rest["command"], spec.cable.tension_rest, atol=1.0e-6)
        self.assertAlmostEqual(rest["opposition"], 0.0, places=6)
        np.testing.assert_allclose(rest["wrist"], np.array(palm_xform, dtype=float), atol=1.0e-6)
        # end of the first wave swing: the palm turned by the wave angle about its normal, about its origin
        swing = script.evaluate(schedule[0]["end"] + 0.3)
        turn = wp.transform_multiply(wp.transform_inverse(palm_xform), wp.transform(*swing["wrist"].tolist()))
        axis, angle = wp.quat_to_axis_angle(wp.transform_get_rotation(turn))
        self.assertAlmostEqual(float(angle) * float(axis[1]), math.radians(20.0), delta=math.radians(0.01))
        np.testing.assert_allclose(np.array(wp.transform_get_translation(turn)), 0.0, atol=1.0e-6)
        # the script kernel writes the same commands into the hand control
        dt = 1.0 / (ch.TENDON_FPS * ch.TENDON_SUBSTEPS)
        steps = round(1.35 / dt)
        for _ in range(steps):
            script.step(control, dt)
        want = script.evaluate(steps * dt)
        np.testing.assert_allclose(control.governor.command.numpy(), want["command"], rtol=1.0e-5)
        self.assertAlmostEqual(float(control.opposition_target.numpy()[0]), want["opposition"], places=5)


TWO_CABLE_DIGITS = ("index", "middle")


class TestCableHandTwoCableSpec(unittest.TestCase):
    """A digit carries a list of cables: route, closed form and tables of the two-cable fingers."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_two_cable_digits_match_frozen_design(self):
        """Compare the cables, springs and masses of the default spec with the frozen values."""
        for name in DIGIT_NAMES:
            digit = self.spec.digits[name]
            self.assertEqual([link.mass for link in digit.links], REF2_LINK_MASSES[name], name)
            self.assertEqual(digit.cable_names, ["curl", "knuckle"] if name in TWO_CABLE_DIGITS else ["flexor"])
            self.assertEqual(digit.cables[0].slider_body, "slider")
        for name in TWO_CABLE_DIGITS:
            digit = self.spec.digits[name]
            for joint, ref in zip(digit.joints, REF2_JOINTS, strict=True):
                for key, value in ref.items():
                    got = getattr(joint, key)
                    if isinstance(value, str):
                        self.assertEqual(got, value)
                    else:
                        self.assertAlmostEqual(got, value, delta=1.0e-12, msg=f"{name} {joint.name} {key}")
            for cable_name, ref in {**REF2_CABLES["index"], **REF2_CABLES[name]}.items():
                cable = digit.cable(cable_name)
                self.assertEqual(
                    {s.name: [s.body, *s.pos, s.radius, s.sense] for s in cable.stations}, ref["stations"], name
                )
                if "role" not in ref:
                    continue
                self.assertEqual(cable.role, ref["role"])
                self.assertEqual(cable.slider_body, ref["slider_body"])
                self.assertEqual(cable.plane_z, ref["plane_z"])
                self.assertEqual(list(cable.moment_arms), ref["moment_arms"])
                self.assertEqual(list(cable.slider.rest_anchor), ref["rest_anchor"])
                self.assertEqual(cable.slider.travel_max, ref["travel_max"])
                self.assertEqual(list(cable.slider.pull_direction), [-1.0, 0.0])
        # the long cable's arms are the joints' sheave radii; the matrix has one row per cable
        arms = ch.moment_arm_matrix(self.spec.digits["index"])
        np.testing.assert_array_equal(arms, [[0.005, 0.0065, 0.005], [0.008, 0.0, 0.0]])
        self.assertEqual(self.spec.rod.segment_length_target, 0.001)
        self.assertEqual(ch.rod_inertia_reference(self.spec), 2.5e-7)

    def test_palm_layout_matches_frozen_design(self):
        """Compare mounts, thumb hinge, palm boxes and the opposition motor with the frozen values."""
        spec = self.spec
        for name, origin in REF2_LAYOUT["finger_origin"].items():
            self.assertEqual(list(spec.digits[name].mount.frame_origin), origin, name)
        mount = spec.digits["thumb"].mount
        self.assertEqual(list(mount.hinge_point), REF2_LAYOUT["thumb_hinge_point"])
        self.assertEqual(list(mount.hinge_axis), REF2_LAYOUT["thumb_hinge_axis"])
        self.assertEqual(mount.hinge_upper, REF2_LAYOUT["thumb_hinge_upper"])
        self.assertEqual(list(mount.frame_origin_at_phi0), REF2_LAYOUT["thumb_origin"])
        self.assertEqual(spec.palm.mass, REF2_LAYOUT["palm_mass"])
        self.assertEqual(spec.palm.thumb_column.mass, REF2_LAYOUT["thumb_column_mass"])
        plain = spec.to_dict()
        for box, ref in zip(plain["palm"]["solid_boxes"], REF2_LAYOUT["solid_boxes"], strict=True):
            self.assertEqual(set(box), set(ref), ref["name"])
            for key, value in ref.items():
                if isinstance(value, str):
                    self.assertEqual(box[key], value)
                else:
                    np.testing.assert_allclose(box[key], value, rtol=0.0, atol=1.0e-9, err_msg=f"{ref['name']} {key}")
        motor = plain["actuator"]["opposition_motor"]
        for key, value in REF2_LAYOUT["opposition_motor"].items():
            np.testing.assert_allclose(motor[key], value, rtol=0.0, atol=1.0e-9, err_msg=key)
        # palm mass properties use the boxes' own height ranges and frames: the thenar housing lies along
        # the tilted hinge line, so its center is the pivot plus the box center along the box axes
        com, inertia = ch._palm_mass_properties(spec)
        self.assertTrue(np.all(np.linalg.eigvalsh(inertia) > 0.0))
        only = ch.default_hand_spec()
        only.palm.solid_boxes = [only.palm.solid_boxes[-1]]
        com_thenar, _ = ch._palm_mass_properties(only)
        box = REF2_LAYOUT["solid_boxes"][-1]
        expected = np.array(box["pivot"]) + 0.5 * (box["x"][0] + box["x"][1]) * np.array(box["axes"][0])
        np.testing.assert_allclose(com_thenar, expected, atol=1.0e-9)
        self.assertTrue(np.all(np.isfinite(com)))

    def test_dict_layouts_and_round_trip(self):
        """Write the spec as dicts with a cable list per digit and read it back."""
        plain = self.spec.to_dict()
        for name in DIGIT_NAMES:
            self.assertNotIn("stations", plain["digits"][name])
            self.assertEqual(
                list(plain["digits"][name]["cables"][0])[:7],
                ["name", "role", "slider_body", "plane_z", "moment_arms", "stations", "slider"],
            )
            # every finger cable carries the envelope of its linear unit; the thumb unit is a cylinder
            self.assertEqual("actuator_unit" in plain["digits"][name]["cables"][0], name != "thumb")
        self.assertEqual(plain["digits"]["index"]["cables"][1]["actuator_unit"]["box"], [0.042, 0.024, 0.012])
        self.assertEqual(plain["digits"]["index"]["cables"][0]["actuator_unit"]["rear_face_x"], -0.078)
        again = ch.HandSpec.from_dict(plain)
        self.assertEqual(again.to_dict(), plain)
        self.assertEqual(again.digits["index"].cable("knuckle").stations[0].name, "Ak")
        # notes a file keeps next to the numbers are ignored when reading
        noted = self.spec.to_dict()
        noted["digits"]["index"]["cables"][0]["stroke_budget"] = {"needed": 0.03}
        noted["palm"]["status"] = "a note"
        self.assertEqual(ch.HandSpec.from_dict(noted).to_dict(), plain)

    def test_single_cable_accessors(self):
        """Use the single-cable accessors on a one-cable digit and expect an error on a digit with two cables."""
        ring = self.spec.to_dict()["digits"]["ring"]
        self.assertEqual(ring["cables"][0]["moment_arms"], [joint["sheave_radius"] for joint in ring["joints"]])
        self.assertIs(self.spec.digits["ring"].stations, self.spec.digits["ring"].cables[0].stations)
        index = self.spec.digits["index"]
        for call in (
            lambda: index.stations,
            lambda: index.slider,
            lambda: ch.digit_route(index),
            lambda: ch.cable_takeup(index, (0.1, 0.1, 0.1)),
            lambda: ch.frictionless_angles(index, 10.0),
            lambda: ch.frictionless_angles(index, (10.0, 10.0, 10.0)),
            lambda: ch.CapstanStatics(self.spec, "index"),
            lambda: ch.sheave_station_indices(index, "knuckle"),
            lambda: index.cable("flexor"),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_routes_of_both_cables(self):
        """Compare lengths, spans, wraps, rod and tendon data of both cables with the frozen values; check the take-up."""
        for name in TWO_CABLE_DIGITS:
            digit = self.spec.digits[name]
            for cable_name, ref in {**REF2_CABLES["index"], **REF2_CABLES[name]}.items():
                data = ch._digit_geometry(self.spec, name, cable_name)
                self.assertAlmostEqual(data["cable_length"], ref["cable_length"], delta=1.0e-11)
                self.assertEqual(data["segments"], ref["segments"])
                if "free_spans" not in ref:
                    continue
                np.testing.assert_allclose(data["free_spans"], ref["free_spans"], rtol=0.0, atol=LENGTH_TOL)
                np.testing.assert_allclose(data["wrap_extension"][1:-1], ref["wrap_extension"], atol=ANGLE_TOL)
                np.testing.assert_allclose(data["wrap_full_flexion"][1:-1], ref["wrap_full_flexion"], atol=ANGLE_TOL)
                self.assertAlmostEqual(data["takeup_full_flexion"], ref["takeup_full_flexion"], delta=1.0e-11)
                links = ch.tendon_links(self.spec, name, cable_name)
                np.testing.assert_allclose([link.compliance for link in links], ref["tendon_compliance"], rtol=1.0e-9)
                self.assertEqual({link.offset[2] for link in links}, {ref["plane_z"]})
                self.assertEqual(links[0].body, ref["slider_body"])
                if name == "index":
                    barrels = [s.collision_barrel_radius for s in ch.rod_stations(self.spec, name, cable_name)]
                    np.testing.assert_allclose(barrels, ref["rod_barrel_radius"], rtol=1.0e-9)
            # the cable length is conserved when each slider travels by its own take-up, and every pulley
            # keeps a wrap over the whole joint range (the curl idler on the first joint has the least)
            arms = ch.moment_arm_matrix(digit)
            smallest = math.inf
            for q in _pose_grid(digit, 4):
                for c, cable in enumerate(digit.cables):
                    takeup = ch.cable_takeup(digit, q, cable.name)
                    self.assertAlmostEqual(takeup, float(arms[c] @ np.array(q)), delta=1.0e-15)
                    route = ch.digit_route(digit, q, takeup, cable=cable.name)
                    self.assertAlmostEqual(route.length, ch.digit_route(digit, cable=c).length, delta=1.0e-9)
                    smallest = min(smallest, float(route.wraps[1:-1].min()))
            self.assertGreater(smallest, math.radians(20.0))
        self.assertEqual(ch.joint_sheave_stations(self.spec.digits["index"], "knuckle"), {0: 2})
        self.assertEqual(ch.sheave_station_indices(self.spec.digits["index"], "curl"), [2, 4, 6])

    def test_two_input_closed_form_tables(self):
        """Compare the staircases on each cable, the tension grid and the named poses with the frozen tables."""
        for name in TWO_CABLE_DIGITS:
            tables = ch._two_cable_reference(self.spec, name)
            for part, ref_part in REF2_TWO_CABLE[name].items():
                ref_rows = ref_part
                if part == "poses":
                    self.assertEqual([row["pose"] for row in tables[part]], list(ref_part))
                    self.assertEqual(tuple(ref_part), ch._TWO_CABLE_POSES)
                    ref_rows = list(ref_part.values())
                self.assertEqual(len(tables[part]), len(ref_rows), part)
                for row, (tension, q, travel) in zip(tables[part], ref_rows, strict=True):
                    np.testing.assert_allclose(row["T"], tension, rtol=0.0, atol=1.0e-8, err_msg=part)
                    self.assertEqual(row["T_by_cable"], dict(zip(("curl", "knuckle"), row["T"], strict=True)))
                    np.testing.assert_allclose(row["q"], q, rtol=0.0, atol=ANGLE_TOL, err_msg=f"{part} {tension}")
                    np.testing.assert_allclose(row["travel"], travel, rtol=0.0, atol=LENGTH_TOL, err_msg=part)
                    parts = [row[k] for k in ("takeup", "cable_stretch", "anchor_stretch", "hinge_stretch")]
                    np.testing.assert_allclose(np.sum(parts, axis=0), row["travel"], rtol=0.0, atol=1.0e-15)
        # thresholds: the tension at which each cable alone (the other at rest) starts to move a joint and
        # has it fully flexed
        digit = self.spec.digits["index"]
        arms = ch.moment_arm_matrix(digit)
        rest = self.spec.cable.tension_rest
        for c, cable in enumerate(digit.cable_names):
            for j, joint in enumerate(digit.joints):
                if arms[c, j] == 0.0:
                    continue
                ref = REF2_THRESHOLDS[cable][joint.name]
                others = rest * (arms[:, j].sum() - arms[c, j])
                self.assertAlmostEqual((joint.spring_preload - others) / arms[c, j], ref["start"], delta=1.0e-9)
                full = (joint.spring_preload + joint.spring_stiffness * joint.upper - others) / arms[c, j]
                self.assertAlmostEqual(full, ref["full"], delta=1.0e-8)

    def test_cross_term_between_the_cables(self):
        """Check the two couplings of the cables at the first joint: carriage travel and the tension rule."""
        spec, name = self.spec, "index"
        digit = spec.digits[name]
        rest = spec.cable.tension_rest
        # only the knuckle cable is pulled: the curl carriage follows by its idler radius times the angle,
        # plus the opening of the first hinge under the extra knuckle tension
        row = ch.frictionless_row(spec, name, {"curl": rest, "knuckle": 20.0})
        self.assertAlmostEqual(row["q"][0], 0.625, delta=1.0e-12)
        self.assertEqual(row["q"][1:], [0.0, 0.0])
        self.assertAlmostEqual(row["travel"][0], 0.005 * 0.625 + spec.cable.hinge_compliance * 18.0, delta=1.0e-15)
        self.assertEqual(row["cable_stretch"][0], 0.0)
        # tensions may be given as a dict by cable name or as a sequence in the order of the cables
        self.assertEqual(ch.frictionless_angles(digit, (rest, 20.0)), row["q"])
        self.assertEqual(ch.cable_joint_torques(digit, (10.0, 20.0)), [0.05 + 0.16, 0.065, 0.05])
        # the knuckle tension that holds a first-joint angle falls by 5/8 of the curl tension
        for angle in (0.0, 0.4, 1.2):
            held = []
            for curl in (2.0, 6.0, 10.0, 14.0):
                tension = ch.cable_tensions_for_knuckle(spec, name, curl, angle)
                if tension["knuckle"] > rest:
                    held.append(tension)
                    self.assertAlmostEqual(ch.frictionless_angles(digit, tension)[0], angle, delta=1.0e-12)
            for a, b in itertools.pairwise(held):
                self.assertAlmostEqual((b["knuckle"] - a["knuckle"]) / (b["curl"] - a["curl"]), -0.625, delta=1.0e-12)
        # an extra torque at the first joint for pressing: 8 mm of arm per newton of knuckle tension
        press = ch.cable_tensions_for_knuckle(spec, name, 6.0, 1.0, knuckle_torque=0.04)
        hold = ch.cable_tensions_for_knuckle(spec, name, 6.0, 1.0)
        self.assertAlmostEqual(press["knuckle"] - hold["knuckle"], 0.04 / 0.008, delta=1.0e-12)
        # the commands stay inside the actuator range
        self.assertEqual(ch.cable_tensions_for_knuckle(spec, name, 100.0, 0.0), {"curl": 60.0, "knuckle": rest})
        # gravity: without gravity the gravity form is the frictionless one; palm down it flexes the knuckle
        tension = {"curl": 6.0, "knuckle": 20.0}
        plain = ch.frictionless_angles(digit, tension)
        np.testing.assert_allclose(ch.gravity_angles(digit, tension, (0.0, 1.0), gravity=0.0), plain, atol=1.0e-12)
        sag = ch.gravity_angles(digit, tension, (0.0, 1.0))
        lift = ch.gravity_angles(digit, tension, (0.0, -1.0))
        self.assertGreater(sag[0], plain[0] + math.radians(2.0))
        self.assertLess(lift[0], plain[0] - math.radians(2.0))


class TestCableHandTwoCableFinger(unittest.TestCase):
    """One two-cable finger: skeleton and governor, massless tendons (short CPU run), rod build (CUDA rest hold)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_skeleton_has_one_carriage_per_cable(self):
        """Build the index finger of the default spec and check carriages, keys, limits and the governor."""
        spec = self.spec
        model, skeleton = ch.build_single_finger_model(spec, "index", base="fixed", device=CPU)
        self.assertEqual(skeleton.cable_names, ("curl", "knuckle"))
        self.assertEqual(list(skeleton.bodies), ["base", "link1", "link2", "link3", "slider", "slider_knuckle"])
        self.assertEqual(model.body_count, 6)
        self.assertEqual(model.joint_count, 6)
        self.assertEqual(skeleton.joints, [*skeleton.joint_flexion, *skeleton.joint_sliders])
        labels = [model.body_label[skeleton.bodies[key]] for key in ("slider", "slider_knuckle")]
        self.assertEqual(labels, ["index_slider", "index_slider_knuckle"])
        self.assertEqual(skeleton.slider_key("curl"), "slider")
        self.assertEqual(skeleton.slider_body("knuckle"), skeleton.bodies["slider_knuckle"])
        np.testing.assert_array_equal(skeleton.moment_arms, ch.moment_arm_matrix(spec.digits["index"]))
        upper = model.joint_limit_upper.numpy()[list(skeleton.dof_sliders)]
        np.testing.assert_allclose(upper, [0.037, 0.022], rtol=1.0e-6)
        self.assertEqual(skeleton.slider_travel_maxs, (0.037, 0.022))
        # each carriage's center of mass is its cable anchor, in its cable plane
        com = model.body_com.numpy()
        for cable in spec.digits["index"].cables:
            expected = (*cable.slider.rest_anchor, cable.plane_z)
            np.testing.assert_allclose(com[skeleton.bodies[cable.slider_body]], expected, atol=1.0e-9)
        # the accessors of a one-cable digit refuse a digit with two carriages
        for attribute in ("joint_slider", "q_slider", "dof_slider", "slider_travel_min", "slider_travel_max"):
            with self.assertRaises(ValueError):
                getattr(skeleton, attribute)
        # governor: one actuator per cable, commands in the order of `actuators`
        gov = ch.create_actuator_governor(model, spec, skeleton)
        self.assertEqual(gov.count, 2)
        self.assertEqual(gov.actuators, [("index", "curl"), ("index", "knuckle")])
        np.testing.assert_array_equal(
            gov.slider_body.numpy(), [skeleton.bodies["slider"], skeleton.bodies["slider_knuckle"]]
        )
        # reference rate per actuator: the curl cable of a two-cable finger is ramped at 24 mm/s, the knuckle
        # cable and the single cables at the actuator's 30 mm/s minus the governor's margin of 2 mm/s (the
        # rate of every cable without an entry of its own)
        self.assertEqual(ch.CURL_REFERENCE_RATE_LIMIT, 0.024)
        self.assertEqual(spec.digits["index"].cable("curl").reference_rate_limit, 0.024)
        self.assertIsNone(spec.digits["index"].cable("knuckle").reference_rate_limit)
        self.assertIsNone(spec.digits["ring"].cables[0].reference_rate_limit)
        np.testing.assert_allclose(gov.rate_limit.numpy(), [0.024, 0.028], rtol=1.0e-6)
        self.assertAlmostEqual(ch.actuator_reference_rate(spec, "index", "curl"), 0.024, places=9)
        self.assertAlmostEqual(ch.actuator_reference_rate(spec, "ring"), 0.028, places=9)
        self.assertEqual({ch.actuator_reference_rate(spec, name) for name in ONE_CABLE_DIGITS}, {0.03 - 0.002})
        round_trip = ch.HandSpec.from_dict(spec.to_dict())
        self.assertEqual(round_trip.digits["middle"].cable("curl").reference_rate_limit, 0.024)
        # rod segment length per cable: the thumb's cable asks for 2 mm, every other cable takes the common
        # target of the rod spec (None)
        lengths = {(name, c.name): c.rod_segment_length for name, d in spec.digits.items() for c in d.cables}
        self.assertEqual(lengths.pop(("thumb", "flexor")), ch.THUMB_ROD_SEGMENT_LENGTH)
        self.assertEqual(ch.THUMB_ROD_SEGMENT_LENGTH, 0.002)
        self.assertEqual(set(lengths.values()), {None})
        self.assertEqual(round_trip.digits["thumb"].cables[0].rod_segment_length, 0.002)
        # both cables stepped to the force limit: each reference moves at its own rate in the first substep
        fast = ch.create_actuator_governor(model, spec, skeleton, command=spec.actuator.force_limit)
        probe_control = model.control()
        fast.step(model.state(), probe_control, 1.0 / 1200.0)
        np.testing.assert_allclose(fast.ref_rate.numpy(), [0.024, 0.028], atol=1.0e-4)
        gov.set_command([6.0, 20.0])
        np.testing.assert_allclose(gov.command.numpy(), [6.0, 20.0])
        # a staircase with one column per actuator commands the two cables independently
        stair = ch.create_staircase_command(spec, levels=[[2.0, 2.0], [6.0, 20.0], [18.0, 2.0]], device=CPU, blend=0.0)
        self.assertEqual(stair.levels.shape, (3, 2))
        for expected in ([2.0, 2.0], [2.0, 2.0], [6.0, 20.0], [6.0, 20.0], [18.0, 2.0]):
            stair.step(gov.command, 0.5)
            np.testing.assert_allclose(gov.command.numpy(), expected)
        # a start travel per cable
        _model, moved = ch.build_single_finger_model(spec, "index", travel=(0.002, 0.001), device=CPU)
        np.testing.assert_allclose(_model.joint_q.numpy()[list(moved.q_sliders)], [0.002, 0.001], rtol=1.0e-6)
        with self.assertRaises(ValueError):
            ch.build_single_finger_model(spec, "index", travel=(0.002, 0.001, 0.0), device=CPU)

    def test_tendon_finger_follows_the_two_input_closed_form(self):
        """Simulate the two-cable tendon finger through two poses (CPU) and compare with the closed form."""
        spec, name = self.spec, "index"
        model, skeleton, tendons = ch.build_tendon_finger_cables_model(spec, name, gravity=(0.0, 0.0, 0.0), device=CPU)
        self.assertEqual([tendon.cable for tendon in tendons], ["curl", "knuckle"])
        self.assertEqual(model.tendon_count, 2)
        self.assertEqual(model.tendon_link_count, 12)
        offsets = model.tendon_link_offset.numpy()
        bodies = model.tendon_link_body.numpy()
        for tendon, cable in zip(tendons, spec.digits[name].cables, strict=True):
            links = slice(tendon.link_start, tendon.link_start + tendon.link_count)
            np.testing.assert_allclose(offsets[links, 2], cable.plane_z, atol=1.0e-9)
            self.assertEqual(int(bodies[tendon.link_start]), skeleton.bodies[cable.slider_body])
            self.assertEqual(tendon.stations, [station.name for station in cable.stations])
        # the two rolling links on the first joint sit side by side on the joint axis, 3 mm apart
        on_joint = [tendons[0].link_start + 2, tendons[1].link_start + 2]
        np.testing.assert_allclose(offsets[on_joint], [[0.0, 0.0, 0.0015], [0.0, 0.0, -0.0015]], atol=1.0e-9)
        with self.assertRaises(ValueError):
            ch.build_tendon_finger_model(spec, name, device=CPU)  # the one-cable builder names its limit
        with self.assertRaises(ValueError):
            ch.build_tendon_finger_cables_model(spec, name, mu=0.05, device=CPU)  # friction: one cable only

        # plain force mode on both carriages: 0.25 s ramp and 0.5 s hold per pose
        ke = model.joint_target_ke.numpy()
        ke[list(skeleton.dof_sliders)] = 0.0
        model.joint_target_ke.assign(ke)
        solver = ch.make_tendon_solver(model)
        state_0, state_1, control = model.state(), model.state(), model.control()
        joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
        ch.write_spring_preload(joint_f, skeleton)
        dt = 1.0 / (ch.TENDON_FPS * ch.TENDON_SUBSTEPS)
        ramp, hold = 15 * ch.TENDON_SUBSTEPS, 30 * ch.TENDON_SUBSTEPS
        rest = spec.cable.tension_rest
        previous = np.array([rest, rest])
        poses = ch.two_cable_poses(spec, name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for pose in ("knuckle_only", "hook"):
                target = np.array(ch.cable_tensions(spec.digits[name], poses[pose]))
                for step in range(ramp + hold):
                    if step <= ramp:
                        joint_f[list(skeleton.dof_sliders)] = previous + (target - previous) * min(step / ramp, 1.0)
                        control.joint_f.assign(joint_f)
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, None, dt)
                    state_0, state_1 = state_1, state_0
                previous = target
                newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
                joint_q = state_0.joint_q.numpy()
                row = ch.frictionless_row(spec, name, poses[pose])
                # Measured on the CPU: angles within 0.07 deg and both carriages within 0.01 mm of the
                # closed form at every hold of the reference tables, tensions equal to the commands.
                np.testing.assert_allclose(
                    joint_q[list(skeleton.q_flexion)], row["q"], atol=math.radians(0.2), err_msg=pose
                )
                np.testing.assert_allclose(joint_q[list(skeleton.q_sliders)], row["travel"], atol=1.0e-4, err_msg=pose)
                for tendon, tension in zip(tendons, target, strict=True):
                    np.testing.assert_allclose(ch.tendon_span_tensions(solver, tendon), tension, rtol=0.01)
        self.assertEqual([str(w.message) for w in caught], [])
        # the hook: outer joints fully curled on a straight first joint, a pose one cable cannot take
        self.assertLess(abs(float(joint_q[skeleton.q_flexion[0]])), math.radians(0.2))
        self.assertGreater(float(joint_q[skeleton.q_flexion[1]]), math.radians(99.0))

    def test_rod_finger_builds_with_one_rod_per_cable(self):
        """Build the two-cable rod finger and check counts, cable planes, collision groups and the start pose."""
        spec, name = self.spec, "index"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, skeleton, cables = ch.build_rod_finger_cables_model(spec, name, device=CPU)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual([cable.cable for cable in cables], ["curl", "knuckle"])
        segments = [REF2_CABLES[name][cable.cable]["segments"] for cable in cables]
        self.assertEqual([len(cable.bodies) for cable in cables], segments)
        self.assertEqual(model.body_count, 6 + sum(segments))
        self.assertEqual(model.articulation_count, 1)
        self.assertEqual(ch.rod_start_travels(spec, name), [cable.start_travel for cable in cables])
        group = model.shape_collision_group.numpy()
        shape_body = model.shape_body.numpy()
        transform = model.shape_transform.numpy()
        body_q = model.body_q.numpy()
        groups = []
        for cable, cable_spec in zip(cables, spec.digits[name].cables, strict=True):
            shapes = [*cable.rod_shapes, *(s for triple in cable.station_shapes.values() for s in triple)]
            # a rod and its own pulleys share one collision group that no other shape uses
            own = {int(group[s]) for s in shapes}
            self.assertEqual(own, {cable.collision_group})
            self.assertEqual(int(np.sum(group == cable.collision_group)), len(shapes))
            groups.append(cable.collision_group)
            self.assertEqual(list(cable.station_shapes), [s.name for s in cable_spec.stations if s.radius > 0.0])
            # barrels and rod lie in the cable plane (the base frame is the digit frame, the world frame here)
            barrels = [triple[0] for triple in cable.station_shapes.values()]
            centers = [
                wp.transform_point(wp.transform(*body_q[shape_body[s]]), wp.vec3(*transform[s][:3])) for s in barrels
            ]
            np.testing.assert_allclose([c[2] for c in centers], cable_spec.plane_z, atol=1.0e-7)
            np.testing.assert_allclose(body_q[cable.bodies, 2], cable_spec.plane_z, atol=1.0e-7)
            self.assertEqual(cable.plane_z, cable_spec.plane_z)
            # the end anchor is on the body of the cable's last station
            end = int(model.joint_parent.numpy()[cable.anchor_joints[1]])
            self.assertEqual(end, skeleton.bodies[cable_spec.stations[-1].body])
            probe = ch.RodTensionProbe(model, skeleton, cable)
            self.assertLess(np.abs(probe.lateral_offset(body_q)).max(), 1.0e-7)
            self.assertLess(probe.anchor_gaps(body_q).max(), 1.0e-6)
        self.assertEqual(len(set(groups)), 2)
        self.assertNotIn(1, groups)
        # the two flanged pulleys on the first joint are 3 mm apart and do not overlap along the axis
        hardware = spec.station_hardware
        self.assertLess(2.0 * (hardware.groove_half_width + hardware.flange_thickness), 0.003)

    def test_rod_finger_rest_hold_cuda(self):
        """Hold the two-cable rod finger for 0.1 s at the rest tension on CUDA: finite, straight, in its grooves."""
        if CUDA is None:
            self.skipTest("needs CUDA")
        spec, name = self.spec, "index"
        model, skeleton, cables = ch.build_rod_finger_cables_model(spec, name, device=CUDA)
        dt = 1.0 / (ch.ROD_FPS * ch.ROD_SUBSTEPS)
        pipeline = ch.create_rod_collision_pipeline(model)
        self.assertEqual(ch.rod_contact_buffer(spec, name), 1024)
        solver = ch.create_rod_solver(
            model,
            dt,
            inertia_reference=ch.rod_inertia_reference(spec),
            rigid_body_contact_buffer_size=ch.rod_contact_buffer(spec, name),
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = pipeline.contacts()
        gov = ch.create_actuator_governor(model, spec, skeleton)
        ch.create_spring_preload(model, skeleton).apply(control)
        weight = ch.create_body_weight(model, cables[0].weight_gravity)
        for _ in range(6 * ch.ROD_SUBSTEPS):
            state_0.clear_forces()
            weight.apply(state_0)
            gov.step(state_0, control, dt)
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        self.assertTrue(np.isfinite(body_q).all())
        joint_q = wp.clone(model.joint_q)
        joint_qd = wp.clone(model.joint_qd)
        newton.eval_ik(model, state_0, joint_q, joint_qd)
        np.testing.assert_allclose(joint_q.numpy()[list(skeleton.q_flexion)], 0.0, atol=math.radians(0.1))
        # No seating stroke here: after a 1 s hold the mean tensions were 2.12 N (curl) and 2.04 N (knuckle)
        # with single joints between 1.93 and 2.59 N, and the rods 0.011 mm off their groove mid-planes.
        for cable in cables:
            probe = ch.RodTensionProbe(model, skeleton, cable)
            tension = probe.tension(body_q)
            self.assertAlmostEqual(float(tension.mean()), spec.cable.tension_rest, delta=0.3)
            self.assertGreater(float(tension.min()), 1.0)
            self.assertLess(float(tension.max()), 3.5)
            self.assertLess(np.abs(probe.groove_offset(body_q)).max(), 0.0003)


class TestCableHandTwoCableRodFinger(unittest.TestCase):
    """The two-cable finger with rod cables at its working point: seating stroke, then a hold (needs CUDA)."""

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()

    def test_seating_levels(self):
        """Check the seating stroke levels per cable and the rod solver defaults that come from the spec."""
        spec = self.spec
        self.assertEqual(ch.rod_seating_levels(spec, "index"), [[2.0, 2.0], [16.0, 16.0], [2.0, 2.0]])
        self.assertEqual(ch.rod_seating_levels(spec, "thumb"), [[2.0], [16.0], [2.0]])
        self.assertEqual(ch.rod_inertia_reference(spec), 2.5e-7)
        # contacts per body: 512 per cable of 1 mm segments, half of that for the thumb's 2 mm segments
        self.assertEqual([ch.rod_contact_buffer(spec, name) for name in spec.digits], [1024, 1024, 512, 512, 256])
        # the solver defaults are those of the default spec: its reference inertia and the largest finger buffer
        dt = 1.0 / (ch.ROD_FPS * ch.ROD_SUBSTEPS)
        settings = ch.rod_solver_settings(dt)
        self.assertEqual(settings["iterations"], ch.ROD_ITERATIONS)
        self.assertEqual(settings["rigid_body_contact_buffer_size"], 1024)
        self.assertAlmostEqual(settings["rigid_articulation_angular_regularization"], 2.5e-7 / dt**2, delta=1.0e-9)
        self.assertEqual(ch.rod_solver_settings(dt, spec=spec), settings)

    def test_seated_hold_against_closed_form_cuda(self):
        """Seat both cables, rest, then hold a combined pose: angles, travels, tensions and grooves."""
        if CUDA is None:
            self.skipTest("needs CUDA")
        spec, name = self.spec, "index"
        digit = spec.digits[name]
        model, skeleton, cables = ch.build_rod_finger_cables_model(spec, name, device=CUDA)
        self.assertEqual(model.body_count, 288)
        self.assertEqual([len(cable.bodies) for cable in cables], [147, 135])
        substeps = ch.ROD_SUBSTEPS
        dt = 1.0 / (ch.ROD_FPS * substeps)
        pipeline = ch.create_rod_collision_pipeline(model)
        solver = ch.create_rod_solver(
            model,
            dt,
            inertia_reference=ch.rod_inertia_reference(spec),
            rigid_body_contact_buffer_size=ch.rod_contact_buffer(spec, name),
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = pipeline.contacts()
        gov = ch.create_actuator_governor(model, spec, skeleton)
        self.assertEqual(gov.actuators, [(name, cable) for cable in digit.cable_names])
        ch.create_spring_preload(model, skeleton).apply(control)
        weight = ch.create_body_weight(model, cables[0].weight_gravity)
        # The seating stroke without its first rest level (the model starts at rest), then one hold. Times: 1.0 s
        # for the stroke (about 24 mm of carriage travel at the speed limit), 1.5 s for the rest that closes it and
        # 1.5 s for the hold; the joints need about 1 s of a level to arrive, and a 1 s level is not settled.
        hold = {"curl": 12.0, "knuckle": 22.0}
        hold_row = [hold[cable] for cable in digit.cable_names]
        _rest_row, seat_row, rest_row = ch.rod_seating_levels(spec, name)
        period = 0.5
        levels = [seat_row] * 2 + [rest_row] * 3 + [hold_row] * 3
        rest_levels = 5  # the rest ends with this many levels done
        # Stepped commands (blend 0): the level times above are those of stepped commands, whose motion the
        # governor's rate limits shape; the blended loop is covered by the example test of the rod finger.
        stair = ch.create_staircase_command(
            spec, levels=levels, hold=0.25, transition=period - 0.25, device=CUDA, blend=0.0
        )
        pivots = wp.zeros(1, dtype=wp.int32, device=CUDA)

        def frame():
            nonlocal state_0, state_1
            for _ in range(substeps):
                state_0.clear_forces()
                weight.apply(state_0)
                stair.step(gov.command, dt)
                gov.step(state_0, control, dt)
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0
                wp.launch(
                    _sum_kernel, dim=1, inputs=[solver.rigid_articulation_pivot_clamped], outputs=[pivots], device=CUDA
                )

        frame()  # compiles the kernels; the captured frame below is launched from then on
        with wp.ScopedCapture(device=CUDA) as capture:
            frame()
        probes = [ch.RodTensionProbe(model, skeleton, cable) for cable in cables]
        joint_q, joint_qd = wp.clone(model.joint_q), wp.clone(model.joint_qd)
        frames_per_level = round(period * ch.ROD_FPS)

        def read():
            body_q = state_0.body_q.numpy()
            newton.eval_ik(model, state_0, joint_q, joint_qd)
            q = joint_q.numpy()
            return body_q, q[list(skeleton.q_flexion)], gov.travel.numpy().copy()

        rest = None
        for f in range(1, len(levels) * frames_per_level):
            wp.capture_launch(capture.graph)
            if f == rest_levels * frames_per_level - 1:
                rest = read()  # end of the rest that closes the seating stroke
        body_q, q, travel = read()
        self.assertTrue(np.isfinite(body_q).all())
        self.assertEqual(int(pivots.numpy()[0]), 0)
        clearance = spec.station_hardware.groove_half_width - spec.rod.radius

        # after the seating stroke: joints on their lower stops, both rods seated at the rest tension
        np.testing.assert_allclose(rest[1], 0.0, atol=math.radians(0.1))
        for probe in probes:
            actuator_end, tip_end = probe.end_tensions(rest[0])
            self.assertAlmostEqual(actuator_end, spec.cable.tension_rest, delta=0.05 * spec.cable.tension_rest)
            self.assertAlmostEqual(tip_end / actuator_end, 1.0, delta=0.03)

        # the hold against the two-input closed form (rod rules: 2 deg, 0.5 mm, tip / actuator 0.97)
        ref = ch.frictionless_row(spec, name, hold)
        np.testing.assert_allclose(q, ref["q"], atol=math.radians(2.0))
        np.testing.assert_allclose(travel - rest[2], ref["travel"], atol=0.5e-3)
        for probe, cable in zip(probes, digit.cable_names, strict=True):
            actuator_end, tip_end = probe.end_tensions(body_q)
            self.assertAlmostEqual(actuator_end, hold[cable], delta=0.03 * hold[cable])
            self.assertGreater(tip_end / actuator_end, 0.97)
            self.assertLess(np.abs(probe.groove_offset(body_q)).max(), clearance)


class TestCableHandRodHand(unittest.TestCase):
    """The rod hand and its grasp scene as built, without stepping: counts, keys, collision groups, buffer, probes."""

    ACTUATORS = ("index", "index_knuckle", "middle", "middle_knuckle", "ring", "little", "thumb")

    @classmethod
    def setUpClass(cls):
        cls.spec = ch.default_hand_spec()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls.model, cls.hand, cls.cables = ch.build_rod_hand_model(cls.spec, device=CPU)
        cls.build_warnings = [str(w.message) for w in caught]

    def _check_cable_groups(self, model, cables):
        """Every cable's rod and stations share one collision group that no other shape uses."""
        group = model.shape_collision_group.numpy()
        groups = []
        for cable in cables.values():
            shapes = [*cable.rod_shapes, *(s for triple in cable.station_shapes.values() for s in triple)]
            self.assertEqual({int(group[s]) for s in shapes}, {cable.collision_group})
            self.assertEqual(int(np.sum(group == cable.collision_group)), len(shapes))
            groups.append(cable.collision_group)
        first = ch.ROD_FIRST_COLLISION_GROUP
        self.assertEqual(groups, list(range(first, first + len(cables))))
        return groups

    def test_rod_hand_counts(self):
        """Check bodies, rod segments, joints, the articulation and the cable keys of the rod hand."""
        model, hand, cables, spec = self.model, self.hand, self.cables, self.spec
        self.assertEqual(self.build_warnings, [])
        self.assertEqual(hand.actuators, list(self.ACTUATORS))
        self.assertEqual(list(cables), hand.actuators)
        segments = [len(cable.bodies) for cable in cables.values()]
        # one segment count per cable from the spec: 1 mm on the six finger cables, 2 mm on the thumb's
        expected = [ch.rod_segments(spec, digit, cable=cable)[0] for digit, cable in hand.actuator_cables]
        self.assertEqual(segments, expected)
        self.assertEqual(segments[-1], 68)
        self.assertEqual(sum(segments), 931)
        self.assertEqual(model.body_count, 24 + 931)
        # 24 skeleton joints, and per cable one joint between neighboring segments and two anchors
        self.assertEqual(model.joint_count, 24 + sum(count - 1 for count in segments) + 2 * len(cables))
        self.assertEqual(model.articulation_count, 1)
        for (digit, cable_name), cable in zip(hand.actuator_cables, cables.values(), strict=True):
            self.assertEqual((cable.name, cable.cable), (digit, cable_name))

    def test_rod_hand_collision_groups(self):
        """Check that the seven cables of the rod hand use seven collision groups of their own."""
        groups = self._check_cable_groups(self.model, self.cables)
        self.assertEqual(len(set(groups)), 7)

    def test_rod_hand_palm_is_prescribed(self):
        """Check that the palm is a kinematic body and that any other wrist is refused."""
        flags = self.model.body_flags.numpy()
        self.assertTrue(int(flags[self.hand.palm]) & int(newton.BodyFlags.KINEMATIC))
        self.assertEqual(int(np.sum((flags & int(newton.BodyFlags.KINEMATIC)) != 0)), 1)
        for build in (ch.build_rod_hand_model, ch.build_rod_hand_grasp_model):
            with self.assertRaisesRegex(ValueError, "kinematic palm"):
                build(self.spec, wrist="fixed", device=CPU)

    def test_rod_hand_contact_buffer(self):
        """Check the contact buffer of the rod hand: the four fingers' cables share the palm."""
        spec = self.spec
        self.assertEqual(ch.rod_hand_contact_buffer(spec), 3072)
        fingers = [name for name in spec.digits if name != "thumb"]
        self.assertEqual(ch.rod_hand_contact_buffer(spec), sum(ch.rod_contact_buffer(spec, name) for name in fingers))

    def test_rod_hand_probes_and_state(self):
        """Check the probes of the rod hand and the shapes and start values of its read-back."""
        model, hand, cables = self.model, self.hand, self.cables
        probes = ch.create_rod_hand_probes(model, hand, cables)
        self.assertEqual(list(probes), hand.actuators)
        state = ch.hand_rod_state(probes, model.body_q.numpy())
        for key in ("tension_actuator", "tension_tip", "tension_min", "tension_max", "groove_offset", "lateral"):
            self.assertEqual(state[key].shape, (7,), key)
            self.assertTrue(np.isfinite(state[key]).all(), key)
        self.assertEqual(state["anchor_gap"].shape, (7, 2))
        # as built: the anchors are closed, the rods lie in their grooves and carry no tension yet
        self.assertLess(np.abs(state["anchor_gap"]).max(), 1.0e-6)
        self.assertLess(state["groove_offset"].max(), 1.0e-6)
        self.assertLess(np.abs(state["tension_max"]).max(), 0.5)
        self.assertTrue((state["tension_min"] <= state["tension_max"]).all())

    def test_rod_grasp_scene(self):
        """Build the rod hand with pads, table and the can: counts, the object's joint and the collision groups."""
        spec = self.spec
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, hand, cables, scene = ch.build_rod_hand_grasp_model(spec, "can", device=CPU)
        self.assertEqual([str(w.message) for w in caught], [])
        self.assertEqual(list(cables), hand.actuators)
        self.assertEqual(model.body_count, 24 + 931 + 1)
        # the hand is one articulation; the object is a free body outside it
        self.assertEqual(model.articulation_count, 2)
        joint_child = model.joint_child.numpy()
        joint_parent = model.joint_parent.numpy()
        joints = np.where((joint_child == scene.object_body) | (joint_parent == scene.object_body))[0]
        self.assertEqual(len(joints), 1)
        self.assertEqual(int(model.joint_type.numpy()[joints[0]]), int(newton.JointType.FREE))
        self.assertEqual(int(joint_parent[joints[0]]), -1)
        # rods and their stations keep their own groups; pads, object and table share the grasp group
        groups = self._check_cable_groups(model, cables)
        group = model.shape_collision_group.numpy()
        self.assertEqual(int(group[scene.object_shape]), ch.GRASP_COLLISION_GROUP)
        self.assertNotIn(ch.GRASP_COLLISION_GROUP, groups)
        rod_shapes = {s for cable in cables.values() for s in cable.rod_shapes}
        self.assertTrue(all(int(group[s]) != ch.GRASP_COLLISION_GROUP for s in rod_shapes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
