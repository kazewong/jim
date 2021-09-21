from jaxgw.likelihood.detector_projection import construct_arm, detector_tensor


H1_lat = 46 + 27. / 60 + 18.528 / 3600
H1_long = -(119 + 24. / 60 + 27.5657 / 3600)
H1_xarm_azimuth = 125.9994
H1_yarm_azimuth = 215.9994
H1_xarm_tilt = -6.195e-4
H1_yarm_tilt = 1.25e-5

L1_lat = 30 + 33. / 60 + 46.4196 / 3600
L1_long = -(90 + 46. / 60 + 27.2654 / 3600)
L1_xarm_azimuth = 197.7165
L1_yarm_azimuth = 287.7165
L1_xarm_tilt = 0
L1_yarm_tilt = 0

H1_arm1 = construct_arm(H1_long, H1_lat, H1_xarm_tilt, H1_xarm_azimuth)
H1_arm2 = construct_arm(H1_long, H1_lat, H1_yarm_tilt, H1_yarm_azimuth)

L1_arm1 = construct_arm(L1_long, L1_lat, L1_xarm_tilt, L1_xarm_azimuth)
L1_arm2 = construct_arm(L1_long, L1_lat, L1_yarm_tilt, L1_yarm_azimuth)

H1 = detector_tensor(H1_arm1, H1_arm2)
L1 = detector_tensor(L1_arm1, L1_arm2)
