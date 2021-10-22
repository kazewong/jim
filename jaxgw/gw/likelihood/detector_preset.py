from jaxgw.gw.likelihood.detector_projection import construct_arm, detector_tensor, antenna_response, get_detector_response, get_vertex_position_geocentric
import jax.numpy as jnp

# See https://git.ligo.org/lscsoft/bilby/-/tree/master/bilby/gw/detector/detectors for detector parameters.

degree_to_radian = jnp.pi/180

def get_H1():
	H1_lat = (46 + 27. / 60 + 18.528 / 3600) * degree_to_radian
	H1_long = -(119 + 24. / 60 + 27.5657 / 3600) * degree_to_radian
	H1_xarm_azimuth = 125.9994 * degree_to_radian
	H1_yarm_azimuth = 215.9994 * degree_to_radian
	H1_xarm_tilt = -6.195e-4
	H1_yarm_tilt = 1.25e-5
	H1_elevation = 142.554
	
	H1_arm1 = construct_arm(H1_long, H1_lat, H1_xarm_tilt, H1_xarm_azimuth)
	H1_arm2 = construct_arm(H1_long, H1_lat, H1_yarm_tilt, H1_yarm_azimuth)

	H1_vertex = get_vertex_position_geocentric(H1_lat, H1_long, H1_elevation)
	
	return detector_tensor(H1_arm1, H1_arm2), H1_vertex

def get_L1():
	L1_lat = 30 + 33. / 60 + 46.4196 / 3600 * degree_to_radian
	L1_long = -(90 + 46. / 60 + 27.2654 / 3600) * degree_to_radian
	L1_xarm_azimuth = 197.7165 * degree_to_radian
	L1_yarm_azimuth = 287.7165 * degree_to_radian
	L1_xarm_tilt = 0 
	L1_yarm_tilt = 0
	L1_elevation = -6.574
	
	L1_arm1 = construct_arm(L1_long, L1_lat, L1_xarm_tilt, L1_xarm_azimuth)
	L1_arm2 = construct_arm(L1_long, L1_lat, L1_yarm_tilt, L1_yarm_azimuth)

	L1_vertex = get_vertex_position_geocentric(L1_lat, L1_long, L1_elevation)
	
	return detector_tensor(L1_arm1, L1_arm2)
