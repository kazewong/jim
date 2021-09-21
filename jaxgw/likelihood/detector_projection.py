import jax.numpy as jnp

##########################################################
# Construction of detector tensor
##########################################################

def construct_arm(longitude, latitude, arm_tilt, arm_azimuth):
    e_long = jnp.array([-jnp.sin(longitude), jnp.cos(longitude), 0])
    e_lat = jnp.array([-jnp.sin(latitude) * jnp.cos(longitude),
                      -jnp.sin(latitude) * jnp.sin(longitude), jnp.cos(latitude)])
    e_h = jnp.array([jnp.cos(latitude) * jnp.cos(longitude),
                    jnp.cos(latitude) * jnp.sin(longitude), jnp.sin(latitude)])

    return (jnp.cos(arm_tilt) * jnp.cos(arm_azimuth) * e_long +
            jnp.cos(arm_tilt) * jnp.sin(arm_azimuth) * e_lat +
            jnp.sin(arm_tilt) * e_h)


def detector_tensor(arm1, arm2):
    return 0.5 * (jnp.einsum('i,j->ij', arm1, arm1) - jnp.einsum('i,j->ij', arm2, arm2))

##########################################################
# Construction of detector tensor
##########################################################

def get_polarization_tensor(ra, dec, time, psi, mode):

    #gmst = fmod(greenwich_mean_sidereal_time(time), 2 * jnp.pi)
    phi = ra #- gmst
    theta = jnp.pi / 2 - dec

    u = jnp.array([jnp.cos(phi) * jnp.cos(theta), jnp.cos(theta) * jnp.sin(phi), -jnp.sin(theta)])
    v = jnp.array([-jnp.sin(phi), jnp.cos(phi), 0])
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)

    if mode.lower() == 'plus':
        return jnp.einsum('i,j->ij', m, m) - jnp.einsum('i,j->ij', n, n)
    elif mode.lower() == 'cross':
        return jnp.einsum('i,j->ij', m, n) + jnp.einsum('i,j->ij', n, m)
    elif mode.lower() == 'breathing':
        return jnp.einsum('i,j->ij', m, m) + jnp.einsum('i,j->ij', n, n)

    # Calculating omega here to avoid calculation when model in [plus, cross, breathing]
    omega = jnp.cross(m, n)
    if mode.lower() == 'longitudinal':
        return jnp.einsum('i,j->ij', omega, omega)
    elif mode.lower() == 'x':
        return jnp.einsum('i,j->ij', m, omega) + jnp.einsum('i,j->ij', omega, m)
    elif mode.lower() == 'y':
        return jnp.einsum('i,j->ij', n, omega) + jnp.einsum('i,j->ij', omega, n)
    else:
        raise ValueError("{} not a polarization mode!".format(mode))

def antenna_response(detector_tensor, ra, dec, time, psi, mode):
    polarization_tensor = gwutils.get_polarization_tensor(ra, dec, time, psi, mode)
    return jnp.einsum('ij,ij->', detector_tensor, polarization_tensor)

def get_detector_response(self, waveform_polarizations, parameters):
    signal = {}
    for mode in waveform_polarizations.keys():
        det_response = self.antenna_response(
            parameters['ra'],
            parameters['dec'],
            parameters['geocent_time'],
            parameters['psi'], mode)

        signal[mode] = waveform_polarizations[mode] * det_response
    signal_ifo = sum(signal.values())

    signal_ifo *= self.strain_data.frequency_mask

    time_shift = self.time_delay_from_geocenter(
        parameters['ra'], parameters['dec'], parameters['geocent_time'])

    # Be careful to first subtract the two GPS times which are ~1e9 sec.
    # And then add the time_shift which varies at ~1e-5 sec
    dt_geocent = parameters['geocent_time'] - self.strain_data.start_time
    dt = dt_geocent + time_shift

    signal_ifo[self.strain_data.frequency_mask] = signal_ifo[self.strain_data.frequency_mask] * jnp.exp(
        -1j * 2 * jnp.pi * dt * self.strain_data.frequency_array[self.strain_data.frequency_mask])

    signal_ifo[self.strain_data.frequency_mask] *= self.calibration_model.get_calibration_factor(
        self.strain_data.frequency_array[self.strain_data.frequency_mask],
        prefix='recalib_{}_'.format(self.name), **parameters)

    return signal_ifo


