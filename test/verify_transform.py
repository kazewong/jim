import numpy as np

class VerifyTransform:
    def verify_sky_location_transform(self):
        from bilby.gw.utils import zenith_azimuth_to_ra_dec as bilby_earth_to_sky
        from bilby.gw.detector.networks import InterferometerList
        
        from jimgw.single_event.utils import zenith_azimuth_to_ra_dec as jimgw_earth_to_sky
        from jimgw.single_event.detector import detector_preset
        from astropy.time import Time

        ifos = ["H1", "L1"]
        geocent_time = 1000000000

        import matplotlib.pyplot as plt

        for zenith in np.linspace(0, np.pi, 5):
            for azimuth in np.linspace(0, 2*np.pi, 5):
                bilby_sky_location = np.array(bilby_earth_to_sky(zenith, azimuth, geocent_time, InterferometerList(ifos)))
                jimgw_sky_location = np.array(jimgw_earth_to_sky(zenith, azimuth, Time(geocent_time, format="gps").sidereal_time("apparent", "greenwich").rad, detector_preset[ifos[0]].vertex - detector_preset[ifos[1]].vertex))
                assert np.allclose(bilby_sky_location, jimgw_sky_location, atol=1e-4)
