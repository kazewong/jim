import os
from jimgw.run.single_event_run_definition import TestSingleEventRun
from jimgw.run.single_event_run_manager import SingleEventRunManager
from jimgw.core.single_event.detector import H1, L1


class TestTestSingleEventRun:

    def test_serialize_deserialize(self, tmp_path):
        # Create a test run
        run = TestSingleEventRun()
        config_path = os.path.join(tmp_path, "test_config.yaml")

        # Test serialization
        run.serialize(config_path)
        assert os.path.exists(config_path)

        # Test deserialization
        loaded_run = TestSingleEventRun.deserialize(config_path)

        # Verify properties match
        assert loaded_run.gps == run.gps
        assert loaded_run.segment_length == run.segment_length
        assert loaded_run.post_trigger_length == run.post_trigger_length
        assert loaded_run.f_min == run.f_min
        assert loaded_run.f_max == run.f_max
        assert [ifo.name for ifo in loaded_run.ifos] == [ifo.name for ifo in run.ifos]
        assert loaded_run.f_ref == run.f_ref

    def test_run_manager(self, tmp_path):
        # Create test run and run manager
        run = TestSingleEventRun()
        run_manager = SingleEventRunManager(run)

        # Test basic parameters dictionary
        params = {
            "mass1": 50.0,
            "mass2": 30.0,
        }

        # Test getting detector waveform
        waveform, hp, hc = run_manager.get_detector_waveform(params)

        # Basic checks
        assert len(hp) == 2  # Should have H1 and L1
        assert len(hc) == 2
        assert H1.name in hp and L1.name in hp
        assert H1.name in hc and L1.name in hc
