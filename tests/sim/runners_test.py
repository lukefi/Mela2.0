import unittest
from pathlib import Path
from lukefi.metsi.sim.simulation_payload import SimulationPayload
from lukefi.metsi.sim.runners import evaluate_sequence, run_full_tree_strategy
from lukefi.metsi.sim.sim_configuration import SimConfiguration
from lukefi.metsi.app.file_io import read_control_module
from tests.test_utils import collect_results, raises, identity, none


class RunnersTest(unittest.TestCase):
    def test_sequence_success(self):
        payload = SimulationPayload(computational_unit=1)
        result = evaluate_sequence(
            payload,
            identity,
            none
        )
        self.assertEqual(None, result)

    def test_sequence_failure(self):
        payload = SimulationPayload(computational_unit=1)

        def prepared_function():
            return evaluate_sequence(payload, identity, raises, identity)

        self.assertRaises(Exception, prepared_function)

    def test_full_formation_evaluation_strategies_by_comparison(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "branching.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        print(config)
        depth_payload = SimulationPayload(
            computational_unit=1,
            operation_history=[]
        )
        results_depth = collect_results(run_full_tree_strategy(depth_payload, config))
        self.assertEqual(8, len(results_depth))

    def test_no_parameters_propagation(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "no_parameters.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        # print(config)
        initial = SimulationPayload(
            computational_unit=1,
            operation_history=[]
        )

        results = collect_results(run_full_tree_strategy(initial, config))
        self.assertEqual(5, results[0])

    def test_parameters_propagation(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "parameters.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        # print(config)
        initial = SimulationPayload(
            computational_unit=1,
            operation_history=[]
        )

        results = collect_results(run_full_tree_strategy(initial, config))
        self.assertEqual(9, results[0])

    def test_parameters_branching(self):
        control_path = str(Path("tests",
                                "resources",
                                "runners_test",
                                "parameters_branching.py").resolve())
        declaration = read_control_module(control_path)
        config = SimConfiguration(**declaration)
        initial = SimulationPayload(
            computational_unit=1,
            operation_history=[]
        )

        results = collect_results(run_full_tree_strategy(initial, config))
        # do_nothing, do_nothing = 1
        # do_nothing, inc#1      = 2
        # do_nothing, inc#2      = 3
        # inc#1, do_nothing      = 2
        # inc#1, inc#1           = 3
        # inc#1, inc#2           = 4
        # inc#2, do_nothing      = 3
        # inc#2, inc#1           = 4
        # inc#2, inc#2           = 5
        expected = [1, 2, 3, 2, 3, 4, 3, 4, 5]
        self.assertEqual(expected, results)
