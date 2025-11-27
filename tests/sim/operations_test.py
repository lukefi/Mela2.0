import unittest
from lukefi.metsi.sim.operations import prepared_treatment, simple_processable_chain
from tests.toy_model import ToyModel, parametrized_treatment


class TestOperations(unittest.TestCase):

    def test_simple_processable_chain_multiparameter_exception(self):
        operation_tags = [parametrized_treatment]
        operation_params = {parametrized_treatment: [{'amplify': True}, {'kissa123': 123}]}
        self.assertRaises(Exception,
                          simple_processable_chain,
                          operation_tags,
                          operation_params)

    def test_prepared_treatment(self):
        assertions = [
            ([ToyModel("", 10), {}], 10),
            ([ToyModel("", 10), {'amplify': True}], 10000),
            ([ToyModel("", 10), {'amplify': False}], 10)
        ]

        for case in assertions:
            function = prepared_treatment(parametrized_treatment, **case[0][1])
            result = function(case[0][0])
            self.assertEqual(case[1], result[0].value)
