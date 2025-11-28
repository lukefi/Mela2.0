import unittest
from lukefi.metsi.sim.operations import simple_processable_chain
from tests.toy_model import parametrized_treatment


class TestOperations(unittest.TestCase):

    def test_simple_processable_chain_multiparameter_exception(self):
        operation_tags = [parametrized_treatment]
        operation_params = {parametrized_treatment: [{'amplify': True}, {'kissa123': 123}]}
        self.assertRaises(Exception,
                          simple_processable_chain,
                          operation_tags,
                          operation_params)
