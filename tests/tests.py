# -*- coding: utf-8 -*-

# xxx update the file as we go along
from .context import sample

import unittest


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_absolute_truth_and_meaning(self):
        assert True


if __name__ == '__main__':
    unittest.main()