import unittest

class my_test(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()