"""
Unit Tests (BME 355 Individual Assignment 3)

@author: Katie C.
"""
import unittest
from circulation import *

class TestCirculation(unittest.TestCase):
	def test_isovolumic(self):
		HR = 75
		Emax = 2
		Emin = 0.06

		model = Circulation(HR, Emax, Emin)
		self.assertAlmostEqual(model._get_normalized_time(0), 0)
		self.assertAlmostEqual(model._get_normalized_time(1), 0.625)
		self.assertAlmostEqual(model._get_normalized_time(2), 1.25)
		self.assertAlmostEqual(model._get_normalized_time(3), 1.875)
		self.assertAlmostEqual(model._get_normalized_time(4), 2.5)
		self.assertAlmostEqual(model._get_normalized_time(5), 0.625)