# -*- encoding: utf-8 -*-


def mapper(dictionary):
	"""A formatter that map a value to another given a dict.
	Useful to quickly map categorical variables with very few levels (e.g a "sex" variable)
	to something more understandable (or translated).
	"""
	return (lambda a: dictionary[a] if a in dictionary else a)


def percent(decimals=1):
	"""A formatter to display a float as a percentage with a given number of decimals.

	Args:
		decimals: The number of decimals to display.

	Returns:
		A formatter function f returning f(a), being variable 'a' displayed as a percentage.
		Please note the formatter function will multiply the float by 100 to get a percentage.

	"""
	return (lambda a: "{:+.{prec}f}%".format(100.0 * a, prec=decimals))


def delta_percent(decimals=1):
	"""A formatter to display the float delta as a percentage with a given number of decimals.
	It's different from the percent formatter in that it's prefixed with a sign (+ or -).

	Args:
		decimals: The number of decimals to display.

	Returns:
		A formatter function f returning f(a), being variable 'a' displayed as a signed percentage.
		Please note the formatter function will multiply the float by 100 to get a percentage.

	"""
	return (lambda a: "{:+.{prec}f}%".format(100.0 * a, prec=decimals))


def value(decimals=1, unit="", sign=""):
	"""A formatter to display a float with a given unit.

	Args:
		decimals = 1: The number of decimals to display.
		unit = "": The unit symbol to display after the delta value.
		sign = "": Allows to move to a signed version ('+5.3M€' vs '5.3M€'). Use sign='+' for this.

	Returns:
		A formatter function f returning f(a), being variable 'a' accompanied with a unit.

	"""
	return (lambda a: "{:{sign},.{prec}f} {unit}".format(1.0 * a, sign=sign, unit=unit, prec=decimals))


def text():
	"""A simple formatter to display a text as it is.

	Returns:
		A formatter function f returning f(a) being variable 'a' displayed as a text.
	"""
	return (lambda a: "{}".format(a))


def integer():
	"""A simple formatter to display an integer with no unit.

	Returns:
		A formatter function f returning f(a) being variable 'a' displayed as an integer.
	"""
	return value(decimals=0, unit="")


def value_simplified(decimals=1, unit="", prefixes=['k', 'M', 'B'], sign=""):
	"""A formatter to display the given float with a given unit in a simplified way (e.g. 10k for 10.000)

	Args:
		decimals = 1: The number of decimals to display.
		unit = "": The unit symbol to concatenate to the value.
		prefixes = ['k', 'M', 'B']: The simplifier symbols to use for 1.000s, 1.000.000s and 1.000.000.000s.
		sign = "": Allows to move to a signed version ('+5.3M€' vs '5.3M€'). Use sign='+' for this.

	Returns:
		A formatter function f returning f(a), being variable 'a' simplified and accompanied with a unit.

	"""
	def formatter(a):
		prefix = ""
		if a >= 1000000000:
			a = a * 1.0 / 1000000000
			prefix = prefixes[2]
		elif a >= 1000000:
			a = a * 1.0 / 1000000
			prefix = prefixes[1]
		elif a >= 1000:
			a = a * 1.0 / 1000
			prefix = prefixes[0]
		return "{:{sign},.{decimals}f} {prefix}{unit}".format(1.0 * a, sign=sign, decimals=decimals, prefix=prefix, unit=unit)
	return formatter
