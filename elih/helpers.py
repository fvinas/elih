# -*- encoding: utf-8 -*-

from six import iteritems
from past.builtins import basestring


def _extract_from_dictionary(dictionary, field='label'):
	'''An ELIH dictionary can embed, for each variable, either just a 'human label' or more than that
	(description, format, etc.). This helper extracts the labels dictionary from it.
	'''
	extracted_labels = {}
	for k, v in iteritems(dictionary):
		if isinstance(v, basestring):
			extracted_labels[k] = v
		elif type(v) is dict and field in v:
			extracted_labels[k] = v[field]
	return extracted_labels


def _extract_mapped_value(additional_features, dictionary, grouped_feature):
	value_from = None
	if grouped_feature in dictionary and 'value_from' in dictionary[grouped_feature]:
		try:
			value_from = additional_features[dictionary[grouped_feature]['value_from']]
		except IndexError:
			raise IndexError("Variable {} required but not provided with the additional features.".format(dictionary[grouped_feature]['value_from']))
	return value_from


def _extract_label(dictionary, feature_name):
	label = None
	if feature_name in dictionary and 'label' in dictionary[feature_name]:
		try:
			label = dictionary[feature_name]['label']
		except:
			raise Exception("Exception when generating the label for variable {}.".format(feature_name))
	return label


def _extract_formatted_value(value, dictionary, feature_name):
	formatted_value = None
	if value is None:
		return None
	if feature_name in dictionary and 'formatter' in dictionary[feature_name]:
		try:
			formatted_value = dictionary[feature_name]['formatter'](value)
		except:
			raise Exception("Exception when generating the formatted value for variable {} with value {}.".format(feature_name, value))
	return formatted_value
