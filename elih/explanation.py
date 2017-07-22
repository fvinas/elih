# -*- encoding: utf-8 -*-

import copy


def translate_explanation(explanation, dictionary):
	'''A simple method that returns an ELI5 Explanation object with features renamed
	following the given dictionary.
	'''
	translated_explanation = copy.deepcopy(explanation)
	for feature_weight in translated_explanation.targets[0].feature_weights.pos + translated_explanation.targets[0].feature_weights.neg:
		if feature_weight.feature in dictionary:
			feature_weight.feature = dictionary[feature_weight.feature]

	return translated_explanation


def translate_keys(dict, dictionary):
	'''A simple method to map dict keys to new labels using a dictionary.
	'''
	new_dict = {}
	for k, v in dict.iteritems():
		if k in dictionary:
			new_dict[dictionary[k]] = v
		else:
			new_dict[k] = v
	return new_dict


class HumanExplanation(object):
	'''A layer on top of ELI5 Explanation object to provide additional services
	'''

	def __init__(self, explanation, additional_features=None, dictionary=None):
		self.explanation = explanation
		if additional_features is None:
			self.additional_features = []
		else:
			self.additional_features = additional_features
		if dictionary is None:
			self.dictionary = {}
		else:
			self.dictionary = dictionary

	def __repr__(self):
		return '{}(explanation={}, additional_features={})'.format(
			'HumandExplanation',
			translate_explanation(self.explanation, self.dictionary),
			translate_keys(self.additional_features, self.dictionary)
		)

	def _repr_html_(self):
		return '{}<br/><b>{}</b><pre>{}</pre>'.format(
			translate_explanation(self.explanation, self.dictionary)._repr_html_(),
			'Additional features',
			translate_keys(self.additional_features, self.dictionary)
		)
