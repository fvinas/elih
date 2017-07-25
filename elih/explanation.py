# -*- encoding: utf-8 -*-

import copy

from jinja2 import Environment, PackageLoader, select_autoescape

from .helpers import _extract_from_dictionary
from .features import group


env = Environment(
	loader=PackageLoader('elih', 'templates'),
	autoescape=select_autoescape(['html'])
)


def translate_explanation(explanation, dictionary):
	'''A simple method that returns an ELI5 Explanation object with features renamed
	following the given dictionary.
	'''
	labels_dictionary = _extract_from_dictionary(dictionary)
	translated_explanation = copy.deepcopy(explanation)
	for feature_weight in translated_explanation.targets[0].feature_weights.pos + translated_explanation.targets[0].feature_weights.neg:
		if feature_weight.feature in labels_dictionary:
			feature_weight.feature = labels_dictionary[feature_weight.feature]

	return translated_explanation


def translate_keys(dict, dictionary):
	'''A simple method to translate dict keys to new labels using a dictionary.
	'''
	new_dict = {}
	labels_dictionary = _extract_from_dictionary(dictionary)
	for k, v in dict.iteritems():
		if k in labels_dictionary:
			new_dict[labels_dictionary[k]] = v
		else:
			new_dict[k] = v
	return new_dict


class HumanExplanation(object):
	'''A layer on top of ELI5 Explanation object to provide additional services
	'''

	def __init__(self, explanation, rules, additional_features=None, dictionary=None):
		if additional_features is None:
			self.additional_features = []
		else:
			self.additional_features = additional_features
		if dictionary is None:
			self.dictionary = {}
		else:
			self.dictionary = dictionary
		self.rules = rules
		self.explanation = group(explanation, rules, additional_features, dictionary)

	def __repr__(self):
		return '{}(explanation={}, additional_features={})'.format(
			'HumanExplanation',
			self.explanation.__repr__(),
			translate_keys(self.additional_features, self.dictionary)
		)

	def _repr_html_(self):
		features = self.explanation.targets[0].feature_weights.pos + self.explanation.targets[0].feature_weights.neg
		template = env.get_template('human_explanation.html')
		return template.render(features=features, additional_features=self.additional_features)

	# TODO: won't work since our features values can be strings as well (ELI5 doesn't handle it)
	# def _repr_html_(self):
	#	return '{}<br/><b>{}</b><pre>{}</pre>'.format(
	#		format_as_html(translate_explanation(self.explanation, self.dictionary), force_weights=False, show=fields.WEIGHTS, show_feature_values=True),
	#		'Additional features',
	#		translate_keys(self.additional_features, self.dictionary)
	#	)
