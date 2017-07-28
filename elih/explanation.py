# -*- encoding: utf-8 -*-

import copy

from jinja2 import Environment, PackageLoader, select_autoescape
from eli5.formatters.html import format_hsl, weight_color_hsl

from .helpers import (
	_extract_from_dictionary,
	_extract_formatted_value,
	_extract_label
)
from .features import apply_rules_layer
from .helpers import format_weight


env = Environment(
	loader=PackageLoader('elih', 'templates'),
	autoescape=select_autoescape(['html'])
)

env.filters.update(dict(
	weight_color=lambda w, w_range: format_hsl(weight_color_hsl(w, w_range)),
	format_weight=format_weight
))


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

	def __init__(self, explanation, rules_layers, additional_features=None, dictionary=None):
		if type(rules_layers) is dict:
			# Only one layer of rules was provided
			rules_layers = [rules_layers]
		if additional_features is None:
			self.additional_features = []
		else:
			self.additional_features = self._translate_additional_features(additional_features, dictionary)

		if dictionary is None:
			self.dictionary = {}
		else:
			self.dictionary = dictionary

		self.rules_layers = rules_layers
		self.explanation_layers = []
		for index, rules in enumerate(rules_layers):
			if index > 0:
				previous_explanation = self.explanation_layers[index - 1]
			else:
				previous_explanation = explanation
			new_explanation = apply_rules_layer(previous_explanation, rules, additional_features, dictionary)
			self.explanation_layers.append(new_explanation)

	def _translate_additional_features(self, additional_features, dictionary):
		new_dict = {}
		for feature_name, value in additional_features.iteritems():
			feature_dict = {}
			# value is not supposed to change because it's coming directly from the source (so, can be controlled)
			feature_dict['value'] = value
			# formatted_value takes the value and formats it
			feature_dict['formatted_value'] = _extract_formatted_value(value, dictionary, feature_name)
			feature_dict['label'] = _extract_label(dictionary, feature_name)

			new_dict[feature_name] = feature_dict
		return new_dict

	def __repr__(self):
		all_repr = ''
		for layer in self.explanation_layers:
			all_repr += '{}(explanation={}, additional_features={})'.format(
				'HumanExplanation',
				layer.__repr__(),
				translate_keys(self.additional_features, self.dictionary)
			)
		return all_repr

	def _repr_html_(self):
		template = env.get_template('human_explanation.html')
		layers = []
		for index, layer in enumerate(self.explanation_layers):
			features = layer.targets[0].feature_weights.pos + layer.targets[0].feature_weights.neg
			weight_range = abs(max([f.weight for f in features]))
			layers.append({
				"features": features,
				"weight_range": weight_range
			})
		return template.render(
			layers=layers,
			additional_features=self.additional_features
		)
