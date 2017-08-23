# -*- encoding: utf-8 -*-

import copy
import fnmatch

from six import iteritems
from past.builtins import basestring
from eli5.base import FeatureWeights, FeatureWeight

from .helpers import _extract_mapped_value, _extract_formatted_value


class EnrichedFeatureWeight(FeatureWeight):
	"""Enriches ELI5 FeatureWeight with additional 'human explanation' data like
	a human readable feature label, description, format, ...
	"""

	def __init__(self, *args, **kwargs):
		self.dictionary = kwargs['dictionary'] if 'dictionary' in kwargs else None
		self.formatted_value = kwargs['formatted_value'] if 'formatted_value' in kwargs else None
		self.score = kwargs['score'] if 'score' in kwargs else None
		if 'dictionary' in kwargs:
			del kwargs['dictionary']
		if 'formatted_value' in kwargs:
			del kwargs['formatted_value']
		if 'score' in kwargs:
			del kwargs['score']
		FeatureWeight.__init__(self, *args, **kwargs)

	def __repr__(self):
		return "{}(feature='{}', weight={}, score={}, std={}, value={}, formatted_value={}, dictionary={})".format(
			'EnrichedFeatureWeight',
			self.feature,
			self.weight,
			self.score,
			self.std,
			self.value,
			self.formatted_value,
			self.dictionary
		)

	def to_dict(self):
		return {
			"feature": self.feature,
			"weight": self.weight,
			"score": self.score,
			"std": self.std,
			"value": self.value,
			"formatted_value": self.formatted_value,
			"label": self.dictionary["label"] if self.dictionary is not None and 'label' in self.dictionary else None
		}


class FeatureWeightGroup(EnrichedFeatureWeight):
	"""A class derived from ELI5's FeatureWeight to store the initial FeatureWeight-s
	regrouped to build up this new FeatureWeight.
	"""

	def __init__(self, *args, **kwargs):
		self.group = kwargs['group'] if 'group' in kwargs else []
		if 'group' in kwargs:
			del kwargs['group']
		EnrichedFeatureWeight.__init__(self, *args, **kwargs)

	def __repr__(self):
		return "{}(feature='{}', weight={}, score={}, std={}, value={}, formatted_value={}, dictionary={}, group={})".format(
			'FeatureWeightGroup',
			self.feature,
			self.weight,
			self.score,
			self.std,
			self.value,
			self.formatted_value,
			self.dictionary,
			self.group
		)

	def to_dict(self):
		return {
			"feature": self.feature,
			"weight": self.weight,
			"score": self.score,
			"std": self.std,
			"value": self.value,
			"formatted_value": self.formatted_value,
			"label": self.dictionary["label"] if self.dictionary is not None and 'label' in self.dictionary else None,
			"group": [f.to_dict() for f in self.group]
		}


def apply_rules_layer(explanation, rules, additional_features=None, dictionary=None, scoring=None):
	"""Regroups several feature weights into one brand new feature weight, whose weight is the sum of the ones from the underlying feature weights.

	The new feature weight will be created inside the FeatureWeights object of the explanation, while the previous weights are discarded.

	Now, it only works with binary classifiers (meaning on the first TargetExplanation object)

	TODO: add support for multiple targets.

	Args:
		explanation: An ELI5 Explanation object (typically the output of explain_prediction).
		rules: a dictionary of rules whose keys are new grouped feature names and whose values are either:
			- lists of exact feature names that will be regrouped
			- string for pattern matching in a "Unix-file" flavour (e.g. 'Embarked=*', 'Class*')
		additional_features: (optional) a dictionary with additional variables that may be used to map values
		dictionary: (optional) a dictionary that allows mapping values and labels to features

	Returns:
		An Explanation object with the new grouped features

	"""

	# Otherwise the object is modified by reference
	explanation = copy.deepcopy(explanation)

	if additional_features is None:
		additional_features = {}
	if dictionary is None:
		dictionary = {}

	# Start by getting out the generic rules (e.g. 'Variable=*') cause they are processed in a different way:
	generic_rules = {
		v: k for (k, v) in rules.items() if isinstance(v, basestring)
	}

	# Then reverse the other rules:
	# {'A': ['1', '2'], 'B': ['3']} to {'1': 'A', '2': 'A', '3': 'B'}
	new_rules = {old_field: grouped_field for (grouped_field, old_fields) in rules.items() if not isinstance(old_fields, basestring) for old_field in old_fields}

	new_weights = {}
	for feature_weight in explanation.targets[0].feature_weights.pos + explanation.targets[0].feature_weights.neg:

		if feature_weight.feature == '<BIAS>':
			continue

		matched = False

		# 'Old feature' matches a standard rule?
		if feature_weight.feature in new_rules:
			matched = True
			# Feature is grouped with others
			grouped_feature = new_rules[feature_weight.feature]
			if grouped_feature in new_weights:
				# Add weight to already created grouped feature
				new_weights[grouped_feature]['weight'] = new_weights[grouped_feature]['weight'] + feature_weight.weight
			else:
				# Create new grouped feature
				_mapped_value = _extract_mapped_value(additional_features, dictionary, grouped_feature)
				new_weights[grouped_feature] = {
					'feature': grouped_feature,
					'weight': feature_weight.weight,
					'score': scoring(feature_weight.weight) if scoring is not None else None,
					'std': None,
					'value': _mapped_value,
					'formatted_value': _extract_formatted_value(_mapped_value, dictionary, grouped_feature),
					'group': [],
					'dictionary': dictionary[grouped_feature] if grouped_feature in dictionary else {}
				}
			new_weights[grouped_feature]['group'].append({
				'feature': feature_weight.feature,
				'weight': feature_weight.weight,
				'score': scoring(feature_weight.weight) if scoring is not None else None,
				'std': feature_weight.std,
				'value': feature_weight.value,
				'formatted_value': _extract_formatted_value(feature_weight.value, dictionary, feature_weight.feature),
				'dictionary': dictionary[feature_weight.feature] if feature_weight.feature in dictionary else {}
			})

		# Match generic rule?
		for rule, grouped_feature in iteritems(generic_rules):
			if fnmatch.fnmatch(feature_weight.feature, rule):
				matched = True
				if grouped_feature in new_weights:
					# Add weight to already created grouped feature
					new_weights[grouped_feature]['weight'] = new_weights[grouped_feature]['weight'] + feature_weight.weight
				else:
					# Create new grouped feature
					_mapped_value = _extract_mapped_value(additional_features, dictionary, grouped_feature)
					new_weights[grouped_feature] = {
						'feature': grouped_feature,
						'weight': feature_weight.weight,
						'score': scoring(feature_weight.weight) if scoring is not None else None,
						'std': None,
						'value': _mapped_value,
						'formatted_value': _extract_formatted_value(_mapped_value, dictionary, grouped_feature),
						'group': [],
						'dictionary': dictionary[grouped_feature] if grouped_feature in dictionary else {}
					}
				new_weights[grouped_feature]['group'].append({
					'feature': feature_weight.feature,
					'weight': feature_weight.weight,
					'score': scoring(feature_weight.weight) if scoring is not None else None,
					'std': feature_weight.std,
					'value': feature_weight.value,
					'formatted_value': _extract_formatted_value(feature_weight.value, dictionary, feature_weight.feature),
					'dictionary': dictionary[feature_weight.feature] if feature_weight.feature in dictionary else {}
				})

		# No match for this feature => remains the same
		if not matched:
			# Feature remains the same ('not a group')
			new_weights[feature_weight.feature] = {
				'feature': feature_weight.feature,
				'weight': feature_weight.weight,
				'score': scoring(feature_weight.weight) if scoring is not None else None,
				'std': feature_weight.std,
				'value': feature_weight.value,
				'formatted_value': _extract_formatted_value(feature_weight.value, dictionary, feature_weight.feature),
				'dictionary': dictionary[feature_weight.feature] if feature_weight.feature in dictionary else {}
			}

	new_features = []
	for new_feature, new_feature_weight in iteritems(new_weights):
		if 'group' in new_feature_weight:
			obj = FeatureWeightGroup(
				feature=new_feature_weight['feature'],
				weight=new_feature_weight['weight'],
				score=new_feature_weight['score'],
				std=new_feature_weight['std'],
				value=new_feature_weight['value'],
				formatted_value=new_feature_weight['formatted_value'],
				group=[
					EnrichedFeatureWeight(
						feature=f['feature'],
						weight=f['weight'],
						score=f['score'],
						std=f['std'],
						value=f['value'],
						formatted_value=f['formatted_value'],
						dictionary=f['dictionary']
					) for f in new_feature_weight['group']
				],
				dictionary=new_feature_weight['dictionary']
			)
		else:
			obj = EnrichedFeatureWeight(
				feature=new_feature_weight['feature'],
				weight=new_feature_weight['weight'],
				score=new_feature_weight['score'],
				std=new_feature_weight['std'],
				value=new_feature_weight['value'],
				formatted_value=new_feature_weight['formatted_value'],
				dictionary=new_feature_weight['dictionary']
			)
		new_features.append(obj)

	# To build a FeatureWeights object, we then need to sort features by weight and separate positives and negatives:
	feature_weights = FeatureWeights(
		pos=sorted([f for f in new_features if f.weight >= 0], key=(lambda o: abs(o.weight)), reverse=True),
		neg=sorted([f for f in new_features if f.weight < 0], key=(lambda o: abs(o.weight)), reverse=False)
	)

	explanation.targets[0].feature_weights = feature_weights

	return explanation
