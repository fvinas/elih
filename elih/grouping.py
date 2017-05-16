# -*- encoding: utf-8 -*-

import numpy as np

from eli5 import explain_prediction
from eli5 import formatters
from eli5.base import Explanation, TargetExplanation, FeatureWeights, FeatureWeight



def group(explanation, rules):
	"""Regroups several feature weights into one brand new feature weight, whose weight is the sum of the ones from the underlying feature weights.

	The new feature weight will be created inside the FeatureWeights object of the explanation, while the previous weights are discarded.

	Now, it only works with binary classifiers (meaning on the first TargetExplanation object)

	TODO: add support for multiple targets.

	Args:
	    explanation: An ELI5 Explanation object (typically the output of explain_prediction).
	    rules: a dictionary whose keys are new grouped feature names and whose values are lists of feature names that will be regrouped

	Returns:
	    An Explanation object with the new grouped features

	"""

	# Start by reversing the rules:
	# {'A': ['1', '2'], 'B': ['3']} to {'1': 'A', '2': 'A', '3': 'B'}
	new_rules = {old_field: grouped_field for (grouped_field, old_fields) in rules.iteritems() for old_field in old_fields}

	new_weights = {}
	for feature_weight in explanation.targets[0].feature_weights.pos + explanation.targets[0].feature_weights.neg:
		if feature_weight.feature in new_rules:
			# Feature is grouped with others
			grouped_feature = new_rules[feature_weight.feature]
			if grouped_feature in new_weights:
				# Add weight to already created grouped feature
				new_weights[grouped_feature]['weight'] = new_weights[grouped_feature]['weight'] + feature_weight.weight
			else:
				# Create new grouped feature
				new_weights[grouped_feature] = {
					'weight': feature_weight.weight,
					'std': None,
					'value': np.nan
				}
		else:
			# Feature remains the same
			new_weights[feature_weight.feature] = {
				'weight': feature_weight.weight ,
				'std': None,
				'value': feature_weight.value
			}

	new_features = []
	for new_feature, new_feature_weight in new_weights.iteritems():
		obj = FeatureWeight(feature=new_feature, weight=new_feature_weight['weight'], std=new_feature_weight['std'], value=new_feature_weight['value'])
		new_features.append(obj)

	# To build a FeatureWeights object, we then need to sort features by weight and separate positives and negatives:	
	feature_weights = FeatureWeights(
		pos=sorted([f for f in new_features if f.weight >= 0], key=(lambda o: abs(o.weight)), reverse=True),
		neg=sorted([f for f in new_features if f.weight < 0], key=(lambda o: abs(o.weight)), reverse=True)
	)

	explanation.targets[0].feature_weights = feature_weights

	return explanation
