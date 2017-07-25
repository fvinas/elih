# -*- encoding: utf-8 -*-


def _extract_from_dictionary(dictionary, field='label'):
	'''An ELIH dictionary can embed, for each variable, either just a 'human label' or more than that
	(description, format, etc.). This helper extracts the labels dictionary from it.
	'''
	extracted_labels = {}
	for k, v in dictionary.iteritems():
		if isinstance(v, basestring):
			extracted_labels[k] = v
		elif type(v) is dict and field in v:
			extracted_labels[k] = v[field]
	return extracted_labels
