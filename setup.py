# -*- encoding: utf-8 -*-

from setuptools import setup

setup(
	name="elih",
	version="0.2",
	description="A library to translate Machine Learning classifiers predictions in a human understandable and simplified form.",
	url="https://github.com/fvinas/elih",
	download_url="https://github.com/fvinas/elih/archive/0.2.tar.gz",
	author="Fabien Vinas",
	author_email="fabien.vinas@gmail.com",
	license="",
	packages=["elih"],
	install_requires=[
		'eli5 >= 0.6.1',
		'six',
		'future'
	],
	zip_safe=False
)
