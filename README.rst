=====
eyepy
=====

.. image:: https://badge.fury.io/py/eyepie.svg
    :target: https://badge.fury.io/py/eyepie

.. image:: https://travis-ci.com/MedVisBonn/eyepy.svg?branch=master
    :target: https://travis-ci.com/MedVisBonn/eyepy

.. image:: https://readthedocs.org/projects/eyepy/badge/?version=latest
        :target: https://eyepy.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


This software is under active development and things might change without
backwards compatibility. If you want to use eyepy in your project make sure to
pin the version in your requirements file.


Features
--------

* Read the HEYEX XML export
* Read the HEYEX VOL export
* Read B-Scans from a folder
* Read the public OCT Dataset from Duke University
* Plot OCT Scans
* Compute Drusen from BM and RPE segmentations


Getting started
---------------

Installation
^^^^^^^^^^^^
Install eyepy with :code:`pip install -U eyepie`. Yes it is :code:`eyepie` and not :code:`eyepy` for
installation with pip.

Loading Data
^^^^^^^^^^^^

.. code-block:: python

   import eyepy as ep

   # Load B-Scans from folder
   data = ep.Oct.from_folder("path/to/folder")

   # Load an OCT volume from the DUKE dataset
   data = ep.Oct.from_duke_mat("path/to/file.mat")

   # Load an HEYEX XML export
   data = ep.Oct.from_heyex_xml("path/to/folder")

   # Load an HEYEX VOL export
   data = ep.Oct.from_heyex_vol("path/to/file.vol")

The Oct object
^^^^^^^^^^^^^^

When loading data as described above an Oct object is returned. You can use
this object to perform common actions on the OCT volume such as:

+ Iterating over the volume to retrieve Bscan objects :code:`for bscan in data`
+ Plotting a localizer (NIR) image associated to the OCT :code:`data.plot(localizer=True)`
+ Accessing an associated localizer image :code:`data.localizer`
+ Reading Meta information from the loaded data if available :code:`data.ScaleXSlo`



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
