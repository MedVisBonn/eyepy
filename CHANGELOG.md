# Changelog

<!--next-version-placeholder-->

## v0.11.1 (2023-03-31)
### Fix
* **EyeVolumeMeta:** Fixes saving issue by storing all dates as string in isoformat to avoid problems with dumping to json and loading ([`3fc7424`](https://github.com/MedVisBonn/eyepy/commit/3fc742485667b03505fd17d3bf25b8d2f20516de))

## v0.11.0 (2023-03-30)
### Feature
* **io.__init__.py:** Adds support for topcon fda files ([#12](https://github.com/MedVisBonn/eyepy/issues/12)) ([`53f2908`](https://github.com/MedVisBonn/eyepy/commit/53f2908f95556c2be7a3259bfd5653208188ff71))

## v0.10.0 (2023-03-30)
### Feature
* **core.plotting:** Adds a scale bar and a watermark to Bscan and Fundus visualizations ([`0e3eaa0`](https://github.com/MedVisBonn/eyepy/commit/0e3eaa0be4420c0d8ba4cbc53381fc36da2cdc81))

### Fix
* **eyepy.core:** Fixes bscan_region and bscan_position plotting when plotting only a part of the fundus ([`2c88074`](https://github.com/MedVisBonn/eyepy/commit/2c880743b6329cf24e74f7019035ad455800e89a))
* **core.grids.py:** Fixes error related to python 3.8 not supporting new type annotation Dict-dict List-list ([`3c6ba41`](https://github.com/MedVisBonn/eyepy/commit/3c6ba41133215bc64efb48f432cb0be78b627b2c))

### Documentation
* **core.annotations:** Adds docstrings ([`236c074`](https://github.com/MedVisBonn/eyepy/commit/236c074e8e28eb4ccb5a1e0538237fcec5dff415))

## v0.9.2 (2023-03-17)
### Fix
* **e2e_reader.py:** Makes exception formatting compatible with python < 3.10 ([`2c4cbb0`](https://github.com/MedVisBonn/eyepy/commit/2c4cbb0c2a91c47d951945aa75145dd1e34e5c4e))

### Documentation
* **eyepy.core:** Adds type annotations to all objects in the core subpackage ([`ea1cb4c`](https://github.com/MedVisBonn/eyepy/commit/ea1cb4c415604a97d1e7ef64998b1743e7c5fe75))
* **annotations.py:** Adds type annotations to all objects in this module ([`986f9bc`](https://github.com/MedVisBonn/eyepy/commit/986f9bc6347227311725d31726ea38741e582d75))

## v0.9.1 (2023-03-15)
### Fix
* **e2e_reader.py:** Extract number of Bscans more reliably by using the number of slice substructures; Skip localizer affine transformation for now, because slodata is not always available; support reading single B-scan data ([`1287a24`](https://github.com/MedVisBonn/eyepy/commit/1287a24d40907085d33e867b35f458078faa6b75))
* **e2e_format.py:** Change the name of E2EFile to E2EFormat to avoid confustion with E2EFileStructure in e2e_reader.py ([`01be6a0`](https://github.com/MedVisBonn/eyepy/commit/01be6a0a6ce4e80491871800a970b7f75eb8c86d))

### Documentation
* **README.md:** Clarify that the localizer is a fundus image ([`4e41acd`](https://github.com/MedVisBonn/eyepy/commit/4e41acd3660f3a07abed11464bdcacba81bda984))

## v0.9.0 (2023-03-09)
### Feature
* **HeE2EReader:** Switch to construct_typed for describing structures; create file hierarchy when parsing the E2E file; add search functions to the HeE2eReader ([`45578a5`](https://github.com/MedVisBonn/eyepy/commit/45578a5a9949a2722a03c1fe4bdef228c3a980c0))
* **eyepy.io.utils.py:** Add functions to search for integer/float values in binary data; set relative tolerance for Bscan distance to 4% (fixes sample data warning) ([`96fd58b`](https://github.com/MedVisBonn/eyepy/commit/96fd58bf1146b6f9b60d63cdb8abbbc8704f6bde))

### Fix
* **HeE2eReader:** Fix issues with inspect after renaming classes ([`891c79c`](https://github.com/MedVisBonn/eyepy/commit/891c79ceb2f0417c5b1b003fa1332ce1d4572b1b))
* **__init__.py:** Exclude __init__.py from isort to prevent circular import ([`9254231`](https://github.com/MedVisBonn/eyepy/commit/92542318b23098323855abf410c5c294ff956466))
* **pyproject.toml:** Add imageio as dependency and umpgrade imagecodecs to latest version ([`bef44a0`](https://github.com/MedVisBonn/eyepy/commit/bef44a0cca752e280f5a405a9b9c4e4d0590f292))

### Breaking
*  ([`45578a5`](https://github.com/MedVisBonn/eyepy/commit/45578a5a9949a2722a03c1fe4bdef228c3a980c0))

### Documentation
* **documentation:** Improve documentation; rename Voxel/Area Annotation to PixelAnnotation for consistency ([`ab38837`](https://github.com/MedVisBonn/eyepy/commit/ab388378522f4be7f1fc55725a101a6483c1893b))

## v0.8.1 (2023-02-22)
### Fix
* **pyproject.toml:** Increase allowed version range for numpy (fixes #10) ([`c66f6f6`](https://github.com/MedVisBonn/eyepy/commit/c66f6f65909aef7b086b62f66d6ce55f0b04220e))

## v0.8.0 (2023-02-13)
### Feature
* **eyepy.core:** Reflect plotted region in x and y axis for both localizer and B-scan; check if bscan_region bscan position indicators are in plotting region ([`2842424`](https://github.com/MedVisBonn/eyepy/commit/284242439163a0148d144938674fe98f10413fa9))
* **eyepy.io:** Fix imagio warnings; raise ValueError for scan-pattern 2 and 5 instead of warning; set maxint to NAN when reading XML layers; fix bscan order in layer data (vol_reader) ([`c4a88e7`](https://github.com/MedVisBonn/eyepy/commit/c4a88e7f390f10000e190c5345a5b465d0def7d2))

### Fix
* **src/eyepy/io:** Convert very large layer heights indicating no valid layer to np.nan ([`352a984`](https://github.com/MedVisBonn/eyepy/commit/352a98425dfa327d3fba7821986673bbd90802c6))
* **eyepy.core:** Make sure ticklabels match plotted image region for EyeEnfac ande EyeBscan plots ([`f389f47`](https://github.com/MedVisBonn/eyepy/commit/f389f476018db51ceb0554c23d9f9af94393a85c))
* **eyepy.core.utils.py:** Ignore nans in layers when computing drusen from layer heights ([`3c4efcd`](https://github.com/MedVisBonn/eyepy/commit/3c4efcdd30bbe491cf0f323be7359257ba8ce263))
* **eyepy.core:** Set axis in all layer height maps to (n_bscans, width) ([`3493b0e`](https://github.com/MedVisBonn/eyepy/commit/3493b0e259a18435b8c537839a359cb1c03e2d66))

### Documentation
* **README.md-/-Cookbook:** Add header image to README ([`7501dde`](https://github.com/MedVisBonn/eyepy/commit/7501dde6dabc48ccf6b905985888fa06ca34d6da))
* **README-and-Cookbook:** Fix import of load function in examples ([`e623c4b`](https://github.com/MedVisBonn/eyepy/commit/e623c4bed6f586c3c11ceba0801d4727da08cff8))

## v0.7.0 (2023-02-10)
### Feature
* **HeE2eReader:** Read E2E volumes ([`9094890`](https://github.com/MedVisBonn/eyepy/commit/9094890a5fd69335b741d238c1fa8661f83941c0))

### Fix
* **eyepy:** Do not use list and tuple for type annotations python 3.8 only supports List / Tuple ([`cc6dfee`](https://github.com/MedVisBonn/eyepy/commit/cc6dfee1098f6e962c2d5c09e33d5c82883000bf))
* **region-plotting:** Remove EllipsisType from allowed types for the region parameter since its not supported in python 3.8 ([`fac7849`](https://github.com/MedVisBonn/eyepy/commit/fac78498c233f765eb06d5bc71b3d71a125626ae))

### Breaking
*  ([`c942c6b`](https://github.com/MedVisBonn/eyepy/commit/c942c6b1b65a9deef07e7e2bd6d10cc96e2b6059))

### Documentation
* **README-and-Cookbook:** Clean up ([`183b317`](https://github.com/MedVisBonn/eyepy/commit/183b317dee559a2da8cc0aec2f0df4b237d7d605))
* **eyepy:** Add docstrings and cookbook examples ([`3b6ce5d`](https://github.com/MedVisBonn/eyepy/commit/3b6ce5d711f79794ffc1db91e1060a3a301bff40))
* **formats:** E2e documentation ([`059d67b`](https://github.com/MedVisBonn/eyepy/commit/059d67b0d335e3b82c1d631c2be24176c4fa2435))

## v0.6.8 (2022-09-15)
### Fix
* **eyepy:** Update pre-commit; remove itk from dependencies ([`0339fb6`](https://github.com/MedVisBonn/eyepy/commit/0339fb6f08aefd63027fb8213fc5da526a6890a7))

## v0.6.7 (2022-06-03)
### Fix
* **eyevolume.py:** Remove reformating of knot data in load - eyelab now does it if needed ([`35060ab`](https://github.com/MedVisBonn/eyepy/commit/35060ab079fd52a415cbe61f833622a73d4a4736))

## v0.6.6 (2022-06-03)
### Fix
* **eyevolume:** Auto convert old layer curves ([`7842120`](https://github.com/MedVisBonn/eyepy/commit/78421205c3cd4ddf4f9ff9700bf36dd9515bc4ab))

## v0.6.5 (2022-04-21)
### Fix
* **io/utils.py:** Check for parallel and equal distance B-scans ([`c5d68d2`](https://github.com/MedVisBonn/eyepy/commit/c5d68d23370a314714e4d0a798336ccb6a60f032))

## v0.6.4 (2022-04-21)
### Fix
* **lazy.py:** Fix shape of lazy volume ([`34b944f`](https://github.com/MedVisBonn/eyepy/commit/34b944fa02538be753352e32d4e2b046cd940b81))
* **eyevolume.py:** Enable import of B-scans with varying distances by replacing the raised Error by a warning; support deleteion of annotations ([`d8b4bb8`](https://github.com/MedVisBonn/eyepy/commit/d8b4bb870aa6a29439db0b46372610d2685b3995))

## v0.6.3 (2022-03-31)
### Fix
* **eyevolume.py:** Set default intensity transform if none is given ([`16b44bc`](https://github.com/MedVisBonn/eyepy/commit/16b44bc19f4839c9cbd5c382273c13942ecd2a10))

## v0.6.2 (2022-03-31)
### Fix
* **eyevolume.py:** Add intensity_transform to the saved meta data ([`c6a2c68`](https://github.com/MedVisBonn/eyepy/commit/c6a2c68e5f6c05dc86eba88db582473d41b2909a))
* **import_retouch:** Transform intensities correctly ([`112d1cc`](https://github.com/MedVisBonn/eyepy/commit/112d1cc178c3ab3bdb8b1f1130b95805eb5d729c))

## v0.6.1 (2022-03-26)
### Fix
* **pyproject.toml:** Set minimum python version to 3.7 for compatibility with pyinstaller docker container ([`75c008c`](https://github.com/MedVisBonn/eyepy/commit/75c008cade9fda68f967590cbfe7cb68251de9c8))

### Documentation
* **README.md:** Add DOI badge ([`c4d046b`](https://github.com/MedVisBonn/eyepy/commit/c4d046bb96a0c7b60cce5a6c4745e512e3f431ad))

## v0.6.0 (2022-03-25)
### Feature
* **eyevolume:** Enable use of EyeVolume in eyelab ([`8479628`](https://github.com/MedVisBonn/eyepy/commit/84796285a2eebb65497ac3cd438bdc212aa41e34))

### Documentation
* **README.md:** Add Related Projects section with reference to OCT-Converter ([`c273e44`](https://github.com/MedVisBonn/eyepy/commit/c273e448f177712cc67de8215b7afe448b005a4a))

## v0.5.0 (2022-03-02)
### Feature
* **eyevolume.py:** Enable custom intensity transformation for OCT data ([`761dd5a`](https://github.com/MedVisBonn/eyepy/commit/761dd5a4a360ca46d79ff2074b2644b4c2fd8ca4))

## v0.4.1 (2022-02-17)
### Fix
* **EyeVolume:** Fix B-scan iteration; enable setting layer heights from EyeBscan ([`f982d68`](https://github.com/MedVisBonn/eyepy/commit/f982d687a7834922866377b70a755e253c785880))

## v0.4.0 (2022-02-17)
### Breaking
*  ([`117ef89`](https://github.com/MedVisBonn/eyepy/commit/117ef89ec5874b366abb0da2b544011799a4e4f5))

## v0.3.7 (2022-01-31)
### Fix
* **base:** Fix error when plotting volumes without drusen; fix visibility of drusen projection ([`9c08c72`](https://github.com/MedVisBonn/eyepy/commit/9c08c7206809da406917a9c05a808a12de594660))

## v0.3.6 (2021-10-14)
### Fix
* **drusen.py:** Fix the drusen height filtering ([`4d1b375`](https://github.com/MedVisBonn/eyepy/commit/4d1b375561a5dbd1df634ba09c4aff20243f52df))

## v0.3.5 (2021-08-16)
### Fix
* **DefaultEyeQuantifier:** Enable radii change for default quantifier ([`ca8aff3`](https://github.com/MedVisBonn/eyepy/commit/ca8aff334a6ba26d187cd5e73178002b101e7d29))

## v0.3.4 (2021-08-16)
### Fix
* Fix the reference ([`eadf100`](https://github.com/MedVisBonn/eyepy/commit/eadf10098635d8e18eb2dea264b44c7535669ac5))

## v0.3.3 (2021-08-16)
### Fix
* **io/heyex/xml_export:** Initalize empty LayerAnnotation if no annotation is provided ([`6626467`](https://github.com/MedVisBonn/eyepy/commit/662646702bcf41ec3f98b4e891f3566ce613ba54))

## v0.3.2 (2021-08-16)
### Fix
* **eyepy/io/heyex:** Allow unknown heyex xml versions ([`5c51b46`](https://github.com/MedVisBonn/eyepy/commit/5c51b4656d0dcc28d4b7c2ff0e8112e176746e28))

## v0.3.1 (2021-05-18)
### Fix
* **base.py:** Fix layer mapping in case LayerAnnotation does not contain all layers ([`5e8621e`](https://github.com/MedVisBonn/eyepy/commit/5e8621ea50722281d1d4d56c12aa9ea574d5ef3a))

### Documentation
* **readme:** Added eyepy logo to readme.rst and removed readme.md ([`bc3e19d`](https://github.com/MedVisBonn/eyepy/commit/bc3e19d3120fa9a5329a6ad67ec9632a735d1d6e))

## v0.3.0 (2021-03-19)
### Feature
* **drusen.py:** Added new histogram based DrusenFinder and made it the new default ([`9a3e667`](https://github.com/MedVisBonn/eyepy/commit/9a3e667ba721c2f16085b4f62225d1ee9ded078d))

## v0.2.6 (2021-03-12)
### Fix
* **base.py:** Fixed bugs for oat ([`7e10ab3`](https://github.com/MedVisBonn/eyepy/commit/7e10ab30e4ac9f7499ed16c00fe09c9567a83765))

## v0.2.5 (2021-02-11)
### Fix
* **docs:** Add requirements.txt for docs ([`8008c63`](https://github.com/MedVisBonn/eyepy/commit/8008c634bad246559c4b7d5b60b18749af5bfb30))

## v0.2.4 (2021-02-11)
### Fix
* **travis.yml:** Removed --user option ([`f68df9e`](https://github.com/MedVisBonn/eyepy/commit/f68df9ed6c7889923ea50573ad10e47140d9f80d))
* **travis.yml:** Switch to new pip version to properly resolve twine dependencies ([`7215226`](https://github.com/MedVisBonn/eyepy/commit/7215226c4ca2a78da8cbbfd73ef36a5322c7bb22))

## v0.2.3 (2021-02-11)
### Fix
* **docs:** Remove eyepy as requirement for building docs ([`05d6293`](https://github.com/MedVisBonn/eyepy/commit/05d6293382368f6cce42286193e6618f2518c5a6))

## v0.2.2 (2021-02-11)
### Fix
* **setup.py-eyepy/__init__.py:** Make sure the version numbers match ([`7357f11`](https://github.com/MedVisBonn/eyepy/commit/7357f11ad7f3af4466ccd03d8fe0f7f846c8915e))

## v0.2.1 (2021-02-11)
### Fix
* **core.drusen.py:** Use logging module instead of prints in this package ([`978fa11`](https://github.com/MedVisBonn/eyepy/commit/978fa11079cff23d7d8ffb423f3767858dfd6f2e))

## v0.2.0 (2021-02-11)
### Fix
* Changed enface to localizer in OCT object ([`bf6ecd2`](https://github.com/MedVisBonn/eyepy/commit/bf6ecd2d7e10bfbadf0e46c9115d0b908014b639))

### Breaking
*  ([`bf6ecd2`](https://github.com/MedVisBonn/eyepy/commit/bf6ecd2d7e10bfbadf0e46c9115d0b908014b639))
*  ([`09f8746`](https://github.com/MedVisBonn/eyepy/commit/09f8746538ef18cda473fdd0644c96a0094a9f68))

## v1.0.0 (2021-02-10)
### Breaking
*  ([`09f8746`](https://github.com/MedVisBonn/eyepy/commit/09f8746538ef18cda473fdd0644c96a0094a9f68))

## v0.1.6 (2021-02-10)
### Fix
* **.travis.yml:** Another fix ([`79e3332`](https://github.com/MedVisBonn/eyepy/commit/79e33325605f87205a1797cb232328ef8698c21d))
* **.travis.yml:** Fixed yaml problem ([`28eac5e`](https://github.com/MedVisBonn/eyepy/commit/28eac5e62684ce749cd45845465bc0fa6e443d2c))
* **ci-testing:** Test whether a fix triggers the travic ci build ([`718e9ee`](https://github.com/MedVisBonn/eyepy/commit/718e9ee612e3345c51e57036f9b51c15c5e1a9b4))

### Documentation
* **readme:** Added eyepy logo to readme.rst and removed readme.md ([`bc3e19d`](https://github.com/MedVisBonn/eyepy/commit/bc3e19d3120fa9a5329a6ad67ec9632a735d1d6e))
