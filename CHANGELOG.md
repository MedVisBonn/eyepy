# CHANGELOG



## v0.12.2 (2023-11-15)

### Build

* build(_quality.yaml): removes caching of poetry ([`a7c20e1`](https://github.com/MedVisBonn/eyepy/commit/a7c20e14c46b1c17ec4b6a644d897852903c21f6))

* build(eyepy): updates versions ([`23fec54`](https://github.com/MedVisBonn/eyepy/commit/23fec5472f5e0dbfd5e9699660acdc2c2e80d9df))

* build(eyepy): removes python 3.8 from testing matrix ([`b2bcf7b`](https://github.com/MedVisBonn/eyepy/commit/b2bcf7bd85814b08affbd5964fed26f43d0b03eb))

* build(eyepy): upgrades dependencies ([`e66fd8b`](https://github.com/MedVisBonn/eyepy/commit/e66fd8b6fa96f4bfc22881a04e8b6c323f350e35))

### Fix

* fix(_release.yaml): corrects python-semantic-release version ([`f7171ba`](https://github.com/MedVisBonn/eyepy/commit/f7171bac689273660c8535b7498613f6c7a1afe8))

* fix(eyemeta.py): convert dates to string for saving as json ([`5d715cd`](https://github.com/MedVisBonn/eyepy/commit/5d715cd4295acb3a1c291f815e958cb3fc2a30e1))

### Refactor

* refactor(eyevolume.py): changes skimage import of GeometricTransform ([`31ce767`](https://github.com/MedVisBonn/eyepy/commit/31ce76776920ce7c91efc4f227b0e582d4f09cb9))

* refactor(EyeVolume): rename method from remove_pixel_annotations to remove_pixel_annotation ([`6727a60`](https://github.com/MedVisBonn/eyepy/commit/6727a60aed792e8f3fd36978a5279789e52b0bc6))


## v0.12.1 (2023-09-26)

### Ci

* ci(PythonSemanticRelease): update python semantic release to v8; minor change to README; added support for py3.11; updated dependencies ([`d4e65a8`](https://github.com/MedVisBonn/eyepy/commit/d4e65a88a608b3a867951d071801f26b50d663c0))

### Fix

* fix(vol_reader.py): get number of read layers from file context instead of assuming 17 layers. ([`2b53a33`](https://github.com/MedVisBonn/eyepy/commit/2b53a336633771373a5f142f63a29c81372af668))


## v0.12.0 (2023-05-31)

### Breaking

* build(eyepy): switch from eyepie to eyepy as a package name on PyPI

Thanks to @dillongwozdz for transfering the eyepy name to us.

BREAKING CHANGE: ([`8d07a17`](https://github.com/MedVisBonn/eyepy/commit/8d07a176aae030e5dd343f4c125f18b8395d7cc4))

### Ci

* ci(github/workflows): update github actions (setup-python to v4, checkout to v3, cache to v3) ([`1727f51`](https://github.com/MedVisBonn/eyepy/commit/1727f5171907328ae867a72026e8fc7136b66f15))


## v0.11.3 (2023-04-04)

### Documentation

* docs(README.md;-cookbook.md): Add examples for importing data from different sources ([`2a71e11`](https://github.com/MedVisBonn/eyepy/commit/2a71e111c75a31dc396ac4464ef627b84dcfa0d5))

### Fix

* fix(eyepy.io): fixes duke layer heights import ([`eb18f2a`](https://github.com/MedVisBonn/eyepy/commit/eb18f2aa10a7da8ab6f53c49053a80cfab764e68))


## v0.11.2 (2023-04-02)

### Fix

* fix(EyeVolume): fix saving, old method sometimes tried to recursively add the archive to the archive ([`e6064cc`](https://github.com/MedVisBonn/eyepy/commit/e6064cc26bb00865a1d9f42fd63ca950239fd55f))

* fix(import_duke_mat): fixes age parsing from duke data ([`ecbcbc2`](https://github.com/MedVisBonn/eyepy/commit/ecbcbc21349414d12746a9d46353b0494b79ef83))


## v0.11.1 (2023-03-31)

### Fix

* fix(EyeVolumeMeta): fixes saving issue by storing all dates as string in isoformat to avoid problems with dumping to json and loading ([`3fc7424`](https://github.com/MedVisBonn/eyepy/commit/3fc742485667b03505fd17d3bf25b8d2f20516de))


## v0.11.0 (2023-03-30)

### Feature

* feat(io.__init__.py): adds support for topcon fda files (#12)

* NF: add basic support for fda files

* refactor(main-and-io-module-__init__-files): specify topcon as vendor in fda import function

* feat(io.__init__.py): add bscan metadata to returned eye volume

before the fda reader only parsed bscans and segmentation. These changes incorporate scaling and bscan position in fundus images.

* feat(io.__init__.py): add fundus image to eyevolume

the changes allow to create a full EyeVolume with fundus image and metadata

* style(io.__init__.py): remove double quotes

---------

Co-authored-by: Olivier Morelle &lt;Oli4@users.noreply.github.com&gt; ([`53f2908`](https://github.com/MedVisBonn/eyepy/commit/53f2908f95556c2be7a3259bfd5653208188ff71))


## v0.10.0 (2023-03-30)

### Ci

* ci(.github): add pre-commit checking to quality workflow and call quality workflow for pull requests ([`9e47a09`](https://github.com/MedVisBonn/eyepy/commit/9e47a09248a29f36525aa7e406bb478ef8bcdf09))

### Documentation

* docs(core.annotations): adds docstrings ([`236c074`](https://github.com/MedVisBonn/eyepy/commit/236c074e8e28eb4ccb5a1e0538237fcec5dff415))

### Feature

* feat(core.plotting): adds a scale bar and a watermark to Bscan and Fundus visualizations ([`0e3eaa0`](https://github.com/MedVisBonn/eyepy/commit/0e3eaa0be4420c0d8ba4cbc53381fc36da2cdc81))

### Fix

* fix(eyepy.core): fixes bscan_region and bscan_position plotting when plotting only a part of the fundus

also adds new header_gif to the README ([`2c88074`](https://github.com/MedVisBonn/eyepy/commit/2c880743b6329cf24e74f7019035ad455800e89a))

* fix(core.grids.py): fixes error related to python 3.8 not supporting new type annotation Dict-dict List-list ([`3c6ba41`](https://github.com/MedVisBonn/eyepy/commit/3c6ba41133215bc64efb48f432cb0be78b627b2c))

### Style

* style(eyepy): applys improved pre-commit style guides

adds dostrings formatting via docformatter \n unifies single/double quotes \n upgrades type hints to be pep585 compliant ([`28b1b4c`](https://github.com/MedVisBonn/eyepy/commit/28b1b4c4767c686541eca7b069c6238d11191dd5))


## v0.9.2 (2023-03-17)

### Documentation

* docs(eyepy.core): adds type annotations to all objects in the core subpackage ([`ea1cb4c`](https://github.com/MedVisBonn/eyepy/commit/ea1cb4c415604a97d1e7ef64998b1743e7c5fe75))

* docs(annotations.py): adds type annotations to all objects in this module ([`986f9bc`](https://github.com/MedVisBonn/eyepy/commit/986f9bc6347227311725d31726ea38741e582d75))

### Fix

* fix(e2e_reader.py): makes exception formatting compatible with python &lt; 3.10 ([`2c4cbb0`](https://github.com/MedVisBonn/eyepy/commit/2c4cbb0c2a91c47d951945aa75145dd1e34e5c4e))


## v0.9.1 (2023-03-15)

### Ci

* ci(ci.yaml): set python version to 3.10 for building and deploying documentation ([`b6feb0d`](https://github.com/MedVisBonn/eyepy/commit/b6feb0d63dc47b732a354ac8b4b58fcc9809a203))

### Documentation

* docs(README.md): clarify that the localizer is a fundus image ([`4e41acd`](https://github.com/MedVisBonn/eyepy/commit/4e41acd3660f3a07abed11464bdcacba81bda984))

### Fix

* fix(e2e_reader.py): extract number of Bscans more reliably by using the number of slice substructures; Skip localizer affine transformation for now, because slodata is not always available; support reading single B-scan data ([`1287a24`](https://github.com/MedVisBonn/eyepy/commit/1287a24d40907085d33e867b35f458078faa6b75))

* fix(e2e_format.py): change the name of E2EFile to E2EFormat to avoid confustion with E2EFileStructure in e2e_reader.py ([`01be6a0`](https://github.com/MedVisBonn/eyepy/commit/01be6a0a6ce4e80491871800a970b7f75eb8c86d))


## v0.9.0 (2023-03-09)

### Breaking

* feat(HeE2EReader): switch to construct_typed for describing structures; create file hierarchy when parsing the E2E file; add search functions to the HeE2eReader

BREAKING CHANGE: ([`45578a5`](https://github.com/MedVisBonn/eyepy/commit/45578a5a9949a2722a03c1fe4bdef228c3a980c0))

### Documentation

* docs(documentation): improve documentation; rename Voxel/Area Annotation to PixelAnnotation for consistency ([`ab38837`](https://github.com/MedVisBonn/eyepy/commit/ab388378522f4be7f1fc55725a101a6483c1893b))

### Feature

* feat(eyepy.io.utils.py): add functions to search for integer/float values in binary data; set relative tolerance for Bscan distance to 4% (fixes sample data warning) ([`96fd58b`](https://github.com/MedVisBonn/eyepy/commit/96fd58bf1146b6f9b60d63cdb8abbbc8704f6bde))

### Fix

* fix(HeE2eReader): fix issues with inspect after renaming classes ([`891c79c`](https://github.com/MedVisBonn/eyepy/commit/891c79ceb2f0417c5b1b003fa1332ce1d4572b1b))

* fix(__init__.py): exclude __init__.py from isort to prevent circular import ([`9254231`](https://github.com/MedVisBonn/eyepy/commit/92542318b23098323855abf410c5c294ff956466))

* fix(pyproject.toml): add imageio as dependency and umpgrade imagecodecs to latest version ([`bef44a0`](https://github.com/MedVisBonn/eyepy/commit/bef44a0cca752e280f5a405a9b9c4e4d0590f292))


## v0.8.1 (2023-02-22)

### Fix

* fix(pyproject.toml): increase allowed version range for numpy (fixes #10) ([`c66f6f6`](https://github.com/MedVisBonn/eyepy/commit/c66f6f65909aef7b086b62f66d6ce55f0b04220e))


## v0.8.0 (2023-02-13)

### Build

* build(pyproject.toml): add itk as optional dependency to read RETOUCH data and pygifsicle as dev dependency to optimize the header.gif ([`ccf0d8e`](https://github.com/MedVisBonn/eyepy/commit/ccf0d8e941b728190c93dc23608e68a45a448533))

### Documentation

* docs(README.md-/-Cookbook): Add header image to README ([`7501dde`](https://github.com/MedVisBonn/eyepy/commit/7501dde6dabc48ccf6b905985888fa06ca34d6da))

* docs(README-and-Cookbook): fix import of load function in examples ([`e623c4b`](https://github.com/MedVisBonn/eyepy/commit/e623c4bed6f586c3c11ceba0801d4727da08cff8))

### Feature

* feat(eyepy.core): reflect plotted region in x and y axis for both localizer and B-scan; check if bscan_region bscan position indicators are in plotting region ([`2842424`](https://github.com/MedVisBonn/eyepy/commit/284242439163a0148d144938674fe98f10413fa9))

* feat(eyepy.io): fix imagio warnings; raise ValueError for scan-pattern 2 and 5 instead of warning; set maxint to NAN when reading XML layers; fix bscan order in layer data (vol_reader) ([`c4a88e7`](https://github.com/MedVisBonn/eyepy/commit/c4a88e7f390f10000e190c5345a5b465d0def7d2))

### Fix

* fix(src/eyepy/io): convert very large layer heights indicating no valid layer to np.nan ([`352a984`](https://github.com/MedVisBonn/eyepy/commit/352a98425dfa327d3fba7821986673bbd90802c6))

* fix(eyepy.core): make sure ticklabels match plotted image region for EyeEnfac ande EyeBscan plots ([`f389f47`](https://github.com/MedVisBonn/eyepy/commit/f389f476018db51ceb0554c23d9f9af94393a85c))

* fix(eyepy.core.utils.py): ignore nans in layers when computing drusen from layer heights ([`3c4efcd`](https://github.com/MedVisBonn/eyepy/commit/3c4efcdd30bbe491cf0f323be7359257ba8ce263))

* fix(eyepy.core): set axis in all layer height maps to (n_bscans, width)

Layer in the sample were not correctly oriented due to a different axis order in the HeXMLReader ([`3493b0e`](https://github.com/MedVisBonn/eyepy/commit/3493b0e259a18435b8c537839a359cb1c03e2d66))

### Style

* style(eyepy/config.py): change default color of bscan region and bscan position indicators from green to limegreen because of better visibility ([`3658a11`](https://github.com/MedVisBonn/eyepy/commit/3658a1111243b89f3fbcddf328cf67ad89962c9c))

### Test

* test(test_eyevolume.py): correct indexing of layer data

The first axis (bscan axis) of the layer data was flipped to reflect B-scan indexing. Therefore layers have to be flipped now when computing drusen and projecting on the localizer. ([`d186317`](https://github.com/MedVisBonn/eyepy/commit/d186317b403c254c84cb5aba014b3c018e0445fe))


## v0.7.0 (2023-02-10)

### Breaking

* refactor(eyepy): work in progress

BREAKING CHANGE: ([`c942c6b`](https://github.com/MedVisBonn/eyepy/commit/c942c6b1b65a9deef07e7e2bd6d10cc96e2b6059))

### Ci

* ci(pyproject.yaml): fix semantic release ([`daaea19`](https://github.com/MedVisBonn/eyepy/commit/daaea194e868f277ab8518fac8ff6dae470cfc84))

* ci(ci.yaml): fix syntax ([`c2647c6`](https://github.com/MedVisBonn/eyepy/commit/c2647c631f02d644b6b633388785847b4461d581))

* ci(ci.yaml): fix dependencies ([`7d4eb5f`](https://github.com/MedVisBonn/eyepy/commit/7d4eb5feb59ff87b82b1da65360c6d43e720de53))

* ci(ci.yaml): fix mkdocs deployment dependencies ([`a70a216`](https://github.com/MedVisBonn/eyepy/commit/a70a21608e4252de014da84e2712c5d69c5c2d55))

* ci(quality-check): upgrade poetry version ([`aedd680`](https://github.com/MedVisBonn/eyepy/commit/aedd6808f0e599dfe91c469826125b94116241ef))

### Documentation

* docs(README-and-Cookbook): clean up ([`183b317`](https://github.com/MedVisBonn/eyepy/commit/183b317dee559a2da8cc0aec2f0df4b237d7d605))

* docs(eyepy): add docstrings and cookbook examples ([`3b6ce5d`](https://github.com/MedVisBonn/eyepy/commit/3b6ce5d711f79794ffc1db91e1060a3a301bff40))

* docs(formats): e2e documentation ([`059d67b`](https://github.com/MedVisBonn/eyepy/commit/059d67b0d335e3b82c1d631c2be24176c4fa2435))

### Feature

* feat(HeE2eReader): read E2E volumes ([`9094890`](https://github.com/MedVisBonn/eyepy/commit/9094890a5fd69335b741d238c1fa8661f83941c0))

### Fix

* fix(eyepy): do not use list and tuple for type annotations python 3.8 only supports List / Tuple ([`cc6dfee`](https://github.com/MedVisBonn/eyepy/commit/cc6dfee1098f6e962c2d5c09e33d5c82883000bf))

* fix(region-plotting): remove EllipsisType from allowed types for the region parameter since its not supported in python 3.8 ([`fac7849`](https://github.com/MedVisBonn/eyepy/commit/fac78498c233f765eb06d5bc71b3d71a125626ae))

### Refactor

* refactor(eyepy): Use isort with google style for imports and yapf for code formating ([`47f9330`](https://github.com/MedVisBonn/eyepy/commit/47f9330483b5c2ecc6f3ee2830fa1091cb863861))

### Test

* test(eyevolume): skip save load test for now ([`bd58274`](https://github.com/MedVisBonn/eyepy/commit/bd582745fd3d09ea7767e27d25b21309e14e1e40))

* test(test_eyevolume): change delete to remove in function names ([`4804dc9`](https://github.com/MedVisBonn/eyepy/commit/4804dc9c8e1ebc660843bf69dd2f2a20976452fe))

* test(eyevolume): skip Vol writing for now ([`cbdc97e`](https://github.com/MedVisBonn/eyepy/commit/cbdc97eabc4e86efd2a97fe87ee5dec249aab7ba))


## v0.6.8 (2022-09-15)

### Fix

* fix(eyepy): update pre-commit; remove itk from dependencies

itk was used to read the RETOUCH dataset and might be added as an extra dependency ([`0339fb6`](https://github.com/MedVisBonn/eyepy/commit/0339fb6f08aefd63027fb8213fc5da526a6890a7))

### Refactor

* refactor(EyeVolumeVoxelAnnotation): simplify code for plotting ([`355e7f6`](https://github.com/MedVisBonn/eyepy/commit/355e7f67fbd082abb54b4b76c30eae7040a66ffa))


## v0.6.7 (2022-06-03)

### Fix

* fix(eyevolume.py): remove reformating of knot data in load - eyelab now does it if needed ([`35060ab`](https://github.com/MedVisBonn/eyepy/commit/35060ab079fd52a415cbe61f833622a73d4a4736))


## v0.6.6 (2022-06-03)

### Fix

* fix(eyevolume): auto convert old layer curves ([`7842120`](https://github.com/MedVisBonn/eyepy/commit/78421205c3cd4ddf4f9ff9700bf36dd9515bc4ab))


## v0.6.5 (2022-04-21)

### Fix

* fix(io/utils.py): check for parallel and equal distance B-scans ([`c5d68d2`](https://github.com/MedVisBonn/eyepy/commit/c5d68d23370a314714e4d0a798336ccb6a60f032))


## v0.6.4 (2022-04-21)

### Fix

* fix(lazy.py): fix shape of lazy volume ([`34b944f`](https://github.com/MedVisBonn/eyepy/commit/34b944fa02538be753352e32d4e2b046cd940b81))

* fix(eyevolume.py): enable import of B-scans with varying distances by replacing the raised Error by a warning; support deleteion of annotations ([`d8b4bb8`](https://github.com/MedVisBonn/eyepy/commit/d8b4bb870aa6a29439db0b46372610d2685b3995))

### Test

* test(test_grid.py): add tests for ETDRS grid creation ([`c9c13d1`](https://github.com/MedVisBonn/eyepy/commit/c9c13d16e8ecbcc616bc6f215008812a7f37dd47))


## v0.6.3 (2022-03-31)

### Fix

* fix(eyevolume.py): set default intensity transform if none is given ([`16b44bc`](https://github.com/MedVisBonn/eyepy/commit/16b44bc19f4839c9cbd5c382273c13942ecd2a10))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`e57a91e`](https://github.com/MedVisBonn/eyepy/commit/e57a91ead0516665540da67048aeeaa892b23fb3))


## v0.6.2 (2022-03-31)

### Fix

* fix(eyevolume.py): add intensity_transform to the saved meta data

this is required to correctly restore saved data with non-default intensity_transform. Custom intensity transform functions can not be saved currently. ([`c6a2c68`](https://github.com/MedVisBonn/eyepy/commit/c6a2c68e5f6c05dc86eba88db582473d41b2909a))

* fix(import_retouch): transform intensities correctly

Topcon and Cirrus data is stored as UCHAR while spectralis is stored as USHORT ([`112d1cc`](https://github.com/MedVisBonn/eyepy/commit/112d1cc178c3ab3bdb8b1f1130b95805eb5d729c))


## v0.6.1 (2022-03-26)

### Documentation

* docs(README.md): add DOI badge ([`c4d046b`](https://github.com/MedVisBonn/eyepy/commit/c4d046bb96a0c7b60cce5a6c4745e512e3f431ad))

### Fix

* fix(pyproject.toml): set minimum python version to 3.7 for compatibility with pyinstaller docker container ([`75c008c`](https://github.com/MedVisBonn/eyepy/commit/75c008cade9fda68f967590cbfe7cb68251de9c8))


## v0.6.0 (2022-03-25)

### Documentation

* docs(README.md): add Related Projects section with reference to OCT-Converter ([`c273e44`](https://github.com/MedVisBonn/eyepy/commit/c273e448f177712cc67de8215b7afe448b005a4a))

### Feature

* feat(eyevolume): enable use of EyeVolume in eyelab ([`8479628`](https://github.com/MedVisBonn/eyepy/commit/84796285a2eebb65497ac3cd438bdc212aa41e34))

* feat(eyevolume.py): enable custom intensity transformation for OCT data ([`761dd5a`](https://github.com/MedVisBonn/eyepy/commit/761dd5a4a360ca46d79ff2074b2644b4c2fd8ca4))


## v0.4.1 (2022-02-17)

### Fix

* fix(EyeVolume): fix B-scan iteration; enable setting layer heights from EyeBscan ([`f982d68`](https://github.com/MedVisBonn/eyepy/commit/f982d687a7834922866377b70a755e253c785880))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`a63cdd8`](https://github.com/MedVisBonn/eyepy/commit/a63cdd8b0430ff45cbb757995c44bd02ec3a21a5))


## v0.4.0 (2022-02-17)

### Breaking

* refactor(everything): Refactor for easier to maintain code

CI based on Github workflows, EyeVolume class as a standard way for handling OCT volumes, support for HEXEX VOL/XML, Bscans from folder, DUKE and RETOUCH data

BREAKING CHANGE: ([`117ef89`](https://github.com/MedVisBonn/eyepy/commit/117ef89ec5874b366abb0da2b544011799a4e4f5))


## v0.3.7 (2022-01-31)

### Fix

* fix(base): fix error when plotting volumes without drusen; fix visibility of drusen projection ([`9c08c72`](https://github.com/MedVisBonn/eyepy/commit/9c08c7206809da406917a9c05a808a12de594660))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`ed503da`](https://github.com/MedVisBonn/eyepy/commit/ed503da470ff7f467ff5ab3a69b5a50044d7e133))

* minor change ([`eb337bd`](https://github.com/MedVisBonn/eyepy/commit/eb337bda7b6721bb83364ec43875ad8d06f42cab))


## v0.3.6 (2021-10-14)

### Fix

* fix(drusen.py): fix the drusen height filtering ([`4d1b375`](https://github.com/MedVisBonn/eyepy/commit/4d1b375561a5dbd1df634ba09c4aff20243f52df))


## v0.3.5 (2021-08-16)

### Fix

* fix(DefaultEyeQuantifier): enable radii change for default quantifier ([`ca8aff3`](https://github.com/MedVisBonn/eyepy/commit/ca8aff334a6ba26d187cd5e73178002b101e7d29))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`de0aba3`](https://github.com/MedVisBonn/eyepy/commit/de0aba39c87251781b2753c9e6c6c38e0b6dc2dc))


## v0.3.4 (2021-08-16)

### Fix

* fix: fix the reference ([`eadf100`](https://github.com/MedVisBonn/eyepy/commit/eadf10098635d8e18eb2dea264b44c7535669ac5))


## v0.3.3 (2021-08-16)

### Fix

* fix(io/heyex/xml_export): initalize empty LayerAnnotation if no annotation is provided ([`6626467`](https://github.com/MedVisBonn/eyepy/commit/662646702bcf41ec3f98b4e891f3566ce613ba54))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`d4dc744`](https://github.com/MedVisBonn/eyepy/commit/d4dc7441f7b9a9713f731a2b200da5b9a063c15a))


## v0.3.2 (2021-08-16)

### Fix

* fix(eyepy/io/heyex): allow unknown heyex xml versions

show a warning and use the base XML specification for HEYEX XML exports ([`5c51b46`](https://github.com/MedVisBonn/eyepy/commit/5c51b4656d0dcc28d4b7c2ff0e8112e176746e28))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`54e0a6c`](https://github.com/MedVisBonn/eyepy/commit/54e0a6c94dd21f942590ed12b002e93f50f08d8c))


## v0.3.1 (2021-05-18)

### Fix

* fix(base.py): fix layer mapping in case LayerAnnotation does not contain all layers ([`5e8621e`](https://github.com/MedVisBonn/eyepy/commit/5e8621ea50722281d1d4d56c12aa9ea574d5ef3a))

### Unknown

* Merge remote-tracking branch &#39;origin/master&#39;

# Conflicts:
#	eyepy/core/drusen.py ([`569a9ed`](https://github.com/MedVisBonn/eyepy/commit/569a9ede6d08050e647407bbe8e119835040583e))


## v0.3.0 (2021-03-19)

### Feature

* feat(drusen.py): added new histogram based DrusenFinder and made it the new default

The old default is renamed to DrusenFinderPolyFit. The new method estimates a single iRPE distance to the BM for the complete volume. We found that the iRPE found like this has a similar distance to the BM as the RPE in healthy regions hasto the BM. The iRPE computed by the old method has a larger difference to the BM. ([`9a3e667`](https://github.com/MedVisBonn/eyepy/commit/9a3e667ba721c2f16085b4f62225d1ee9ded078d))


## v0.2.6 (2021-03-12)

### Fix

* fix(base.py): fixed bugs for oat ([`7e10ab3`](https://github.com/MedVisBonn/eyepy/commit/7e10ab30e4ac9f7499ed16c00fe09c9567a83765))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`64e6f46`](https://github.com/MedVisBonn/eyepy/commit/64e6f46f2c8661f4f4d56fa1f48ceb48123c0154))

* Layer annotation works with DB sync ([`e676eec`](https://github.com/MedVisBonn/eyepy/commit/e676eec4b902ecf9252cce154a1cb7c480f9af98))

* Upload and show line annotations ([`7386893`](https://github.com/MedVisBonn/eyepy/commit/7386893c4729755ff5f454e33baf8283d5e1b3be))

* Lines and Areas Groups; Import of vol, xml and bscan folder; bug fixes; annotation view active image indicator; ([`d858a6f`](https://github.com/MedVisBonn/eyepy/commit/d858a6f33630a98e3ef997ebb4d68b7e6a36b324))


## v0.2.5 (2021-02-11)

### Fix

* fix(docs): add requirements.txt for docs

readthedocs nee to know the dependencies for building the documentation ([`8008c63`](https://github.com/MedVisBonn/eyepy/commit/8008c634bad246559c4b7d5b60b18749af5bfb30))


## v0.2.4 (2021-02-11)

### Fix

* fix(travis.yml): removed --user option ([`f68df9e`](https://github.com/MedVisBonn/eyepy/commit/f68df9ed6c7889923ea50573ad10e47140d9f80d))

* fix(travis.yml): switch to new pip version to properly resolve twine dependencies ([`7215226`](https://github.com/MedVisBonn/eyepy/commit/7215226c4ca2a78da8cbbfd73ef36a5322c7bb22))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`d14ef68`](https://github.com/MedVisBonn/eyepy/commit/d14ef6868b37e5a98f479a4b8398016fc7f1d549))


## v0.2.3 (2021-02-11)

### Fix

* fix(docs): remove eyepy as requirement for building docs ([`05d6293`](https://github.com/MedVisBonn/eyepy/commit/05d6293382368f6cce42286193e6618f2518c5a6))


## v0.2.2 (2021-02-11)

### Fix

* fix(setup.py-eyepy/__init__.py): make sure the version numbers match ([`7357f11`](https://github.com/MedVisBonn/eyepy/commit/7357f11ad7f3af4466ccd03d8fe0f7f846c8915e))

### Style

* style(project): run pre-commit hooks and cleaned documentation ([`a882ad4`](https://github.com/MedVisBonn/eyepy/commit/a882ad496bc6477ddeeb63934572143b05a09429))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`edd53a0`](https://github.com/MedVisBonn/eyepy/commit/edd53a0a557f0b91d1518b4969fb22d687971cd6))


## v0.2.1 (2021-02-11)

### Fix

* fix(core.drusen.py): use logging module instead of prints in this package ([`978fa11`](https://github.com/MedVisBonn/eyepy/commit/978fa11079cff23d7d8ffb423f3767858dfd6f2e))


## v0.2.0 (2021-02-11)

### Breaking

* fix: changed enface to localizer in OCT object

BREAKING CHANGE: ([`bf6ecd2`](https://github.com/MedVisBonn/eyepy/commit/bf6ecd2d7e10bfbadf0e46c9115d0b908014b639))

* refactor(eyepy/core/base.py): rename enface to localizer in Oct object for more consistency

BREAKING CHANGE: ([`09f8746`](https://github.com/MedVisBonn/eyepy/commit/09f8746538ef18cda473fdd0644c96a0094a9f68))

### Unknown

* Merge branch &#39;master&#39; of github.com:MedVisBonn/eyepy ([`710d6d8`](https://github.com/MedVisBonn/eyepy/commit/710d6d8a51a2ec607f2acb09099269b636ff4878))


## v0.1.6 (2021-02-10)

### Ci

* ci(Travis-CI): changed supported versions and location of version string ([`397deca`](https://github.com/MedVisBonn/eyepy/commit/397decabbe64e6ab812de9aa8adc1450d17ae1e7))

* ci: add changed files before commiting ([`8e1cb24`](https://github.com/MedVisBonn/eyepy/commit/8e1cb24a6bc59f0cf1f986da39d36e90c98e11b0))

### Documentation

* docs(readme): added eyepy logo to readme.rst and removed readme.md ([`bc3e19d`](https://github.com/MedVisBonn/eyepy/commit/bc3e19d3120fa9a5329a6ad67ec9632a735d1d6e))

### Fix

* fix(.travis.yml): another fix ([`79e3332`](https://github.com/MedVisBonn/eyepy/commit/79e33325605f87205a1797cb232328ef8698c21d))

* fix(.travis.yml): fixed yaml problem ([`28eac5e`](https://github.com/MedVisBonn/eyepy/commit/28eac5e62684ce749cd45845465bc0fa6e443d2c))

* fix(ci-testing): test whether a fix triggers the travic ci build

no significant changes ([`718e9ee`](https://github.com/MedVisBonn/eyepy/commit/718e9ee612e3345c51e57036f9b51c15c5e1a9b4))

### Refactor

* refactor(registration): Added rigid registration of multimodal images based on mean phase images and HOG features. Also added an examplary notebook ([`7ce9ebd`](https://github.com/MedVisBonn/eyepy/commit/7ce9ebdd1717ed3b171f132b13d9e452b43a6cb8))

* refactor(project): added 2D rigid registration ([`67f5908`](https://github.com/MedVisBonn/eyepy/commit/67f59085b8fad04c9601a045ce9c327e4658fc22))

### Style

* style: removed index.html files ([`81bf883`](https://github.com/MedVisBonn/eyepy/commit/81bf883de614a7488aeec843d657f8c3c70faab7))

### Unknown

* Another try ([`a450ec6`](https://github.com/MedVisBonn/eyepy/commit/a450ec61db05a9faf0b1bea5bf17cba0858596e2))

* Changed project name from eyepy to eyepie for PyPi ([`64313c4`](https://github.com/MedVisBonn/eyepy/commit/64313c4429bc8383deb2461369498998f62201ec))

* Setting up pypi upload ([`88a6a33`](https://github.com/MedVisBonn/eyepy/commit/88a6a3365e684aa8da93487a7ad1b097de80b2e4))

* work in progress ([`b6833a1`](https://github.com/MedVisBonn/eyepy/commit/b6833a1d04eed7cf5208a1f0abfcd6ac40845843))

* Added annotation comparison ([`1bb8f33`](https://github.com/MedVisBonn/eyepy/commit/1bb8f3388ef376335169d1bcd388cd1898ec1fbb))

* small changes ([`ae3b464`](https://github.com/MedVisBonn/eyepy/commit/ae3b4644fd91fa6c4fc6db0ca639513f16f137e7))

* unknown ([`b518e0c`](https://github.com/MedVisBonn/eyepy/commit/b518e0ce1047b0c18f7c5e4f4e92f0d5fdc966c1))

* Fixed drusen filters ([`c265d6c`](https://github.com/MedVisBonn/eyepy/commit/c265d6cafb1ffbad37471f3a3e902ce789e0486a))

* Severall small fixes and improvements ([`161ecb4`](https://github.com/MedVisBonn/eyepy/commit/161ecb45f0660465cbfea41a89bda3c57be32fe1))

* many small improvements; drusen computation now in 4s instead of 90s ([`cbae8db`](https://github.com/MedVisBonn/eyepy/commit/cbae8db2ebf17eaad9828cbe82ba354365a8b73e))

* layer ploting with region works ([`fb67ede`](https://github.com/MedVisBonn/eyepy/commit/fb67ede4a94fd4914fdacda99f53bd01d49310ba))

* minor change ([`a1fa0bb`](https://github.com/MedVisBonn/eyepy/commit/a1fa0bb0fca5f19c55b825ef9b47ac5c0d0f2460))

* default Annotation for B-Scan ([`758d23f`](https://github.com/MedVisBonn/eyepy/commit/758d23fdf82e7e6d7f48f9e2451cd6b171c79351))

* some changes ([`8935ab4`](https://github.com/MedVisBonn/eyepy/commit/8935ab4719b21974e2eaf662b3eddd1dbcb1d9a9))

* Default empty LayerAnnotation works ([`72a6a43`](https://github.com/MedVisBonn/eyepy/commit/72a6a43322c087eadb308cd64fedb1ffc4da62ce))

* fix the layer annotation fix :/ ([`fa5ee81`](https://github.com/MedVisBonn/eyepy/commit/fa5ee81cd690ff35e08b3cb15a999c3bea4daa9e))

* Added default empty layer annotation and fixed Oct.layer ([`5c8a395`](https://github.com/MedVisBonn/eyepy/commit/5c8a3957f2fc1b60cb49e115cc0cff1845124dd5))

* Assume square area for OCT if no enface and or no layer positions are given; Enabled changing the layer annotations ([`5e244f0`](https://github.com/MedVisBonn/eyepy/commit/5e244f096b48eb0191d048272fa299200be002f5))

* Added name to Bscan. If B-Scans are save as individual files, this is the respective filename, else it becomes a string version of the B-Scan index ([`0b09ebe`](https://github.com/MedVisBonn/eyepy/commit/0b09ebe5deb1d8f366b788eb96bea3ac244b1ec8))

* Merge remote-tracking branch &#39;public/master&#39; ([`dd352c6`](https://github.com/MedVisBonn/eyepy/commit/dd352c65e855cbab2b75e7c2a1d14e732560a9cf))

* added layer_indices property to Bscan ([`bd4f669`](https://github.com/MedVisBonn/eyepy/commit/bd4f6691b3431bbdc2dd158c9c11934492240167))

* Merge remote-tracking branch &#39;public/master&#39; ([`64db876`](https://github.com/MedVisBonn/eyepy/commit/64db8764c99ee13c4caa7c583f0496c36a990cd6))

* Imports fixed ([`cb2759a`](https://github.com/MedVisBonn/eyepy/commit/cb2759a1e432f2c492846a561049b7302a811251))

* Merge remote-tracking branch &#39;public/master&#39;

# Conflicts:
#	eyepy/core/base.py ([`1b4de44`](https://github.com/MedVisBonn/eyepy/commit/1b4de44c6dd9734d99ad811071a17da5bcd97904))

* Rewrite Heyex vol and xml readers. Clean up ([`f97da6e`](https://github.com/MedVisBonn/eyepy/commit/f97da6ea8509b4c21484e53c956683f63aba098a))

* Merge remote-tracking branch &#39;public/master&#39;

# Conflicts:
#	eyepy/core/octbase.py
#	eyepy/io/heyex/he_xml.py ([`965bc45`](https://github.com/MedVisBonn/eyepy/commit/965bc456b5f5a53365874486f14e62f98b01e6c9))

* minor fix ([`b1b97f3`](https://github.com/MedVisBonn/eyepy/commit/b1b97f3429a08635d54b6acc82704848ed7a7cbc))

* minor fix ([`266f837`](https://github.com/MedVisBonn/eyepy/commit/266f837ce5aa508f3e6ab51e106b766e42e915f3))

* added drusen depth filter ([`8acd827`](https://github.com/MedVisBonn/eyepy/commit/8acd82757a0482785194e81f6d458f76dca9de08))

* bumbed version ([`5a1000a`](https://github.com/MedVisBonn/eyepy/commit/5a1000a462d8484409c20384fa01436c25edff87))

* Added drusen saving in data_path/.eyepy folder. Added function to recompute drusen if needed with a custom drusenfinder ([`40ed65d`](https://github.com/MedVisBonn/eyepy/commit/40ed65d946db3e630294a3a99165aea89da506ef))

* Merge remote-tracking branch &#39;origin/master&#39; into dev

# Conflicts:
#	eyepy/core/drusen.py ([`69a6f71`](https://github.com/MedVisBonn/eyepy/commit/69a6f712a71a1690ce3ea6b55584bcb877151ac1))

* Drusen depth filter added ([`d33f0d6`](https://github.com/MedVisBonn/eyepy/commit/d33f0d66a4d6810ce5d33a73fc8de02936ad3ad3))

* layer_indices and enface filename ([`2ee6f41`](https://github.com/MedVisBonn/eyepy/commit/2ee6f41470ad75be07e420d81e4d4c62dd5e1a06))

* Work in progress, adding reading functionallity for bscan only folders ([`8d859c0`](https://github.com/MedVisBonn/eyepy/commit/8d859c0b433c10a81ae80ad6e1b31828c54f276b))

* Changed Repository ([`5e0aa32`](https://github.com/MedVisBonn/eyepy/commit/5e0aa327684a93da006fb24242a33867eda3d8b5))

* Merge branch &#39;dev&#39;

# Conflicts:
#	eyepy/preprocess/loggabor.py ([`1368604`](https://github.com/MedVisBonn/eyepy/commit/13686042b3d0579963c45a6f3e739a5d529f6025))

* Added DefaultEyeQuantifier and improved plotting ([`a00b25a`](https://github.com/MedVisBonn/eyepy/commit/a00b25a6602771b060661ca6e9d4e12c056cf8b7))

* Minor fixes and clean up ([`2ce4962`](https://github.com/MedVisBonn/eyepy/commit/2ce4962a7f1b01d88b6f07e418fd454aba503b90))

* Removed code duplication ([`c02d2dd`](https://github.com/MedVisBonn/eyepy/commit/c02d2dd8729ea89dd7fb7a698a6149c560a8ee41))

* Added EyeQuantifier and DefaultEyeQuantifier for standard drusen quantification ([`6b33b6f`](https://github.com/MedVisBonn/eyepy/commit/6b33b6fd4a034af55705baeae41fce2103976470))

* Added loader for sample data ([`451586d`](https://github.com/MedVisBonn/eyepy/commit/451586daa68b2bebe607f0f064013c73575c9f73))

* Added DrusenFinder to octbase ([`46ec78c`](https://github.com/MedVisBonn/eyepy/commit/46ec78c54778143b23003a0c583e43239f307edd))

* Work in progress on the DrusenFinder ([`bb69833`](https://github.com/MedVisBonn/eyepy/commit/bb69833bc2735c960c19a16fe966637486a8a40f))

* When loading B-Scans assume image has 2 dimensions. In case it as 3 dimensions (last dimension for RGB) keep only the R array ([`0290a7b`](https://github.com/MedVisBonn/eyepy/commit/0290a7b11d2ebcb0de11f8c46958bec923316783))

* minor fix ([`04f8d6b`](https://github.com/MedVisBonn/eyepy/commit/04f8d6b461096d70c818e17e4bdf5755d8b6f71e))

* Merge branch &#39;master&#39; into merge_to_public

# Conflicts:
#	setup.py ([`b300bd6`](https://github.com/MedVisBonn/eyepy/commit/b300bd6d18bd60a01e1f4341cceae2d7e63b4be0))

* added drusen metrics and fixed drusen computation ([`1e38f65`](https://github.com/MedVisBonn/eyepy/commit/1e38f65f456b9e799003c079f314e376fb73850c))

* Drusen code is clean, tests missing ([`c28de74`](https://github.com/MedVisBonn/eyepy/commit/c28de743ca1a943cab8ff18f1af2a444b58edc47))

* Started to add drusen computation from layer segmentation. ([`5e8d511`](https://github.com/MedVisBonn/eyepy/commit/5e8d511352cebb6a2916d621b34d0a7ac7fcdd0a))

* Store only one channel of the loaded bscan ([`b7d0827`](https://github.com/MedVisBonn/eyepy/commit/b7d0827895a532380812f750cbdd6024ca1a77b8))

* Added seaborn dependcy ([`9e41356`](https://github.com/MedVisBonn/eyepy/commit/9e413569524eed787b1b87d9a14a172d13c98ccb))

* relaxed dependencies for now ([`c2a3ed8`](https://github.com/MedVisBonn/eyepy/commit/c2a3ed8b23fc8c23a6a7b159528e065340dc32f2))

* initial commit ([`c0cb0ca`](https://github.com/MedVisBonn/eyepy/commit/c0cb0cad26c2615c48acd5312fd712ac057b18c0))

* minor plotting changes ([`ceb5fed`](https://github.com/MedVisBonn/eyepy/commit/ceb5fed3380451a3d1c3970d437ddb0b2816e407))

* fixed specification for vol files ([`0afb7ee`](https://github.com/MedVisBonn/eyepy/commit/0afb7ee3136171826a4dce47ea1bda4d697f7b63))

* bug fixes and reorganization ([`2a1dde1`](https://github.com/MedVisBonn/eyepy/commit/2a1dde1ad4dd804c150559053e49c4ed719322a9))

* latest changes ([`39ccccd`](https://github.com/MedVisBonn/eyepy/commit/39ccccd45cd86076068854c5e0132afc00e5655b))

* old changes ([`2ca90dc`](https://github.com/MedVisBonn/eyepy/commit/2ca90dc852bb8d798078fa97bb86ebe603db7a1f))

* Added base classes for Oct and Bscan objects which define the interface and deliver basic plotting functionality ([`a5f82df`](https://github.com/MedVisBonn/eyepy/commit/a5f82df29db52d1335fcd2d0063cae62b38f4514))

* Fixed parsing B-Scans for the Heyex XML export ([`70b101c`](https://github.com/MedVisBonn/eyepy/commit/70b101cb06d586873c0c75f76bc04b55fabadb2e))

* read Heyex XML exports with the same class as the Heyex VOL export. This makes both exports accessible using the same interface ([`b05b3a0`](https://github.com/MedVisBonn/eyepy/commit/b05b3a0672ade242968e0670c2b2249c11f78056))

* memory mapped file for reading .vol ([`37b31d5`](https://github.com/MedVisBonn/eyepy/commit/37b31d5fbb151145a619f6ca74a1f5b21e1fd394))

* fixed comma ([`efb4dc7`](https://github.com/MedVisBonn/eyepy/commit/efb4dc7d69528796753bd19586cb0a1e7a15b586))

* Added support for reading segmentations from .vol ([`5cebe5e`](https://github.com/MedVisBonn/eyepy/commit/5cebe5ebfd5abd14004f4bd8954ed5c49ea5786e))

* Merge remote-tracking branch &#39;origin/master&#39;

# Conflicts:
#	docs/conf.py
#	eyepy/io/__init__.py
#	eyepy/io/base.py
#	eyepy/io/he_vol.py ([`95eb730`](https://github.com/MedVisBonn/eyepy/commit/95eb73041d750d0b5ffe295b89ba5b2f67bb8076))

* HeyexOct docstring ([`f539013`](https://github.com/MedVisBonn/eyepy/commit/f539013293872040d5930b88da29339b3f58efb0))

* Added sphinx extensions for numpy style docstrings (napoleon) and typehints (sphinx_autodoc_typehints) ([`8faa2d7`](https://github.com/MedVisBonn/eyepy/commit/8faa2d7b87c081efd875f59ed50af958c0858815))

* Documentation and clean up in progress ([`8770649`](https://github.com/MedVisBonn/eyepy/commit/877064904c5ba8b8551aa7bdc5056a773fb5da2e))

* Rewrite vol import ([`d9b138e`](https://github.com/MedVisBonn/eyepy/commit/d9b138eab1630fc540b006111a6902c6fa848b5f))

* changes to the filtergrid to use it more flexible ([`5c3c0a7`](https://github.com/MedVisBonn/eyepy/commit/5c3c0a7bc9c5d2d98c7e8fa4a2a1cbefd6a2e6e7))

* Another commit ([`64e1d1b`](https://github.com/MedVisBonn/eyepy/commit/64e1d1bf1d689636950e27d0ed60f0e2ae8deda9))

* new approach ([`9d883af`](https://github.com/MedVisBonn/eyepy/commit/9d883afacb95b541c677fb9d3ec307f596043269))

* registartion progress ([`5705254`](https://github.com/MedVisBonn/eyepy/commit/5705254950fbac17c52d7e12b965edbfa3c82e33))

* log gabor completeted ([`7f42ea6`](https://github.com/MedVisBonn/eyepy/commit/7f42ea6e812cfe81c64b01ecdaac021c9d31eba7))

* ... ([`f5b451a`](https://github.com/MedVisBonn/eyepy/commit/f5b451acaf9d9960f2bcdec059e8d10aa99671fd))

* progress on registration ([`8fb4755`](https://github.com/MedVisBonn/eyepy/commit/8fb475538ab6cb290dc3b7dda0562cf77ecf6347))

* progress on log gabor ([`51e5205`](https://github.com/MedVisBonn/eyepy/commit/51e5205aa42992e752da56812f6f63436493d6b5))

* work in progress register nir/cfp ([`b09762d`](https://github.com/MedVisBonn/eyepy/commit/b09762dbc5af344e92967215989297fb06b24c02))

* Project init ([`ad34732`](https://github.com/MedVisBonn/eyepy/commit/ad3473284d8c5edf26283a16f86c966d49d51647))

* Initial commit ([`d1b9af2`](https://github.com/MedVisBonn/eyepy/commit/d1b9af22e3b60145ed43307b785832b480731dad))
