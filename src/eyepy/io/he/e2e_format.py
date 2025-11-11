import dataclasses
import datetime
import logging
import typing as t

import construct as cs
from construct_typed import csfield
from construct_typed import DataclassMixin
from construct_typed import DataclassStruct
from construct_typed import EnumBase
from construct_typed import EnumValue
from construct_typed import TEnum
import numpy as np

logger = logging.getLogger(__name__)


class BscanAdapter(cs.Adapter):

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)
        self.LUT = self._make_LUT()
        self.inv_LUT = self._make_inv_LUT()

    def _uint16_to_ufloat16(self, uint_16):
        """Implementation of custom float type used in .e2e files.

        Notes:
            Custom float is a floating point type with no sign, 6-bit exponent, and 10-bit mantissa.
        Args:
            uint16 (int):
        Returns:
            float
        """
        bits = '{0:016b}'.format(uint_16)[::-1]
        # get mantissa and exponent
        mantissa = bits[:10]
        exponent = bits[10:]
        exponent = exponent[::-1]

        # convert to decimal representations
        mantissa_sum = 1 + int(mantissa, 2) / pow(2, 10)
        exponent_sum = int(exponent, 2) - 63
        decimal_value = mantissa_sum * np.float_power(2, exponent_sum)
        return decimal_value.astype(np.float32)

    def _make_LUT(self):
        LUT = []
        for i in range(0, pow(2, 16)):
            LUT.append(self._uint16_to_ufloat16(i))
        return np.array(LUT)

    def _make_inv_LUT(self):
        return {f: i for i, f in enumerate(self.LUT)}

    def _decode(self, obj: bytes, context, path):
        return self.LUT[np.ndarray(buffer=obj,
                                   dtype=np.uint16,
                                   shape=(context.height, context.width))]

    def _encode(self, obj: np.ndarray, context, path):
        return np.array([self.inv_LUT[f] for f in obj.flatten()],
                        dtype=np.uint16).tobytes()


class LocalizerNIRAdapter(cs.Adapter):

    def _decode(self, obj: bytes, context, path):
        return np.ndarray(buffer=obj,
                          dtype='uint8',
                          shape=(context.height, context.width))

    def _encode(self, obj: np.ndarray, context, path):
        return obj.astype(np.uint8).tobytes()


class SegmentationAdapter(cs.Adapter):

    def _decode(self, obj: bytes, context, path):
        return np.ndarray(buffer=obj, dtype='float32', shape=(context.width))

    def _encode(self, obj: np.ndarray, context, path):
        return obj.astype(np.float32).tobytes()


class DateTimeAdapter(cs.Adapter):
    # with utc bias 120
    start_epoch = datetime.datetime(year=1601,
                                    month=1,
                                    day=1,
                                    hour=0,
                                    minute=0)

    def _decode(self, obj: bytes, context, path):
        return (self.start_epoch + datetime.timedelta(
            seconds=cs.Int64ul.parse(obj) * 1e-7)).isoformat()

    def _encode(self, obj: str, context, path):
        return cs.Int64ul.build(
            int((datetime.datetime.fromisoformat(obj) -
                 self.start_epoch).total_seconds() * 1e7))


LocalizerNIR = LocalizerNIRAdapter(cs.Bytes(cs.this.n_values))
Segmentation = SegmentationAdapter(cs.Bytes(cs.this.width * 4))
Bscan = BscanAdapter(cs.Bytes(cs.this.n_values * 2))
DateTime = DateTimeAdapter(cs.Bytes(8))


class LateralityEnum(EnumBase):
    """Enum for laterality of eye.

    The laterality is stored as single character in ASCII code.
    """
    OD = EnumValue(82, doc="82 is the ASCII code for 'R'")
    OS = EnumValue(76, doc="76 is the ASCII code for 'L'")


class TypesEnum(EnumBase):
    """Enum for types of data stored in .e2e files."""
    patient = EnumValue(9)
    laterality = EnumValue(11)
    diagnose = EnumValue(17)
    bscanmeta = EnumValue(10004)
    layer_annotation = EnumValue(10019)
    slodata = EnumValue(10025)
    image = EnumValue(1073741824)
    measurements = EnumValue(7)
    studyname = EnumValue(9000)
    device = EnumValue(9001, doc='eg. Heidelberg Retina Angiograph HRA')
    examined_structure = EnumValue(9005, doc='eg. Retina')
    scanpattern = EnumValue(9006, doc='eg. OCT ART Volume')
    enface_modality = EnumValue(9007, doc='?enface modality eg Infra-Red IR')
    oct_modality = EnumValue(9008,
                             doc='?OCT modality eg OCT-OCT or OCT-Angio?')


class TypeMixin:

    @property
    def type_id(self) -> int:
        return int(self.__class__.__name__.lstrip('Type'))


@dataclasses.dataclass
class Type10004(DataclassMixin, TypeMixin):
    """B-scan Metadata Metadata for a single B-scan.

    Size: 428 bytes

    Notes:
    The current Bscan-Meta structure builds on the implementation found in [LibE2E](https://github.com/neurodial/LibE2E/blob/d26d2d9db64c5f765c0241ecc22177bb0c440c87/E2E/dataelements/bscanmetadataelement.cpp#L75).
    """
    unknown0: int = csfield(cs.Int32ul, )
    size_y: int = csfield(cs.Int32ul, doc='Bscan height')
    size_x: int = csfield(cs.Int32ul, doc='Bscan width')
    start_x: float = csfield(cs.Float32l, doc='Start X coordinate of Bscan')
    start_y: float = csfield(cs.Float32l, doc='Start Y coordinate of Bscan')
    end_x: float = csfield(cs.Float32l, doc='End X coordinate of Bscan')
    end_y: float = csfield(cs.Float32l, doc='End Y coordinate of Bscan')
    zero1: int = csfield(cs.Int32ul)
    unknown1: float = csfield(cs.Float32l)
    scale_y: float = csfield(cs.Float32l, doc='Scale of Bscan y-axis (height)')
    unknown2: float = csfield(cs.Float32l)
    zero2: int = csfield(cs.Int32ul)
    unknown3: t.List[float] = csfield(cs.Array(2, cs.Float32l))
    zero3: int = csfield(cs.Int32ul)
    imgSizeWidth: int = csfield(
        cs.Int32ul,
        doc=
        'Might differ from size_x, and is probably the number of filled A-scans in the Bscan. Bscans sometimes have empty A-scans at the ends.'
    )
    n_bscans: int = csfield(cs.Int32ul,
                            doc='Number of Bscans in the respective volume')
    aktImage: int = csfield(cs.Int32ul,
                            doc='Index of the current Bscan in the volume')
    scan_pattern: int = csfield(
        cs.Int32ul,
        doc=
        'Scan pattern of the volume. <br>**Does this corresponds to the scan pattterns in VOL and XML export?**'
    )
    center_x: float = csfield(cs.Float32l,
                              doc='Exactly the average of start_x and end_x.')
    center_y: float = csfield(cs.Float32l,
                              doc='Exactly the average of start_y and end_y.')
    unknown4: int = csfield(cs.Int32ul, doc='Maybe the UTC bias?')
    acquisitionTime: int = csfield(DateTime, doc='Acquisition time of Bscan')
    numAve: int = csfield(
        cs.Int32ul,
        doc=
        'Number of averages according to LibE2E<br>**Not coherent with XML export**'
    )
    quality: float = csfield(
        cs.Float32l,
        doc=
        'Quality according to LibE2E<br>**Does not match the quality value in the XML export which is an integer compared to a float here with value 0.84 for a complete volume. Maybe this is the focus length, at least it is similar to the value given in the XML (0.87)**'
    )
    unknown5: float = csfield(cs.Float32l)


type10004_format = DataclassStruct(Type10004)


@dataclasses.dataclass
class Type7(DataclassMixin, TypeMixin):
    """Measurements Global measurements of the eye.

    Size: 68 bytes

    Notes:
    """
    eye_side: LateralityEnum = csfield(TEnum(cs.Int8ul, LateralityEnum))
    c_curve_mm: float = csfield(cs.Float64l)
    refraction_dpt: float = csfield(cs.Float64l)
    cylinder_dpt: float = csfield(cs.Float64l)
    axis_deg: float = csfield(cs.Float64l)
    pupil_size_mm: float = csfield(cs.Float64l)
    iop_mmHg: float = csfield(cs.Float64l)
    vfield_mean: float = csfield(cs.Float64l)
    vfield_var: float = csfield(cs.Float64l)
    corrective_lens: int = csfield(cs.Int16ul)
    rest: bytes = csfield(cs.Bytes(1))


type7_format = DataclassStruct(Type7)


@dataclasses.dataclass
class Type1073741824(DataclassMixin, TypeMixin):
    """Image data Stores various kinds of images.

    Size: variable

    Notes:
    Different kinds of images are stored in this structure. Currently we know the following types:

    * 33620481: LocalizerNIR (`int8u`)
    * 35652097: Bscan (`float16u`)

    The custom `float16u` used to store the Bscan data, has no sign, a 6-bit exponent und 10-bit mantissa.
    """
    size: int = csfield(cs.Int32ul, doc='Size of the data')
    type: int = csfield(cs.Int32ul, doc='Type of the data')
    n_values: int = csfield(cs.Int32ul, doc='Number of values in the data')
    height: int = csfield(cs.Int32ul, doc='Height of the image')
    width: int = csfield(cs.Int32ul, doc='Width of the image')
    data: t.Any = csfield(cs.Switch(cs.this.type, {
        33620481: LocalizerNIR,
        35652097: Bscan,
    },
                                    default=cs.Bytes(cs.this.size)),
                          doc='Image data')


type1073741824_format = DataclassStruct(Type1073741824)


@dataclasses.dataclass
class Type9(DataclassMixin, TypeMixin):
    """Patient data Personal data of the patient.

    Size: 131 bytes

    Notes:
    """
    firstname: str = csfield(cs.PaddedString(31, 'ascii'))
    surname: str = csfield(cs.PaddedString(66, 'ascii'))
    birthdate: int = csfield(cs.Int32ul)
    sex: str = csfield(cs.PaddedString(1, 'ascii'))
    patient_id: str = csfield(cs.PaddedString(25, 'ascii'))


type9_format = DataclassStruct(Type9)


@dataclasses.dataclass
class Type10019(DataclassMixin, TypeMixin):
    """Layer Annotation Stores one layer for one Bscan.

    Size: variable

    Notes:
    """
    unknown0: int = csfield(cs.Int32ul)
    id: int = csfield(cs.Int32ul, doc='ID of the layer')
    unknown1: int = csfield(cs.Int32ul)
    width: int = csfield(cs.Int32ul, doc='Width of the layer')
    data: t.List[float] = csfield(Segmentation, doc='Layer annotation data')


type10019_format = DataclassStruct(Type10019)


@dataclasses.dataclass
class Type11(DataclassMixin, TypeMixin):
    """Type 11.

    Size: 27 bytes

    Notes:
    We don't know what this data is used for, only that the 15th byte indicates the laterality of the eye.
    """
    unknown: bytes = csfield(cs.Bytes(14))
    laterality: LateralityEnum = csfield(TEnum(cs.Int8ul, LateralityEnum))


type11_format = DataclassStruct(Type11)


@dataclasses.dataclass
class Type59(DataclassMixin, TypeMixin):
    """Type 59.

    Size: 27 bytes

    Notes:
    We don't know what this data is used for, only that the 14th byte indicates the laterality of the eye.
    """
    unknown: bytes = csfield(cs.Bytes(14))
    laterality: LateralityEnum = csfield(TEnum(cs.Int8ul, LateralityEnum))


type59_format = DataclassStruct(Type59)


@dataclasses.dataclass
class Type3(DataclassMixin, TypeMixin):
    """Type 3.

    Size: 96 bytes

    Notes:
    We don't know what this data is used for, only that the 5th byte indicates the laterality of the eye.
    """
    unknown: bytes = csfield(cs.Bytes(4))
    laterality: LateralityEnum = csfield(TEnum(cs.Int8ul, LateralityEnum))


type3_format = DataclassStruct(Type3)


@dataclasses.dataclass
class Type5(DataclassMixin, TypeMixin):
    """Type 5.

    Size: 59 bytes

    Notes:
    We don't know what this data is used for, only that the 3rd byte indicates the laterality of the eye.
    """
    unknown: bytes = csfield(cs.Bytes(2))
    laterality: LateralityEnum = csfield(TEnum(cs.Int8ul, LateralityEnum))


type5_format = DataclassStruct(Type5)


@dataclasses.dataclass
class Type10013(DataclassMixin, TypeMixin):
    """Type 10013.

    Size: variable

    Notes:
    """
    unknown: bytes = csfield(cs.Bytes(12))
    n_bscans: int = csfield(cs.Int32ul)


type10013_format = DataclassStruct(Type10013)


@dataclasses.dataclass
class Type10012(DataclassMixin, TypeMixin):
    """Type 10012.

    Size: variable

    Notes:
    """
    unknown0: bytes = csfield(cs.Bytes(28))
    value_1: float = csfield(cs.Float32l)
    unknown1: bytes = csfield(cs.Bytes(1))
    value_2: float = csfield(cs.Float32l)


type10012_format = DataclassStruct(Type10012)


@dataclasses.dataclass
class Type10010(DataclassMixin, TypeMixin):
    """Type 10010.

    Size: variable

    Notes:
    """
    unknown: bytes = csfield(cs.Bytes(12))
    n_bscans: int = csfield(cs.Int32ul)


type10010_format = DataclassStruct(Type10010)


@dataclasses.dataclass
class Type9000(DataclassMixin, TypeMixin):
    """Studyname Name of the study/visit.

    Size: 264 bytes

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type9000_format = DataclassStruct(Type9000)


@dataclasses.dataclass
class Type9006(DataclassMixin, TypeMixin):
    """Scan pattern Bscan pattern used for the aquisition.

    Size: 520 bytes

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type9006_format = DataclassStruct(Type9006)


@dataclasses.dataclass
class Type9001(DataclassMixin, TypeMixin):
    """Device Name of the used device.

    Size: 776 bytes

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type9001_format = DataclassStruct(Type9001)


@dataclasses.dataclass
class Type9005(DataclassMixin, TypeMixin):
    """Examined structure Name of the examined structure.

    Size: 264 bytes

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type9005_format = DataclassStruct(Type9005)


@dataclasses.dataclass
class Type9007(DataclassMixin, TypeMixin):
    """Enface Modality Modality of the enface (eg IR)

    Size: 520 bytes

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type9007_format = DataclassStruct(Type9007)


@dataclasses.dataclass
class Type9008(DataclassMixin, TypeMixin):
    """OCT Modality Modality of the OCT (eg OCT)

    Size: 520 bytes

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type9008_format = DataclassStruct(Type9008)


@dataclasses.dataclass
class Type17(DataclassMixin, TypeMixin):
    """Diagnose data.

    Size: variable

    Notes:
    """
    n_strings: int = csfield(cs.Int32ul)
    string_size: int = csfield(cs.Int32ul)
    text: t.List[str] = csfield(
        cs.Array(cs.this.n_strings,
                 cs.PaddedString(cs.this.string_size, 'utf16')))


type17_format = DataclassStruct(Type17)


@dataclasses.dataclass
class Type10025(DataclassMixin, TypeMixin):
    """Localizer Metadata.

    Size: 100 bytes

    Notes:
    """
    unknown: bytes = csfield(cs.Bytes(24))
    windate: int = csfield(DateTime)
    transform: t.List[float] = csfield(
        cs.Array(6, cs.Float32l), doc='Parameters of affine transformation')


type10025_format = DataclassStruct(Type10025)

item_switch = cs.Switch(cs.this.header.type, {
    TypesEnum.patient: type9_format,
    TypesEnum.laterality: type11_format,
    TypesEnum.bscanmeta: type10004_format,
    TypesEnum.layer_annotation: type10019_format,
    TypesEnum.image: type1073741824_format,
    TypesEnum.measurements: type7_format,
    TypesEnum.studyname: type9000_format,
    TypesEnum.scanpattern: type9006_format,
    TypesEnum.device: type9001_format,
    TypesEnum.examined_structure: type9005_format,
    TypesEnum.enface_modality: type9007_format,
    TypesEnum.oct_modality: type9008_format,
    TypesEnum.diagnose: type17_format,
    TypesEnum.slodata: type10025_format,
},
                        default=cs.Bytes(cs.this.header.size))


@dataclasses.dataclass
class ContainerHeader(DataclassMixin):
    """Container header data.

    Size: 60 bytes

    Notes:
    """
    magic3: str = csfield(cs.PaddedString(12, 'ascii'))
    unknown0: int = csfield(cs.Int32ul)
    header_pos: int = csfield(cs.Int32ul, doc='Position of the header')
    pos: int = csfield(cs.Int32ul, doc='Position of the data')
    size: int = csfield(cs.Int32ul, doc='Size of the container')
    unknown1: int = csfield(
        cs.Int32ul,
        doc="Always 0 (b'\\x00\\x00\\x00\\x00')? At least in our data")
    patient_id: int = csfield(cs.Int32sl, doc='Patient ID')
    study_id: int = csfield(cs.Int32sl, doc='Study ID')
    series_id: int = csfield(cs.Int32sl, doc='Series ID')
    slice_id: int = csfield(
        cs.Int32sl,
        doc='Slice ID, has to be divided by 2 to get the correct slice number')
    ind: int = csfield(
        cs.Int16ul,
        doc=
        "Takes only values 65333, 0 and 1 (b'\xff\xff', b'\x00\x00', b'\x01\x00') at least in our data - 0 for enface and 1 for bscan for image containers"
    )
    unknown2: int = csfield(cs.Int16ul,
                            doc="Always 0 (b'\x00\x00')? At least in our data")
    type: TypesEnum = csfield(TEnum(cs.Int32ul, TypesEnum),
                              doc='Type ID of the contained data')
    unknown3: int = csfield(
        cs.Int32ul,
        doc=
        'Large integer that increases in steps of folder header size (=44) Maybe the folder header position in HEYEX database not in this file? - Possibly related to the folder header unknown4 value'
    )


containerheader_format = DataclassStruct(ContainerHeader)


@dataclasses.dataclass
class DataContainer(DataclassMixin):
    """Data container.

    Size: variable

    Notes:
    """
    header: ContainerHeader = csfield(containerheader_format)
    item: t.Any = csfield(
        item_switch,
        doc=
        'There are many kinds of DataItems indicated by different type IDs in the folder/container header'
    )


datacontainer_format = DataclassStruct(DataContainer)


@dataclasses.dataclass
class FolderHeader(DataclassMixin):
    """Folder header.

    Size: 44 bytes

    Notes:
    """
    pos: int = csfield(
        cs.Int32ul,
        doc=
        'Position of the folder (In a chunk all 512 folder headers are stored sequentially, refering to the data that follows after this header block)'
    )
    start: int = csfield(
        cs.Int32ul,
        doc='Start of the data container, after the header block in the chunk')
    size: int = csfield(cs.Int32ul, doc='Size of the data container')
    unknown0: int = csfield(cs.Int32ul,
                            doc="Always 0 (b'\x00\x00')? At leat in our data")
    patient_id: int = csfield(cs.Int32sl, doc='Patient ID')
    study_id: int = csfield(cs.Int32sl, doc='Study ID')
    series_id: int = csfield(cs.Int32sl, doc='Series ID')
    slice_id: int = csfield(
        cs.Int32sl,
        doc='Slice ID, has to be divided by 2 to get the correct slice number')
    ind: int = csfield(cs.Int16ul,
                       doc='0 for enface and 1 for bscan for image containers')
    unknown1: int = csfield(cs.Int16ul)
    type: TypesEnum = csfield(TEnum(cs.Int32ul, TypesEnum),
                              doc='Type ID of the contained data')
    unknown3: int = csfield(
        cs.Int32ul,
        doc=
        'Large integer possibly related to data_container.unknown5. Maybe the position in HEYEX DB?'
    )


folderheader_format = DataclassStruct(FolderHeader)


@dataclasses.dataclass
class Header(DataclassMixin):
    """Chunk header.

    Size: 52 bytes

    Notes:
    """
    magic2: str = csfield(cs.PaddedString(12, 'ascii'))
    version: int = csfield(cs.Int32ul)
    unknown0: t.List[int] = csfield(cs.Array(10, cs.Int16ul))
    num_entries: int = csfield(cs.Int32ul,
                               doc='Number of entries in the chunk')
    current: int = csfield(cs.Int32ul, doc='Position of the current chunk')
    prev: int = csfield(cs.Int32ul, doc='Position of the previous chunk')
    unknown1: int = csfield(cs.Int32ul)


header_format = DataclassStruct(Header)


@dataclasses.dataclass
class Chunk(DataclassMixin):
    """Data chunk.

    Size: variable

    Notes:
    Every chunk has a header similar to the file header. A chunk then holds the headers of all contained folders sequentially, followed by data containers, that are referenced by the folder headers. A chunk can contain folders with data of different patients, studies, series, slices and types. Each folder contains data for a single (patient, study, series, slice, type) combination which is given in the folder header as well as the data container header. For the last chunk to have 512 folders, empty folders of type=0 are appended.
    """
    chunk_header: Header = csfield(
        header_format,
        doc=
        'Each chunk refers to the start position of the previous chunk (`prev` field)'
    )
    folders: t.List[FolderHeader] = csfield(
        cs.Array(cs.this.chunk_header.num_entries, folderheader_format),
        doc=
        'In the data we have seen each chunk has 512 folders with headers of size 44'
    )
    jump: int = csfield(
        cs.Seek(cs.this.folders[-1].start + cs.this.folders[-1].size +
                datacontainer_format.header.sizeof()))


chunk_format = DataclassStruct(Chunk)


@dataclasses.dataclass
class Version(DataclassMixin):
    """Version header.

    Size: 36 bytes

    Notes:
    """
    name: str = csfield(cs.PaddedString(12, 'ascii'),
                        doc='Name of the version')
    version: int = csfield(cs.Int32ul, doc='Verion of the file')
    unknown0: t.List[int] = csfield(cs.Array(10, cs.Int16ul))


version_format = DataclassStruct(Version)


@dataclasses.dataclass
class E2EFormat(DataclassMixin):
    """E2E file format.

    Size: variable

    Notes:
    An E2E file starts with a version structure, followed by a header structure. After that the data comes in chunks of 512 folders.
    """
    version: Version = csfield(version_format)
    header: Header = csfield(
        header_format,
        doc=
        'The `prev` field in the main header refers to the start position of the last chunk'
    )
    chunks: t.List[Chunk] = csfield(
        cs.GreedyRange(chunk_format),
        doc='The number and size of the chunks depends on the data')


e2e_format = DataclassStruct(E2EFormat)

__e2efile_structures__ = [
    E2EFormat,
    Version,
    Chunk,
    Header,
    FolderHeader,
    DataContainer,
    ContainerHeader,
]
__all_types__ = [
    Type3, Type5, Type7, Type9, Type11, Type17, Type59, Type9000, Type9001,
    Type9005, Type9006, Type9007, Type9008, Type10004, Type10010, Type10012,
    Type10013, Type10019, Type10025, Type1073741824
]
