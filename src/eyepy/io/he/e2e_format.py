from collections import defaultdict
import logging
from typing import List, Tuple, Union

import construct as cs
import numpy as np

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.eyevolume import EyeVolume
from eyepy.io.utils import _compute_localizer_oct_transform
from eyepy.io.utils import get_bscan_spacing

logger = logging.getLogger(__name__)


class BscanAdapter(cs.Adapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LUT = self._make_LUT()

    def _uint16_to_ufloat16(self, uint_16):
        """Implementation of custom float type used in .e2e files.
        Notes:
            Custom float is a floating point type with no sign, 6-bit exponent, and 10-bit mantissa.
        Args:
            uint16 (int):
        Returns:
            float
        """
        bits = "{0:016b}".format(uint_16)[::-1]
        # get mantissa and exponent
        mantissa = bits[:10]
        exponent = bits[10:]
        exponent = exponent[::-1]

        # convert to decimal representations
        mantissa_sum = 1 + int(mantissa, 2) / pow(2, 10)
        exponent_sum = int(exponent, 2) - 63
        decimal_value = mantissa_sum * np.float_power(2, exponent_sum)
        return decimal_value

    def _make_LUT(self):
        LUT = []
        for i in range(0, pow(2, 16)):
            LUT.append(self._uint16_to_ufloat16(i))
        return np.array(LUT)

    def _decode(self, obj, context, path):
        return self.LUT[np.ndarray(buffer=obj,
                                   dtype=np.uint16,
                                   shape=(context.height, context.width))]

    def _encode(self, obj, context, path):
        return obj.tobytes()


class LocalizerNIRAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj,
                          dtype="uint8",
                          shape=(context.height, context.width))

    def _encode(self, obj, context, path):
        return obj.tobytes()


class SegmentationAdapter(cs.Adapter):

    def _decode(self, obj, context, path):
        return np.ndarray(buffer=obj, dtype="float32", shape=(context.width))

    def _encode(self, obj, context, path):
        return obj.tobytes()


LocalizerNIR = LocalizerNIRAdapter(cs.Bytes(cs.this.n_values))
Segmentation = SegmentationAdapter(cs.Bytes(cs.this.width * 4))
Bscan = BscanAdapter(cs.Bytes(cs.this.n_values * 2))

image_structure = cs.Struct(size=cs.Int32ul,
                            type=cs.Int32ul,
                            n_values=cs.Int32ul,
                            height=cs.Int32ul,
                            width=cs.Int32ul,
                            data=cs.Switch(cs.this.type, {
                                33620481: LocalizerNIR,
                                35652097: Bscan,
                            },
                                           default=cs.Bytes(cs.this.size)))

patient_structure = cs.LazyStruct(first_name=cs.PaddedString(31, "ascii"),
                                  surname=cs.PaddedString(66, "ascii"),
                                  birthdate=cs.Int32ul,
                                  sex=cs.PaddedString(1, "ascii"),
                                  patient_id=cs.PaddedString(25, "ascii"),
                                  rest=cs.Bytes(4))
laterality_structure = cs.Struct(unknown0=cs.Bytes(14),
                                 laterality=cs.Enum(cs.Int8ul, OS=76, OD=82),
                                 rest=cs.Bytes(12))

layer_structure = cs.Struct(unknown0=cs.Int32ul,
                            id=cs.Int32ul,
                            unknown1=cs.Int32ul,
                            width=cs.Int32ul,
                            data=Segmentation)

bscanmeta_structure = cs.Struct(
    unknown0=cs.Int32ul,
    size_y=cs.Int32ul,
    size_x=cs.Int32ul,
    start_x=cs.Float32l,
    start_y=cs.Float32l,
    end_x=cs.Float32l,
    end_y=cs.Float32l,
    zero1=cs.Int32ul,
    unknown1=cs.Float32l,
    scale_y=cs.Float32l,
    unknown2=cs.Float32l,
    zero2=cs.Int32ul,
    unknown3=cs.Array(2, cs.Float32l),
    zero3=cs.Int32ul,
    imgSizeWidth=cs.Int32ul,
    n_bscans=cs.Int32ul,
    aktImage=cs.Int32ul,
    scan_pattern=cs.Int32ul,
    center_x=cs.Float32l,
    center_y=cs.Float32l,
    unknown4=cs.Int32ul,
    acquisitionTime=cs.Int64ul,
    numAve=cs.Int32ul,
    quality=cs.Float32l,
)

measurements = cs.Struct(eye_side=cs.PaddedString(1, "ascii"),
                         c_curve_mm=cs.Float64l,
                         refraction_dpt=cs.Float64l,
                         cylinder_dpt=cs.Float64l,
                         axis_deg=cs.Float64l,
                         pupil_size_mm=cs.Float64l,
                         iop_mmHg=cs.Float64l,
                         vfield_mean=cs.Float64l,
                         vfield_var=cs.Float64l,
                         corrective_lens=cs.Int16ul,
                         rest=cs.Bytes(1))

textdata = cs.Struct(
    n_strings=cs.Int32ul,
    string_size=cs.Int32ul,
    text=cs.Array(cs.this.n_strings,
                  cs.PaddedString(cs.this.string_size, "utf16")),
)

slodata = cs.Struct(
    unknown0=cs.Bytes(24),
    windate=cs.Int64ul,
    transform=cs.Array(6, cs.Float32l),
    unknown1=cs.Bytes(cs.this._.header.size - 80),
)

types = cs.Enum(
    cs.Int32un,
    patient=9,
    laterality=11,
    diagnose=17,
    bscanmeta=10004,
    layer_annotation=10019,
    slodata=10025,
    image=1073741824,
    measurements=7,
    studyname=9000,
    device=9001,  # eg. Heidelberg Retina Angiograph HRA
    examined_structure=9005,  # eg. Retina
    scanpattern=9006,  # eg. OCT ART Volume
    enface_modality=9007,  # ?enface modality eg Infra-Red IR
    oct_modality=9008,  # ?OCT modality eg OCT-OCT or OCT-Angio?
    empty_folder=0,  # Data of size 0, so probably not important
)

item_switch = cs.Switch(cs.this.header.type, {
    "patient": patient_structure,
    "laterality": laterality_structure,
    "bscanmeta": bscanmeta_structure,
    "layer_annotation": layer_structure,
    "image": image_structure,
    "measurements": measurements,
    "studyname": textdata,
    "scanpattern": textdata,
    "device": textdata,
    "examined_structure": textdata,
    "enface_modality": textdata,
    "oct_modality": textdata,
    "diagnose": textdata,
    "slodata": slodata,
},
                        default=cs.Bytes(cs.this.header.size))

raw_item_switch = cs.RawCopy(item_switch)

container_header_structure = cs.Struct(
    magic3=cs.PaddedString(12, "ascii"),
    unknown0=cs.Int32ul,
    header_pos=cs.Int32ul,
    pos=cs.Int32ul,
    size=cs.Int32ul,
    # Always 0 (b'\x00\x00\x00\x00')? At leat in our data
    unknown1=cs.Int32ul,
    patient_id=cs.Int32sl,
    study_id=cs.Int32sl,
    series_id=cs.Int32sl,
    # Has to be divided by 2 to get the correct slice number
    slice_id=cs.Int32sl,
    # Takes only values 65333, 0 and 1 (b'\xff\xff', b'\x00\x00', b'\x01\x00') at least in our data
    # 0 for enface and 1 for bscan for image containers
    ind=cs.Int16ul,
    # Always 0 (b'\x00\x00')? At leat in our data
    unknown2=cs.Int16ul,
    type=types,
    # Large integer that increases in steps of folder header size (=44) Maybe the folder header position in HEYEX database not in this file?
    # Possibly related to the folder header unknown4 value
    unknown3=cs.Int32ul,
)

data_container_structure = cs.Struct(header=container_header_structure,
                                     item=item_switch)

folder_header_structure = cs.Struct(
    # Position of the folder (In a chunk all 512 folder headers are stored sequentially, refering to the data that follows after this header block)
    pos=cs.Int32ul,
    # Start of the data container, after the header block in the chunk
    start=cs.Int32ul,
    # Size of the data container
    size=cs.Int32ul,
    # Always 0 (b'\x00\x00')? At leat in our data
    unknown0=cs.Int32ul,
    patient_id=cs.Int32sl,
    study_id=cs.Int32sl,
    series_id=cs.Int32sl,
    slice_id=cs.Int32sl,
    # 0 for enface and 1 for bscan for image containers
    ind=cs.Int16un,
    unknown1=cs.Int16ul,
    type=types,
    # Large integer possibly related to data_container.unknown5. Maybe the position in HEYEX DB?
    unknown2=cs.Int32ul,
)

folder_structure = cs.Struct(
    header=folder_header_structure,
    data_container=cs.If(
        cs.this.header.start > cs.this.header.
        pos,  # Sometimes the start is a small int like 0 or 3 which does not refer to a data container. Only allow start if it is after the header block.
        cs.Pointer(cs.this.header.start, data_container_structure),
    ))

header_structure = cs.Struct(
    magic2=cs.PaddedString(12, "ascii"),
    version=cs.Int32ul,
    unknown0=cs.Array(10, cs.Int16ul),
    num_entries=cs.Int32ul,
    current=cs.Int32ul,
    prev=cs.Int32ul,
    unknown1=cs.Int32ul,
)

chunk_structure = cs.Struct(
    chunk_header=header_structure,
    folders=cs.Array(cs.this.chunk_header.num_entries, folder_structure),
    jump=cs.Seek(cs.this.folders[-1].header.start +
                 cs.this.folders[-1].header.size +
                 data_container_structure.header.sizeof()))

version_structure = cs.Struct(
    name=cs.PaddedString(12, "ascii"),
    version=cs.Int32ul,
    unknown0=cs.Array(10, cs.Int16ul),
)

e2e_format = cs.Struct(version=version_structure,
                       header=header_structure,
                       chunks=cs.GreedyRange(chunk_structure))
