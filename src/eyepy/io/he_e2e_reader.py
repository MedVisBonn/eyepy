import logging
from typing import List

import construct as cs
import numpy as np

from eyepy.core.eyeenface import EyeEnface
from eyepy.core.eyemeta import EyeBscanMeta
from eyepy.core.eyemeta import EyeEnfaceMeta
from eyepy.core.eyemeta import EyeVolumeMeta
from eyepy.core.eyevolume import EyeVolume
from eyepy.io.he_vol_reader import SEG_MAPPING
from eyepy.io.utils import _compute_localizer_oct_transform

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
        return self.LUT[
            np.ndarray(
                buffer=obj, dtype=np.uint16, shape=(context.width, context.height)
            )
        ]

    def _encode(self, obj, context, path):
        return obj.tobytes()


class LocalizerNIRAdapter(cs.Adapter):
    def _decode(self, obj, context, path):
        return np.ndarray(
            buffer=obj, dtype="uint8", shape=(context.height, context.width)
        )

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

image_structure = cs.Struct(
    "size" / cs.Int32un,
    "type" / cs.Int32un,
    "n_values" / cs.Int32un,
    "width" / cs.Int32un,
    "height" / cs.Int32un,
    "data"
    / cs.Switch(
        cs.this.type,
        {33620481: LocalizerNIR, 35652097: Bscan},
        default=cs.Bytes(cs.this.size),
    ),
)
patient_structure = cs.Struct(
    "first_name" / cs.PaddedString(31, "ascii"),
    "surname" / cs.PaddedString(66, "ascii"),
    "birthdate" / cs.Int32un,
    "sex" / cs.PaddedString(1, "ascii"),
    "patient_id" / cs.PaddedString(25, "ascii"),
)
laterality_structure = cs.Struct(
    "unknown" / cs.Array(14, cs.Int8un),
    "laterality" / cs.Int8un,
    "unknown2" / cs.Int8un,
)
layer_structure = cs.Struct(
    "unknown0" / cs.Int32un,
    "id" / cs.Int32un,
    "unknown1" / cs.Int32un,
    "width" / cs.Int32un,
    "data" / Segmentation,
)

# following the spec from
# https://github.com/neurodial/LibE2E/blob/d26d2d9db64c5f765c0241ecc22177bb0c440c87/E2E/dataelements/bscanmetadataelement.cpp#L75
bscanmeta_structure = cs.Struct(
    "unknown1" / cs.Int32un,
    "imgSizeX" / cs.Int32un,
    "imgSizeY" / cs.Int32un,
    "posX1" / cs.Float32l,
    "posX2" / cs.Float32l,
    "posY1" / cs.Float32l,
    "posY2" / cs.Float32l,
    "zero1" / cs.Int32un,
    "unknown2" / cs.Float32l,
    "scaley" / cs.Float32l,
    "unknown3" / cs.Float32l,
    "zero2" / cs.Int32un,
    "unknown4" / cs.Array(2, cs.Float32l),
    "zero3" / cs.Int32un,
    "imgSizeWidth" / cs.Int32un,
    "numImages" / cs.Int32un,
    "aktImage" / cs.Int32un,
    "scanType" / cs.Int32un,
    "centerPosX" / cs.Float32l,
    "centerPosY" / cs.Float32l,
    "unknown5" / cs.Int32un,
    "acquisitionTime" / cs.Int64un,
    "numAve" / cs.Int32un,
    "imgQuality" / cs.Float32l,
)

header_structure = cs.Struct(
    "magic1" / cs.PaddedString(12, "ascii"),
    "version" / cs.Int32un,
    "unknown" / cs.Array(10, cs.Int16un),
)
folder_header_structure = cs.Struct(
    "magic2" / cs.PaddedString(12, "ascii"),
    "version" / cs.Int32un,
    "unknown" / cs.Array(10, cs.Int16un),
    "num_entries" / cs.Int32un,
    "current" / cs.Int32un,
    "prev" / cs.Int32un,
    "unknown3" / cs.Int32un,
)

data_structure = cs.Struct(
    "data_header"
    / cs.Struct(
        "magic3" / cs.PaddedString(12, "ascii"),
        "unknown" / cs.Int32un,
        "unknown2" / cs.Int32un,
        "pos" / cs.Int32un,
        "size" / cs.Int32un,
        "unknown3" / cs.Int32un,
        "patient_id" / cs.Int32un,
        "study_id" / cs.Int32un,
        "series_id" / cs.Int32un,
        "slice_id" / cs.Int32sn,
        "ind" / cs.Int16un,
        "unknown4" / cs.Int16un,
        "type"
        / cs.Enum(
            cs.Int32un,
            patient=9,
            laterality=11,
            bscanmeta=10004,
            layer=10019,
            image=1073741824,
        ),
        "unknown5" / cs.Int32un,
    ),
    "data"
    / cs.Switch(
        cs.this.data_header.type,
        {
            "patient": patient_structure,
            "laterality": laterality_structure,
            "bscanmeta": bscanmeta_structure,
            "layer": layer_structure,
            "image": image_structure,
        },
        default=cs.Bytes(cs.this.data_header.size),
    ),
)

sub_folder_structure = cs.Struct(
    "header"
    / cs.Struct(
        "pos" / cs.Int32un,
        "start" / cs.Int32un,
        "size" / cs.Int32un,
        "unknown" / cs.Int32un,
        "patient_id" / cs.Int32un,
        "study_id" / cs.Int32un,
        "series_id" / cs.Int32un,
        "slice_id" / cs.Int32sn,
        "unknown2" / cs.Int16un,
        "unknown3" / cs.Int16un,
        "type" / cs.Int32un,
        "unknown4" / cs.Int32un,
    ),
    "data"
    / cs.If(
        cs.this.header.start > cs.this.header.pos,
        cs.Pointer(cs.this.header.start, data_structure),
    ),
)

folder_structure = cs.Struct(
    "folder_header" / folder_header_structure,
    "sub_folders" / cs.Array(cs.this.folder_header.num_entries, sub_folder_structure),
    "Jump"
    / cs.Seek(
        cs.this.sub_folders[-1].header.start
        + cs.this.sub_folders[-1].header.size
        + data_structure.data_header.sizeof()
    ),
)

e2e_format = cs.Struct(
    "main_header" / header_structure,
    "folder_header" / folder_header_structure,
    "folders" / cs.GreedyRange(folder_structure),
)


class HeE2eReader:
    def __init__(self, path):
        self.path = path
        with open(self.path, "rb") as e2e_file:
            self.parsed_file = e2e_format.parse_stream(e2e_file)

    @property
    def volume(self) -> EyeVolume:
        ## Check if scan is a volume scan
        if not self.parsed_file.scan_pattern in [1, 3, 4]:
            msg = f"Only volumes with ScanPattern 1, 3 or 4 are supported. The ScanPattern is {self.parsed_file.scan_pattern} which might lead to exceptions or unexpected behaviour."
            logger.warning(msg)

        data = np.stack([bscan.data for bscan in self.parsed_file.bscans], axis=0)
        volume_meta = self.meta
        localizer = self.localizer
        volume = EyeVolume(
            data=data,
            meta=volume_meta,
            localizer=localizer,
            transformation=_compute_localizer_oct_transform(
                volume_meta, localizer.meta, data.shape
            ),
        )

        layer_height_maps = self.layers
        for name, i in SEG_MAPPING.items():
            volume.add_layer_annotation(layer_height_maps[i], name=name)

        return volume

    @property
    def layers(self):
        return np.stack(
            [b.layer_segmentations for b in self.parsed_file.bscans], axis=1
        )

    @property
    def localizer_meta(self) -> EyeEnfaceMeta:
        return EyeEnfaceMeta(
            scale_x=self.parsed_file.scale_x_slo,
            scale_y=self.parsed_file.scale_y_slo,
            scale_unit="mm",
            modality="NIR",
            laterality=self.parsed_file.scan_position,
            field_size=self.parsed_file.field_size_slo,
            scan_focus=self.parsed_file.scan_focus,
            visit_date=self.parsed_file.visit_date,
            exam_time=self.parsed_file.exam_time,
        )

    @property
    def localizer(self) -> EyeEnface:
        localizer = self.parsed_file.localizer
        return EyeEnface(localizer, self.localizer_meta)

    @property
    def bscan_meta(self) -> List[EyeBscanMeta]:
        return [
            EyeBscanMeta(
                quality=b.quality,
                start_pos=(b.start_x, b.start_y),
                end_pos=(b.end_x, b.end_y),
                pos_unit="mm",
            )
            for b in self.parsed_file.bscans
        ]

    @property
    def meta(self) -> EyeVolumeMeta:
        bscan_meta = self.bscan_meta
        return EyeVolumeMeta(
            scale_x=self.parsed_file.scale_x,
            scale_y=self.parsed_file.scale_y,
            scale_z=self.parsed_file.distance,
            scale_unit="mm",
            laterality=self.parsed_file.scan_position,
            visit_date=self.parsed_file.visit_date,
            exam_time=self.parsed_file.exam_time,
            bscan_meta=bscan_meta,
            intensity_transform="vol",
        )
