import eyepy as ep


def test_eyevolumemeta_json_conversion():
    meta_dict = {
        'scale_z':
        1,
        'scale_x':
        1,
        'scale_y':
        1,
        'scale_unit':
        'px',
        'bscan_meta': [{
            'start_pos': (0, 0),
            'end_pos': [10, 0],
            'pos_unit': 'px'
        }, {
            'start_pos': (0, 1),
            'end_pos': [10, 1],
            'pos_unit': 'px'
        }]
    }

    evm = ep.EyeVolumeMeta.from_dict(meta_dict)
    assert evm.as_dict() == meta_dict
