"""Tests for EyeMeta and EyeEnfaceMeta copy functionality."""

import datetime
import pytest

from eyepy.core.eyemeta import EyeMeta, EyeEnfaceMeta


class TestEyeMetaCopy:
    """Test suite for EyeMeta.copy() method."""
    
    def test_copy_creates_new_instance(self):
        """Test that copy() creates a new instance, not a reference."""
        meta = EyeMeta(key1='value1', key2='value2')
        copied = meta.copy()
        
        assert copied is not meta
        assert isinstance(copied, EyeMeta)
    
    def test_copy_preserves_data(self):
        """Test that copy() preserves all metadata."""
        meta = EyeMeta(key1='value1', key2=42, key3=3.14)
        copied = meta.copy()
        
        assert copied['key1'] == 'value1'
        assert copied['key2'] == 42
        assert copied['key3'] == 3.14
    
    def test_copy_is_independent(self):
        """Test that modifying the copy doesn't affect the original."""
        meta = EyeMeta(key1='value1', key2='value2')
        copied = meta.copy()
        
        # Modify the copy
        copied['key1'] = 'modified'
        copied['key3'] = 'new_key'
        
        # Original should be unchanged
        assert meta['key1'] == 'value1'
        assert 'key3' not in meta
    
    def test_copy_with_datetime(self):
        """Test that copy() works with datetime objects."""
        now = datetime.datetime.now()
        meta = EyeMeta(timestamp=now, description='test')
        copied = meta.copy()
        
        assert copied['timestamp'] == now
        assert copied['description'] == 'test'
    
    def test_copy_empty_meta(self):
        """Test that copy() works with empty metadata."""
        meta = EyeMeta()
        copied = meta.copy()
        
        assert len(copied) == 0
        assert copied is not meta


class TestEyeEnfaceMetaCopy:
    """Test suite for EyeEnfaceMeta.copy() method."""
    
    def test_copy_creates_new_instance(self):
        """Test that copy() creates a new instance."""
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')
        copied = meta.copy()
        
        assert copied is not meta
        assert isinstance(copied, EyeEnfaceMeta)
    
    def test_copy_preserves_required_fields(self):
        """Test that copy() preserves required fields."""
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=15.0, scale_unit='µm')
        copied = meta.copy()
        
        assert copied['scale_x'] == 10.0
        assert copied['scale_y'] == 15.0
        assert copied['scale_unit'] == 'µm'
    
    def test_copy_preserves_optional_fields(self):
        """Test that copy() preserves optional fields."""
        meta = EyeEnfaceMeta(
            scale_x=10.0,
            scale_y=10.0,
            scale_unit='µm',
            laterality='OD',
            patient_id='12345'
        )
        copied = meta.copy()
        
        assert copied['laterality'] == 'OD'
        assert copied['patient_id'] == '12345'
    
    def test_copy_is_independent(self):
        """Test that modifying the copy doesn't affect the original."""
        meta = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')
        copied = meta.copy()
        
        # Modify the copy
        copied['scale_x'] = 20.0
        copied['custom_field'] = 'custom_value'
        
        # Original should be unchanged
        assert meta['scale_x'] == 10.0
        assert 'custom_field' not in meta
    
    def test_copy_with_datetime_fields(self):
        """Test that copy() works with datetime fields."""
        visit_date = datetime.datetime(2025, 10, 7, 14, 30)
        exam_time = datetime.datetime(2025, 10, 7, 15, 0)
        
        meta = EyeEnfaceMeta(
            scale_x=10.0,
            scale_y=10.0,
            scale_unit='µm',
            visit_date=visit_date,
            exam_time=exam_time
        )
        copied = meta.copy()
        
        assert copied['visit_date'] == visit_date
        assert copied['exam_time'] == exam_time
    
    def test_copy_with_laterality(self):
        """Test that copy() preserves laterality."""
        for laterality in ['OD', 'OS', 'R', 'L']:
            meta = EyeEnfaceMeta(
                scale_x=10.0,
                scale_y=10.0,
                scale_unit='µm',
                laterality=laterality
            )
            copied = meta.copy()
            
            assert copied['laterality'] == laterality
    
    def test_copy_then_modify_scale(self):
        """Test a common use case: copy and modify scale for transformation."""
        original = EyeEnfaceMeta(scale_x=10.0, scale_y=10.0, scale_unit='µm')
        
        # Simulate what happens in a 2x scaling transformation
        transformed = original.copy()
        transformed['scale_x'] = transformed['scale_x'] / 2.0
        transformed['scale_y'] = transformed['scale_y'] / 2.0
        
        # Check transformed values
        assert transformed['scale_x'] == 5.0
        assert transformed['scale_y'] == 5.0
        
        # Check original is unchanged
        assert original['scale_x'] == 10.0
        assert original['scale_y'] == 10.0


class TestMetaCopyIntegration:
    """Integration tests for metadata copying in realistic scenarios."""
    
    def test_copy_preserves_all_data_types(self):
        """Test that copy() handles various data types correctly."""
        meta = EyeEnfaceMeta(
            scale_x=10.5,
            scale_y=12.3,
            scale_unit='µm',
            laterality='OD',
            patient_id='PAT-12345',
            age=45,
            quality_score=0.95,
            notes='High quality scan',
            flags={'artifact': False, 'motion': True}
        )
        
        copied = meta.copy()
        
        # Verify all fields
        assert copied['scale_x'] == 10.5
        assert copied['scale_y'] == 12.3
        assert copied['scale_unit'] == 'µm'
        assert copied['laterality'] == 'OD'
        assert copied['patient_id'] == 'PAT-12345'
        assert copied['age'] == 45
        assert copied['quality_score'] == 0.95
        assert copied['notes'] == 'High quality scan'
        assert copied['flags'] == {'artifact': False, 'motion': True}
        
        # Verify independence
        copied['age'] = 50
        assert meta['age'] == 45
