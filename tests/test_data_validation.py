import pandas as pd
import pytest

from src.components.data_validation import DataValidation
from src.exception import CustomException


def test_validate_columns():
    """Test that missing columns are detected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["female"],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException):
        validator._validate_columns(bad_data)


def test_validate_no_nulls():
    """Test that null values are detected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["female", "male", None],
        "race_ethnicity": ["group A", "group B", "group C"],
        "parental_level_of_education": ["high school", "some college", "bachelor's degree"],
        "lunch": ["standard", "free/reduced", "standard"],
        "test_preparation_course": ["none", "completed", "none"],
        "math_score": [65, 72, 88],
        "reading_score": [70, 75, 90],
        "writing_score": [68, 74, 85],
    })

    with pytest.raises(CustomException, match="null value"):
        validator._validate_no_nulls(bad_data)


def test_validate_data_types_scores_not_int():
    """Test that non-int score columns are rejected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["female"],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "math_score": [65.5],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException, match="int64"):
        validator._validate_data_types(bad_data)


def test_validate_data_types_categorical_not_string():
    """Test that non-string categorical columns are rejected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": [1],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "math_score": [65],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException, match="string/object"):
        validator._validate_data_types(bad_data)


def test_validate_score_ranges():
    """Test that out-of-range scores are detected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["female"],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "math_score": [150],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException):
        validator._validate_score_ranges(bad_data)


def test_validate_categorical_values():
    """Test that invalid categorical values are detected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["invalid_gender"],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "math_score": [70],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException):
        validator._validate_categorical_values(bad_data)


def test_validate_no_empty_strings_whitespace():
    """Test that whitespace-only strings are detected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["   "],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "math_score": [70],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException, match="empty or whitespace"):
        validator._validate_no_empty_strings(bad_data)


def test_validate_no_empty_strings_placeholder():
    """Test that placeholder values are detected."""
    validator = DataValidation()

    bad_data = pd.DataFrame({
        "gender": ["N/A"],
        "race_ethnicity": ["group A"],
        "parental_level_of_education": ["high school"],
        "lunch": ["standard"],
        "test_preparation_course": ["none"],
        "math_score": [70],
        "reading_score": [70],
        "writing_score": [68],
    })

    with pytest.raises(CustomException, match="placeholder"):
        validator._validate_no_empty_strings(bad_data)


def test_remove_duplicates():
    """Test that duplicates are removed."""
    validator = DataValidation()

    data_with_dupes = pd.DataFrame({
        "gender": ["female", "female"],
        "race_ethnicity": ["group A", "group A"],
        "parental_level_of_education": ["high school", "high school"],
        "lunch": ["standard", "standard"],
        "test_preparation_course": ["none", "none"],
        "math_score": [70, 70],
        "reading_score": [70, 70],
        "writing_score": [68, 68],
    })

    result = validator._remove_duplicates(data_with_dupes)
    assert len(result) == 1


def test_validate_all_success():
    """Test that validate_all passes with clean data."""
    validator = DataValidation()

    good_data = pd.DataFrame({
        "gender": ["female", "male"],
        "race_ethnicity": ["group A", "group B"],
        "parental_level_of_education": ["high school", "some college"],
        "lunch": ["standard", "free/reduced"],
        "test_preparation_course": ["none", "completed"],
        "math_score": [65, 72],
        "reading_score": [70, 75],
        "writing_score": [68, 74],
    })

    result = validator.validate_all(good_data)
    assert len(result) == 2
