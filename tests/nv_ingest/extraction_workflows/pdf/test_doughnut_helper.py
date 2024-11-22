from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nv_ingest.extraction_workflows.pdf.doughnut_helper import _construct_table_metadata
from nv_ingest.extraction_workflows.pdf.doughnut_helper import doughnut
from nv_ingest.extraction_workflows.pdf.doughnut_helper import preprocess_and_send_requests
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.util.nim import doughnut as doughnut_utils
from nv_ingest.util.nim.helpers import call_image_inference_model
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import LatexTable

_MODULE_UNDER_TEST = "nv_ingest.extraction_workflows.pdf.doughnut_helper"


@pytest.fixture
def document_df():
    """Fixture to create a DataFrame for testing."""
    return pd.DataFrame(
        {
            "source_id": ["source1"],
        }
    )


@pytest.fixture
def sample_pdf_stream():
    with open("data/test.pdf", "rb") as f:
        pdf_stream = BytesIO(f.read())
    return pdf_stream


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{_MODULE_UNDER_TEST}.call_image_inference_model")
def test_doughnut_text_extraction(mock_call_inference, mock_create_client, sample_pdf_stream, document_df):
    mock_create_client.return_value = MagicMock()
    mock_call_inference.return_value = "<x_0><y_1>testing<x_10><y_20><class_Text>"

    result = doughnut(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        doughnut_config=MagicMock(doughnut_batch_size=1),
    )

    mock_call_inference.assert_called()

    assert len(result) == 1
    assert result[0][0].value == "text"
    assert result[0][1]["content"] == "testing"
    assert result[0][1]["source_metadata"]["source_id"] == "source1"


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{_MODULE_UNDER_TEST}.call_image_inference_model")
def test_doughnut_table_extraction(mock_call_inference, mock_create_client, sample_pdf_stream, document_df):
    mock_create_client.return_value = MagicMock()
    mock_call_inference.return_value = "<x_17><y_0>table text<x_1007><y_1280><class_Table>"

    result = doughnut(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        row_data=document_df.iloc[0],
        text_depth="page",
        doughnut_config=MagicMock(doughnut_batch_size=1),
    )

    mock_call_inference.assert_called()

    assert len(result) == 2
    assert result[0][0].value == "structured"
    assert result[0][1]["content"] == "table text"
    assert result[0][1]["table_metadata"]["table_location"] == (0, 0, 1024, 1280)
    assert result[0][1]["table_metadata"]["table_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"
    assert result[1][1]["content"] == ""


@patch(f"{_MODULE_UNDER_TEST}.create_inference_client")
@patch(f"{_MODULE_UNDER_TEST}.call_image_inference_model")
def test_doughnut_image_extraction(mock_call_inference, mock_create_client, sample_pdf_stream, document_df):
    mock_create_client.return_value = MagicMock()
    mock_call_inference.return_value = "<x_17><y_0><x_1007><y_1280><class_Picture>"

    result = doughnut(
        pdf_stream=sample_pdf_stream,
        extract_text=True,
        extract_images=True,
        extract_tables=False,
        row_data=document_df.iloc[0],
        text_depth="page",
        doughnut_config=MagicMock(doughnut_batch_size=1),
    )

    mock_call_inference.assert_called()

    assert len(result) == 2
    assert result[0][0].value == "image"
    assert result[0][1]["content"][:10] == "iVBORw0KGg"  # PNG format header
    assert result[0][1]["image_metadata"]["image_location"] == (0, 0, 1024, 1280)
    assert result[0][1]["image_metadata"]["image_location_max_dimensions"] == (1024, 1280)
    assert result[1][0].value == "text"
    assert result[1][1]["content"] == ""


@patch(f"{_MODULE_UNDER_TEST}.pdfium_pages_to_numpy")
@patch(f"{_MODULE_UNDER_TEST}.call_image_inference_model")
def test_preprocess_and_send_requests(mock_call_inference, mock_pdfium_pages_to_numpy):
    mock_call_inference.return_value = ["<x_0><y_1>testing<x_10><y_20><class_Text>"] * 3
    mock_pdfium_pages_to_numpy.return_value = (np.array([[1], [2], [3]]), [0, 1, 2])

    mock_client = MagicMock()
    batch = [MagicMock()] * 3
    batch_offset = 0

    result = preprocess_and_send_requests(mock_client, batch, batch_offset)

    assert len(result) == 3, "Result should have 3 entries"
    assert all(
        isinstance(item, tuple) and len(item) == 3 for item in result
    ), "Each entry should be a tuple with 3 items"

    mock_call_inference.assert_called()