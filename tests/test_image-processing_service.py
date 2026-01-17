"""
Unit tests for facial recognition microservice.
Tests cover image processing, face encoding comparison, and multi-image registration.
"""

import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from src.service.image_processing.face_encoding_comparing_service import (
    FaceEncodingService,
)

# services
from src.service.image_processing.image_processing_service import ImageProcessingService
from src.service.image_processing.multiple_image_processing import (
    MultiImageRegistrationService,
)

# FIXTURES


@pytest.fixture
def sample_encoding():
    """Generate a sample 128-dimensional face encoding."""
    return np.random.rand(128).tolist()


@pytest.fixture
def sample_encoding_array():
    """Generate a sample encoding as numpy array."""
    return np.random.rand(128)


@pytest.fixture
def valid_base64_image():
    """Create a valid base64-encoded image."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile."""
    file = MagicMock(spec=UploadFile)
    file.filename = "test_image.jpg"
    file.content_type = "image/jpeg"

    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    file.read = AsyncMock(return_value=buffer.getvalue())

    return file


@pytest.fixture
def mock_upload_files_list():
    """Create a list of mock UploadFile objects."""
    files = []
    for i in range(5):
        file = MagicMock(spec=UploadFile)
        file.filename = f"image_{i}.jpg"
        file.content_type = "image/jpeg"

        img = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        file.read = AsyncMock(return_value=buffer.getvalue())
        files.append(file)

    return files


# IMAGE PROCESSING SERVICE TESTS


class TestImageProcessingService:
    """Test suite for ImageProcessingService."""

    def test_extract_encoding_from_base64_success(
        self, valid_base64_image, sample_encoding_array
    ):
        """Test successful encoding extraction from base64 image."""
        service = ImageProcessingService()

        with patch("face_recognition.face_encodings") as mock_encodings:
            mock_encodings.return_value = [sample_encoding_array]

            result = service.extract_encoding_from_base64(valid_base64_image)

            assert result["success"] is True
            assert "facialEncoding" in result
            assert len(result["facialEncoding"]) == 128

    def test_extract_encoding_from_base64_with_header(
        self, valid_base64_image, sample_encoding_array
    ):
        """Test extraction with data URI header."""
        service = ImageProcessingService()
        base64_with_header = f"data:image/png;base64,{valid_base64_image}"

        with patch("face_recognition.face_encodings") as mock_encodings:
            mock_encodings.return_value = [sample_encoding_array]

            result = service.extract_encoding_from_base64(base64_with_header)

            assert result["success"] is True
            assert len(result["facialEncoding"]) == 128

    def test_extract_encoding_no_faces_detected(self, valid_base64_image):
        """Test when no faces are detected in image."""
        service = ImageProcessingService()

        with patch("face_recognition.face_encodings") as mock_encodings:
            mock_encodings.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                service.extract_encoding_from_base64(valid_base64_image)

            assert exc_info.value.status_code == 400
            assert "No faces detected" in exc_info.value.detail

    def test_extract_encoding_multiple_faces(
        self, valid_base64_image, sample_encoding_array
    ):
        """Test when multiple faces are detected."""
        service = ImageProcessingService()

        with patch("face_recognition.face_encodings") as mock_encodings:
            mock_encodings.return_value = [sample_encoding_array, sample_encoding_array]

            with pytest.raises(HTTPException) as exc_info:
                service.extract_encoding_from_base64(valid_base64_image)

            assert exc_info.value.status_code == 400
            assert "Multiple faces detected" in exc_info.value.detail

    def test_extract_encoding_invalid_base64(self):
        """Test with invalid base64 string."""
        service = ImageProcessingService()

        with pytest.raises(HTTPException) as exc_info:
            service.extract_encoding_from_base64("invalid_base64_string!!!")

        assert exc_info.value.status_code == 400
        assert "Invalid Base64 format" in exc_info.value.detail

    def test_extract_encoding_empty_data(self):
        """Test with empty base64 data."""
        service = ImageProcessingService()

        with pytest.raises(HTTPException) as exc_info:
            service.extract_encoding_from_base64("")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_extract_single_encoding_success(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test successful single encoding extraction from upload."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(0, 100, 100, 0)]
            mock_encodings.return_value = [sample_encoding_array]

            result = await service.extract_single_encoding(mock_upload_file)

            assert isinstance(result, np.ndarray)
            assert len(result) == 128

    @pytest.mark.asyncio
    async def test_extract_single_encoding_invalid_file_type(self):
        """Test with invalid file type."""
        service = ImageProcessingService()
        file = MagicMock(spec=UploadFile)
        file.filename = "document.pdf"
        file.content_type = "application/pdf"

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_single_encoding(file)

        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_single_encoding_empty_file(self):
        """Test with empty file."""
        service = ImageProcessingService()
        file = MagicMock(spec=UploadFile)
        file.filename = "empty.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=b"")

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_single_encoding(file)

        assert exc_info.value.status_code == 400
        assert "Empty or corrupted" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_single_encoding_no_faces(self, mock_upload_file):
        """Test when no faces found in uploaded file."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = []
            mock_encodings.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_single_encoding(mock_upload_file)

            assert exc_info.value.status_code == 400
            assert "Expected 1 face" in exc_info.value.detail


# FACE ENCODING SERVICE TESTS


class TestFaceEncodingService:
    """Test suite for FaceEncodingService."""

    def test_compare_encodings_match(self, sample_encoding):
        """Test comparing matching encodings."""
        service = FaceEncodingService()

        # creating very similar encodings
        encoding1 = sample_encoding
        # with a slight variation
        encoding2 = [x + 0.01 for x in sample_encoding]

        result = service.compare_encodings(encoding1, encoding2)

        assert "is_face_matched" in result
        assert "face_distance" in result
        assert "confidence" in result
        assert isinstance(result["is_face_matched"], bool)
        assert result["face_distance"] < 0.6

    def test_compare_encodings_no_match(self):
        """Test comparing non-matching encodings."""
        service = FaceEncodingService()

        encoding1 = np.random.rand(128).tolist()
        encoding2 = np.random.rand(128).tolist()

        result = service.compare_encodings(encoding1, encoding2, threshold=0.6)

        assert "is_face_matched" in result
        assert "face_distance" in result

    def test_compare_encodings_invalid_dimensions(self, sample_encoding):
        """Test with invalid encoding dimensions."""
        service = FaceEncodingService()

        invalid_encoding = [0.5] * 64  # Wrong size

        with pytest.raises(HTTPException) as exc_info:
            service.compare_encodings(sample_encoding, invalid_encoding)

        assert exc_info.value.status_code == 400
        assert "Invalid encoding dimensions" in exc_info.value.detail

    def test_compare_encodings_non_numeric(self, sample_encoding):
        """Test with non-numeric data."""
        service = FaceEncodingService()

        # create an invalid encoding with string values
        invalid_encoding = ["not", "a", "number"] * 43  # 129 items, but strings

        with pytest.raises(HTTPException) as exc_info:
            service.compare_encodings(sample_encoding, invalid_encoding[:128])  # type: ignore

        assert exc_info.value.status_code == 400

    def test_compare_encodings_with_nan(self, sample_encoding):
        """Test with NaN values."""
        service = FaceEncodingService()

        invalid_encoding = sample_encoding.copy()
        invalid_encoding[0] = float("nan")

        with pytest.raises(HTTPException) as exc_info:
            service.compare_encodings(sample_encoding, invalid_encoding)

        assert exc_info.value.status_code == 400
        assert "invalid values" in exc_info.value.detail.lower()

    def test_compare_encodings_with_infinity(self, sample_encoding):
        """Test with infinite values."""
        service = FaceEncodingService()

        invalid_encoding = sample_encoding.copy()
        invalid_encoding[0] = float("inf")

        with pytest.raises(HTTPException) as exc_info:
            service.compare_encodings(sample_encoding, invalid_encoding)

        assert exc_info.value.status_code == 400

    def test_compare_encodings_custom_threshold(self, sample_encoding):
        """Test with custom threshold."""
        service = FaceEncodingService()

        encoding1 = sample_encoding
        encoding2 = [x + 0.5 for x in sample_encoding]

        result = service.compare_encodings(encoding1, encoding2, threshold=1.0)

        assert "is_face_matched" in result

    def test_calculate_consistency_score_empty(self):
        """Test consistency score with empty list."""
        service = FaceEncodingService()

        score = service.calculate_consistency_score([])

        assert score == 1.0

    def test_calculate_consistency_score_single(self, sample_encoding_array):
        """Test consistency score with single encoding."""
        service = FaceEncodingService()

        score = service.calculate_consistency_score([sample_encoding_array])

        assert score == 1.0

    def test_calculate_consistency_score_multiple(self, sample_encoding_array):
        """Test consistency score with multiple encodings."""
        service = FaceEncodingService()

        encodings = [sample_encoding_array, sample_encoding_array + 0.1]
        score = service.calculate_consistency_score(encodings)

        assert 0.0 <= score <= 1.0


# MULTI-IMAGE REGISTRATION SERVICE TESTS


class TestMultiImageRegistrationService:
    """Test suite for MultiImageRegistrationService."""

    @pytest.mark.asyncio
    async def test_process_multi_encodings_success(
        self, sample_encoding_array, mock_upload_files_list
    ):
        """Test successful multi-image processing."""
        encoding_service = FaceEncodingService()
        image_service = ImageProcessingService()
        registration_service = MultiImageRegistrationService(
            encoding_service=encoding_service, image_service=image_service
        )

        with patch.object(
            image_service, "extract_single_encoding", new_callable=AsyncMock
        ) as mock_extract:
            mock_extract.return_value = sample_encoding_array

            result = await registration_service.process_multi_encodings(
                mock_upload_files_list
            )

            assert result["success"] is True
            assert "facialEncoding" in result
            assert len(result["facialEncoding"]) == 128
            assert mock_extract.call_count == 5

    @pytest.mark.asyncio
    async def test_process_multi_encodings_insufficient_images(self):
        """Test with fewer than 5 images."""
        encoding_service = FaceEncodingService()
        image_service = ImageProcessingService()
        registration_service = MultiImageRegistrationService(
            encoding_service=encoding_service, image_service=image_service
        )

        # create only 3 mock files
        files = []
        for i in range(3):
            file = MagicMock(spec=UploadFile)
            file.filename = f"image_{i}.jpg"
            files.append(file)

        with pytest.raises(HTTPException) as exc_info:
            await registration_service.process_multi_encodings(files)

        assert exc_info.value.status_code == 400
        assert "At least 5 images required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_multi_encodings_inconsistent_faces(
        self, sample_encoding_array, mock_upload_files_list
    ):
        """Test with inconsistent facial encodings."""
        encoding_service = FaceEncodingService()
        image_service = ImageProcessingService()
        registration_service = MultiImageRegistrationService(
            encoding_service=encoding_service, image_service=image_service
        )

        with patch.object(
            image_service, "extract_single_encoding", new_callable=AsyncMock
        ) as mock_extract:
            encodings = [sample_encoding_array]
            encodings.extend([sample_encoding_array + 2.0 for _ in range(4)])
            mock_extract.side_effect = encodings

            with pytest.raises(HTTPException) as exc_info:
                await registration_service.process_multi_encodings(
                    mock_upload_files_list
                )

            assert exc_info.value.status_code == 400
            assert "Inconsistent facial encodings" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_multi_encodings_averaging(
        self, sample_encoding_array, mock_upload_files_list
    ):
        """Test that encodings are properly averaged."""
        encoding_service = FaceEncodingService()
        image_service = ImageProcessingService()
        registration_service = MultiImageRegistrationService(
            encoding_service=encoding_service, image_service=image_service
        )

        # create a slightly different encodings
        encodings = []
        base = sample_encoding_array
        for i in range(5):
            encodings.append(base + (i * 0.01))

        with patch.object(
            image_service, "extract_single_encoding", new_callable=AsyncMock
        ) as mock_extract:
            mock_extract.side_effect = encodings

            result = await registration_service.process_multi_encodings(
                mock_upload_files_list
            )

            expected_avg = np.mean(encodings, axis=0)
            result_encoding = np.array(result["facialEncoding"])

            np.testing.assert_array_almost_equal(
                result_encoding, expected_avg, decimal=5
            )


# INTEGRATION TESTS


class TestServiceIntegration:
    """Integration tests for service interactions."""

    def test_full_verification_flow(self, sample_encoding):
        """Test complete verification workflow."""
        encoding_service = FaceEncodingService()

        # simulate registration encoding
        registered_encoding = sample_encoding

        # simulate authentication encoding (slightly different)
        auth_encoding = [x + 0.05 for x in sample_encoding]

        result = encoding_service.compare_encodings(
            uploaded_encoding=auth_encoding, reference_encoding=registered_encoding
        )

        assert "is_face_matched" in result
        assert "confidence" in result
        assert result["confidence"] > 0

    @pytest.mark.asyncio
    async def test_registration_to_verification_flow(
        self, sample_encoding_array, mock_upload_files_list
    ):
        """Test flow from registration to verification."""
        encoding_service = FaceEncodingService()
        image_service = ImageProcessingService()
        registration_service = MultiImageRegistrationService(
            encoding_service=encoding_service, image_service=image_service
        )

        # Step 1: Register with 5 images
        with patch.object(
            image_service, "extract_single_encoding", new_callable=AsyncMock
        ) as mock_extract:
            mock_extract.return_value = sample_encoding_array

            registration_result = await registration_service.process_multi_encodings(
                mock_upload_files_list
            )
            registered_encoding = registration_result["facialEncoding"]

        # Step 2: Verify with similar encoding
        auth_encoding = (sample_encoding_array + 0.05).tolist()

        verification_result = encoding_service.compare_encodings(
            uploaded_encoding=auth_encoding, reference_encoding=registered_encoding
        )

        assert registration_result["success"] is True
        assert "is_face_matched" in verification_result


# EDGE CASES AND ERROR HANDLING


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_encoding_all_zeros(self):
        """Test with all-zero encoding."""
        service = FaceEncodingService()

        zero_encoding = [0.0] * 128
        normal_encoding = np.random.rand(128).tolist()

        result = service.compare_encodings(zero_encoding, normal_encoding)

        assert "face_distance" in result

    def test_encoding_all_ones(self):
        """Test with all-ones encoding."""
        service = FaceEncodingService()

        ones_encoding = [1.0] * 128
        normal_encoding = np.random.rand(128).tolist()

        result = service.compare_encodings(ones_encoding, normal_encoding)

        assert "face_distance" in result

    def test_identical_encodings(self, sample_encoding):
        """Test with identical encodings."""
        service = FaceEncodingService()

        result = service.compare_encodings(sample_encoding, sample_encoding)

        assert result["is_face_matched"] is True
        assert result["face_distance"] == 0.0
        assert result["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_large_image_handling(self):
        """Test handling of large images."""
        service = ImageProcessingService()

        large_img = Image.new("RGB", (4000, 3000), color="green")
        buffer = io.BytesIO()
        large_img.save(buffer, format="JPEG")

        file = MagicMock(spec=UploadFile)
        file.filename = "large_image.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=buffer.getvalue())

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(0, 640, 640, 0)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_single_encoding(file)

            assert isinstance(result, np.ndarray)
