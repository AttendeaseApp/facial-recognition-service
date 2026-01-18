"""
Unit tests for facial recognition microservice.
Tests cover improved image processing with quality checks and face encoding comparison.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from src.service.image_processing.face_encoding_comparing_service import (
    FaceEncodingService,
)
from src.service.image_processing.image_processing_service import ImageProcessingService

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
def mock_upload_file():
    """Create a mock UploadFile."""
    file = MagicMock(spec=UploadFile)
    file.filename = "test_image.jpg"
    file.content_type = "image/jpeg"

    img = Image.new("RGB", (640, 480), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    file.read = AsyncMock(return_value=buffer.getvalue())

    return file


@pytest.fixture
def mock_upload_files_list():
    """Create a list of 5 mock UploadFile objects."""
    files = []
    for i in range(5):
        file = MagicMock(spec=UploadFile)
        file.filename = f"image_{i}.jpg"
        file.content_type = "image/jpeg"

        img = Image.new("RGB", (640, 480), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        file.read = AsyncMock(return_value=buffer.getvalue())
        files.append(file)

    return files


@pytest.fixture
def small_image_file():
    """Create a mock UploadFile with image too small."""
    file = MagicMock(spec=UploadFile)
    file.filename = "small_image.jpg"
    file.content_type = "image/jpeg"

    img = Image.new("RGB", (100, 100), color="red")  # Below MIN_DIMENSION
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    file.read = AsyncMock(return_value=buffer.getvalue())

    return file


# IMPROVED IMAGE PROCESSING SERVICE TESTS


class TestImprovedImageProcessingService:
    """Test suite for improved ImageProcessingService with quality checks."""

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_success(
        self, mock_upload_files_list, sample_encoding_array
    ):
        """Test successful extraction from 5 images with quality metadata."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            result = await service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )

            assert result["success"] is True
            assert "facialEncoding" in result
            assert len(result["facialEncoding"]) == 128
            assert "metadata" in result
            assert result["metadata"]["images_processed"] == 5
            assert "average_quality" in result["metadata"]
            assert "quality_scores" in result["metadata"]
            assert len(result["metadata"]["quality_scores"]) == 5

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_wrong_count(self, mock_upload_files_list):
        """Test with incorrect number of images."""
        service = ImageProcessingService()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings(
                mock_upload_files_list[:3], required_count=5
            )

        assert exc_info.value.status_code == 400
        assert "Expected 5 images, received 3" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_no_face(self, mock_upload_files_list):
        """Test when no face detected in one image."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )

            assert exc_info.value.status_code == 400
            assert "No face detected in image 1" in exc_info.value.detail
            assert "clearly visible and well-lit" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_multiple_faces(
        self, mock_upload_files_list
    ):
        """Test when multiple faces detected."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.return_value = [
                (50, 590, 430, 50),
                (50, 200, 200, 50),
            ]

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )

            assert exc_info.value.status_code == 400
            assert "Multiple faces detected in image 1" in exc_info.value.detail
            assert "only one face is visible" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_face_too_small(
        self, mock_upload_files_list
    ):
        """Test when face is too small."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.return_value = [(10, 40, 40, 10)]

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )

            assert exc_info.value.status_code == 400
            assert "Face too small in image 1" in exc_info.value.detail
            assert "move closer to the camera" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_invalid_file_type(self):
        """Test with invalid file type."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "document.pdf"
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"fake pdf content")

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings([file], required_count=1)

        assert exc_info.value.status_code == 400
        assert "Invalid file type for image 1" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_empty_file(self, mock_upload_file):
        """Test with empty file."""
        service = ImageProcessingService()

        mock_upload_file.read = AsyncMock(return_value=b"")

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings(
                [mock_upload_file], required_count=1
            )

        assert exc_info.value.status_code == 400
        assert "Empty image file" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_corrupt_image(self, mock_upload_file):
        """Test with corrupted image data."""
        service = ImageProcessingService()

        mock_upload_file.read = AsyncMock(return_value=b"not an image")

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings(
                [mock_upload_file], required_count=1
            )

        assert exc_info.value.status_code == 400
        assert "Could not process image 1" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_averaging(self, mock_upload_files_list):
        """Test that encodings are properly averaged."""
        service = ImageProcessingService()

        base_encoding = np.random.rand(128)
        encodings = [base_encoding + (i * 0.01) for i in range(5)]

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.side_effect = [[enc] for enc in encodings]

            result = await service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )

            expected_avg = np.mean(encodings, axis=0)
            result_encoding = np.array(result["facialEncoding"])

            np.testing.assert_array_almost_equal(
                result_encoding, expected_avg, decimal=5
            )

    @pytest.mark.asyncio
    async def test_preprocess_image_too_small(self, small_image_file):
        """Test preprocessing rejects images below minimum size."""
        service = ImageProcessingService()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings(
                [small_image_file], required_count=1
            )

        assert exc_info.value.status_code == 400
        assert "small" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_preprocess_image_resizing(self):
        """Test that large images are resized."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "large_image.jpg"
        file.content_type = "image/jpeg"

        large_img = Image.new("RGB", (3000, 2000), color="green")
        buffer = io.BytesIO()
        large_img.save(buffer, format="JPEG")
        file.read = AsyncMock(return_value=buffer.getvalue())

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_multiple_encodings([file], required_count=1)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, mock_upload_files_list):
        """Test that quality scores are calculated and valid."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )

            for score in result["metadata"]["quality_scores"]:
                assert 0 <= score <= 100

            avg_quality = result["metadata"]["average_quality"]
            assert 0 <= avg_quality <= 100
            assert avg_quality == np.mean(result["metadata"]["quality_scores"])

    @pytest.mark.asyncio
    async def test_extract_single_encoding_backward_compatibility(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test legacy single encoding method still works."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            result = await service.extract_multiple_encodings(mock_upload_file)

            assert isinstance(result, np.ndarray)
            assert len(result) == 128

    @pytest.mark.asyncio
    async def test_grayscale_image_conversion(self):
        """Test handling of grayscale images."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "grayscale.jpg"
        file.content_type = "image/jpeg"

        gray_img = Image.new("L", (640, 480), color=128)
        buffer = io.BytesIO()
        gray_img.save(buffer, format="JPEG")
        file.read = AsyncMock(return_value=buffer.getvalue())

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_multiple_encodings([file], required_count=1)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_rgba_image_conversion(self):
        """Test handling of RGBA images with alpha channel."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "transparent.png"
        file.content_type = "image/png"

        rgba_img = Image.new("RGBA", (640, 480), color=(100, 100, 200, 255))
        buffer = io.BytesIO()
        rgba_img.save(buffer, format="PNG")
        file.read = AsyncMock(return_value=buffer.getvalue())

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_multiple_encodings([file], required_count=1)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_different_image_formats(self):
        """Test support for different image formats."""
        service = ImageProcessingService()

        formats = [
            ("test.jpg", "image/jpeg", "JPEG"),
            ("test.png", "image/png", "PNG"),
            ("test.bmp", "image/bmp", "BMP"),
        ]

        for filename, content_type, img_format in formats:
            file = MagicMock(spec=UploadFile)
            file.filename = filename
            file.content_type = content_type

            img = Image.new("RGB", (640, 480), color="blue")
            buffer = io.BytesIO()
            img.save(buffer, format=img_format)
            file.read = AsyncMock(return_value=buffer.getvalue())

            with (
                patch("face_recognition.face_locations") as mock_locations,
                patch("face_recognition.face_encodings") as mock_encodings,
            ):
                mock_locations.return_value = [(50, 590, 430, 50)]
                mock_encodings.return_value = [np.random.rand(128)]

                result = await service.extract_multiple_encodings(
                    [file], required_count=1
                )

                assert result["success"] is True


# FACE ENCODING SERVICE TESTS


class TestFaceEncodingService:
    """Test suite for FaceEncodingService."""

    def test_compare_encodings_match(self, sample_encoding):
        """Test comparing matching encodings."""
        service = FaceEncodingService()

        encoding1 = sample_encoding
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

        invalid_encoding = [0.5] * 64

        with pytest.raises(HTTPException) as exc_info:
            service.compare_encodings(sample_encoding, invalid_encoding)

        assert exc_info.value.status_code == 400
        assert "Invalid encoding dimensions" in exc_info.value.detail

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

    def test_identical_encodings(self, sample_encoding):
        """Test with identical encodings."""
        service = FaceEncodingService()

        result = service.compare_encodings(sample_encoding, sample_encoding)

        assert result["is_face_matched"] is True
        assert result["face_distance"] == 0.0
        assert result["confidence"] == 1.0


# INTEGRATION TESTS


class TestServiceIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_registration_flow(self, mock_upload_files_list):
        """Test complete registration workflow with quality checks."""
        service = ImageProcessingService()

        base_encoding = np.random.rand(128)
        encodings = [base_encoding + np.random.rand(128) * 0.05 for _ in range(5)]

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.side_effect = [[enc] for enc in encodings]

            result = await service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )

            assert result["success"] is True
            assert "facialEncoding" in result
            assert len(result["facialEncoding"]) == 128
            assert "metadata" in result
            assert result["metadata"]["images_processed"] == 5
            assert 0 <= result["metadata"]["average_quality"] <= 100

            for val in result["facialEncoding"]:
                assert isinstance(val, (int, float))
                assert not np.isnan(val)
                assert not np.isinf(val)

    @pytest.mark.asyncio
    async def test_registration_then_verification_flow(
        self, mock_upload_files_list, sample_encoding_array
    ):
        """Test complete flow from registration to verification."""
        image_service = ImageProcessingService()
        encoding_service = FaceEncodingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            registration_result = await image_service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )
            registered_encoding = registration_result["facialEncoding"]

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

    @pytest.mark.asyncio
    async def test_large_image_handling(self):
        """Test handling of very large images."""
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
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_multiple_encodings(file)

            assert isinstance(result, np.ndarray)

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, mock_upload_files_list):
        """Test that service can handle multiple concurrent requests."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            results = []
            for _ in range(3):
                result = await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )
                results.append(result)

            for result in results:
                assert result["success"] is True
                assert len(result["facialEncoding"]) == 128
