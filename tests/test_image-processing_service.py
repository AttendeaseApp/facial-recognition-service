"""
Unit tests for optimized facial recognition microservice.
Tests cover improved image processing with parallel processing, security checks, and quality validation.
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
    image_bytes = buffer.getvalue()

    file.read = AsyncMock(return_value=image_bytes)
    file.seek = AsyncMock()

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
        image_bytes = buffer.getvalue()

        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()
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
    image_bytes = buffer.getvalue()

    file.read = AsyncMock(return_value=image_bytes)
    file.seek = AsyncMock()

    return file


@pytest.fixture
def large_image_file():
    """Create a mock UploadFile exceeding MAX_FILE_SIZE."""
    file = MagicMock(spec=UploadFile)
    file.filename = "large_file.jpg"
    file.content_type = "image/jpeg"

    # Create data larger than 10MB limit
    large_data = b"x" * (11 * 1024 * 1024)
    file.read = AsyncMock(return_value=large_data)
    file.seek = AsyncMock()

    return file


# OPTIMIZED IMAGE PROCESSING SERVICE TESTS


class TestOptimizedImageProcessingService:
    """Test suite for optimized ImageProcessingService with parallel processing and security."""

    @pytest.mark.asyncio
    async def test_extract_single_encoding_success(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test successful single image encoding extraction."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            result = await service.extract_single_encoding(mock_upload_file)

            assert result["success"] is True
            assert "facialEncoding" in result
            assert len(result["facialEncoding"]) == 128
            assert "metadata" in result
            assert result["metadata"]["images_processed"] == 1
            assert "average_quality" in result["metadata"]
            assert "processing_time" in result["metadata"]
            assert len(result["metadata"]["quality_scores"]) == 1

    @pytest.mark.asyncio
    async def test_extract_single_encoding_file_too_large(self, large_image_file):
        """Test single encoding with file exceeding size limit."""
        service = ImageProcessingService()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_single_encoding(large_image_file)

        assert exc_info.value.status_code == 413
        assert "exceeds maximum size" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_success(
        self, mock_upload_files_list, sample_encoding_array
    ):
        """Test successful parallel extraction from 5 images with quality metadata."""
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
            assert "processing_time" in result["metadata"]
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
    async def test_extract_multiple_encodings_total_size_exceeded(
        self, mock_upload_files_list
    ):
        """Test when total upload size exceeds limit."""
        service = ImageProcessingService()

        # Mock files to return large data
        for file in mock_upload_files_list:
            large_data = b"x" * (11 * 1024 * 1024)
            file.read = AsyncMock(return_value=large_data)

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )

        assert exc_info.value.status_code == 413
        assert "Total upload size" in exc_info.value.detail
        assert "exceeds maximum" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_no_face(self, mock_upload_files_list):
        """Test when no face detected in one image during parallel processing."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )

            assert exc_info.value.status_code == 400
            assert "No face detected" in exc_info.value.detail
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
            assert "Multiple faces detected" in exc_info.value.detail
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
            assert "Face too small" in exc_info.value.detail
            assert "move closer to the camera" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_invalid_file_type(self):
        """Test with invalid file type."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "document.pdf"
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"fake pdf content")
        file.seek = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings([file], required_count=1)

        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_unsupported_format(self):
        """Test with unsupported image format (BMP not in ALLOWED_FORMATS)."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "test.bmp"
        file.content_type = "image/bmp"

        img = Image.new("RGB", (640, 480), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="BMP")
        image_bytes = buffer.getvalue()

        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings([file], required_count=1)

        assert exc_info.value.status_code == 400
        assert "Unsupported image format" in exc_info.value.detail

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
        assert "Could not process image" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_averaging(self, mock_upload_files_list):
        """Test that encodings are properly averaged in parallel processing."""
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
            await service.extract_single_encoding(small_image_file)

        assert exc_info.value.status_code == 400

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
        image_bytes = buffer.getvalue()

        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_single_encoding(file)

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
            assert avg_quality == pytest.approx(
                np.mean(result["metadata"]["quality_scores"])
            )

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
        image_bytes = buffer.getvalue()

        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_single_encoding(file)

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
        image_bytes = buffer.getvalue()

        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_single_encoding(file)

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_allowed_image_formats(self):
        """Test support for allowed image formats (JPEG, PNG, WEBP)."""
        service = ImageProcessingService()

        formats = [
            ("test.jpg", "image/jpeg", "JPEG"),
            ("test.png", "image/png", "PNG"),
            ("test.webp", "image/webp", "WEBP"),
        ]

        for filename, content_type, img_format in formats:
            file = MagicMock(spec=UploadFile)
            file.filename = filename
            file.content_type = content_type

            img = Image.new("RGB", (640, 480), color="blue")
            buffer = io.BytesIO()
            img.save(buffer, format=img_format)
            image_bytes = buffer.getvalue()

            file.read = AsyncMock(return_value=image_bytes)
            file.seek = AsyncMock()

            with (
                patch("face_recognition.face_locations") as mock_locations,
                patch("face_recognition.face_encodings") as mock_encodings,
            ):
                mock_locations.return_value = [(50, 590, 430, 50)]
                mock_encodings.return_value = [np.random.rand(128)]

                result = await service.extract_single_encoding(file)

                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_parallel_processing_performance(self, mock_upload_files_list):
        """Test that parallel processing completes successfully."""
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

            assert result["success"] is True
            # Processing time should be logged
            assert "processing_time" in result["metadata"]
            assert result["metadata"]["processing_time"] > 0

    @pytest.mark.asyncio
    async def test_parallel_processing_with_mixed_errors(self, mock_upload_files_list):
        """Test parallel processing stops on first error."""
        service = ImageProcessingService()

        call_count = 0

        def side_effect_locations(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # Third image fails
                return []
            return [(50, 590, 430, 50)]

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.side_effect = side_effect_locations

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )

            assert exc_info.value.status_code == 400
            assert "No face detected" in exc_info.value.detail


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
        """Test complete registration workflow with parallel processing and quality checks."""
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
    async def test_single_verification_flow(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test single image verification workflow."""
        image_service = ImageProcessingService()
        encoding_service = FaceEncodingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            # Extract encoding from verification image
            verification_result = await image_service.extract_single_encoding(
                mock_upload_file
            )
            verification_encoding = verification_result["facialEncoding"]

        # Simulate stored registration encoding
        registered_encoding = (sample_encoding_array + 0.05).tolist()

        # Compare encodings
        comparison_result = encoding_service.compare_encodings(
            uploaded_encoding=verification_encoding,
            reference_encoding=registered_encoding,
        )

        assert verification_result["success"] is True
        assert "is_face_matched" in comparison_result

    @pytest.mark.asyncio
    async def test_registration_then_verification_flow(
        self, mock_upload_files_list, mock_upload_file, sample_encoding_array
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

            # Registration with 5 images
            registration_result = await image_service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )
            registered_encoding = registration_result["facialEncoding"]

            # Verification with 1 image
            verification_result = await image_service.extract_single_encoding(
                mock_upload_file
            )
            verification_encoding = verification_result["facialEncoding"]

        # Compare encodings
        comparison_result = encoding_service.compare_encodings(
            uploaded_encoding=verification_encoding,
            reference_encoding=registered_encoding,
        )

        assert registration_result["success"] is True
        assert verification_result["success"] is True
        assert "is_face_matched" in comparison_result


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
    async def test_very_large_image_handling(self):
        """Test handling of very large images (should be resized)."""
        service = ImageProcessingService()

        large_img = Image.new("RGB", (4000, 3000), color="green")
        buffer = io.BytesIO()
        large_img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        file = MagicMock(spec=UploadFile)
        file.filename = "large_image.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [np.random.rand(128)]

            result = await service.extract_single_encoding(file)

            assert result["success"] is True
            assert len(result["facialEncoding"]) == 128

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        service = ImageProcessingService()

        dangerous_filename = "../../../etc/passwd"
        sanitized = service._sanitize_filename(dangerous_filename)

        assert ".." not in sanitized
        assert "/" not in sanitized

    def test_hash_encoding(self, sample_encoding_array):
        """Test encoding hash generation."""
        service = ImageProcessingService()

        hash1 = service.hash_encoding(sample_encoding_array, "user123", "secret")
        hash2 = service.hash_encoding(sample_encoding_array, "user123", "secret")

        # Same input should produce same hash
        assert hash1 == hash2
        assert len(hash1) > 0

        # Different user should produce different hash
        hash3 = service.hash_encoding(sample_encoding_array, "user456", "secret")
        assert hash1 != hash3

        # Different secret should produce different hash
        hash4 = service.hash_encoding(sample_encoding_array, "user123", "different")
        assert hash1 != hash4


# SECURITY TESTS


class TestSecurityFeatures:
    """Test suite for security features in optimized service."""

    @pytest.mark.asyncio
    async def test_file_size_limit_per_file(self):
        """Test that individual file size limit is enforced."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "huge_file.jpg"
        file.content_type = "image/jpeg"

        # Create file larger than 10MB
        huge_data = b"x" * (11 * 1024 * 1024)
        file.read = AsyncMock(return_value=huge_data)
        file.seek = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_single_encoding(file)

        assert exc_info.value.status_code == 413
        assert "exceeds maximum size" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_total_size_limit_multiple_files(self, mock_upload_files_list):
        """Test that total upload size is validated before processing."""
        service = ImageProcessingService()

        # Make each file 11MB (total 55MB > 50MB limit)
        for file in mock_upload_files_list:
            large_data = b"x" * (11 * 1024 * 1024)
            file.read = AsyncMock(return_value=large_data)
            file.seek = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_multiple_encodings(
                mock_upload_files_list, required_count=5
            )

        assert exc_info.value.status_code == 413
        assert "Total upload size" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_content_type_validation(self):
        """Test that content type is strictly validated."""
        service = ImageProcessingService()

        invalid_types = [
            ("script.js", "application/javascript"),
            ("doc.pdf", "application/pdf"),
            ("data.json", "application/json"),
            ("page.html", "text/html"),
        ]

        for filename, content_type in invalid_types:
            file = MagicMock(spec=UploadFile)
            file.filename = filename
            file.content_type = content_type
            file.read = AsyncMock(return_value=b"fake content")
            file.seek = AsyncMock()

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_single_encoding(file)

            assert exc_info.value.status_code == 400
            assert "Invalid file type" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_format_mismatch_detection(self):
        """Test that actual format is validated, not just content-type header."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "fake_image.jpg"
        file.content_type = "image/jpeg"  # Claims to be JPEG

        # But actually is a text file
        file.read = AsyncMock(return_value=b"This is not an image")
        file.seek = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await service.extract_single_encoding(file)

        assert exc_info.value.status_code == 400
        assert "Could not process image" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_filename_sanitization(self):
        """Test that dangerous filenames are sanitized."""
        service = ImageProcessingService()

        dangerous_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "file; rm -rf /",
            "<script>alert('xss')</script>.jpg",
            "file with spaces and special !@#$%^&*().jpg",
        ]

        for filename in dangerous_filenames:
            sanitized = service._sanitize_filename(filename)

            # Should not contain path traversal
            assert ".." not in sanitized
            assert "/" not in sanitized
            assert "\\" not in sanitized

            # Should be limited to 100 chars
            assert len(sanitized) <= 100

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self):
        """Test that image resources are properly closed even on errors."""
        service = ImageProcessingService()

        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = "image/jpeg"

        img = Image.new("RGB", (640, 480), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        file.read = AsyncMock(return_value=image_bytes)
        file.seek = AsyncMock()

        with patch("face_recognition.face_locations") as mock_locations:
            # Force an error after image is opened
            mock_locations.side_effect = Exception("Simulated error")

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_single_encoding(file)

            # Should get a 500 error with generic message
            assert exc_info.value.status_code == 500
            assert "Failed to process image" in exc_info.value.detail


# PERFORMANCE TESTS


class TestPerformanceOptimizations:
    """Test suite for performance optimizations."""

    @pytest.mark.asyncio
    async def test_parallel_processing_faster_than_sequential(
        self, mock_upload_files_list, sample_encoding_array
    ):
        """Test that parallel processing is utilized (not testing actual speed, just structure)."""
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

            # Verify parallel processing completed successfully
            assert result["success"] is True
            assert result["metadata"]["images_processed"] == 5

    @pytest.mark.asyncio
    async def test_processing_time_tracked(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test that processing time is tracked and returned."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            result = await service.extract_single_encoding(mock_upload_file)

            assert "metadata" in result
            assert "processing_time" in result["metadata"]
            assert isinstance(result["metadata"]["processing_time"], float)
            assert result["metadata"]["processing_time"] >= 0

    @pytest.mark.asyncio
    async def test_hog_model_used_by_default(self, mock_upload_file):
        """Test that HOG model is used for faster processing."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.return_value = [(50, 590, 430, 50)]

            with patch("face_recognition.face_encodings") as mock_encodings:
                mock_encodings.return_value = [np.random.rand(128)]

                await service.extract_single_encoding(mock_upload_file)

                # Verify HOG model is used
                mock_locations.assert_called()
                call_args = mock_locations.call_args
                assert call_args[1].get("model") == "hog"


# BACKWARDS COMPATIBILITY TESTS


class TestBackwardsCompatibility:
    """Test that the service maintains backwards compatibility."""

    @pytest.mark.asyncio
    async def test_extract_multiple_encodings_with_one_file(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test that extract_multiple_encodings still works with 1 file for backwards compatibility."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            result = await service.extract_multiple_encodings(
                [mock_upload_file], required_count=1
            )

            assert result["success"] is True
            assert len(result["facialEncoding"]) == 128
            assert result["metadata"]["images_processed"] == 1

    @pytest.mark.asyncio
    async def test_response_format_unchanged(
        self, mock_upload_files_list, sample_encoding_array
    ):
        """Test that response format is backwards compatible."""
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

            # Check all expected fields are present
            assert "success" in result
            assert "facialEncoding" in result
            assert "metadata" in result

            # Check metadata structure
            metadata = result["metadata"]
            assert "images_processed" in metadata
            assert "average_quality" in metadata
            assert "quality_scores" in metadata

            # New field added (not breaking)
            assert "processing_time" in metadata


# CONCURRENT REQUEST TESTS


class TestConcurrency:
    """Test handling of concurrent requests."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_single_extractions(
        self, mock_upload_file, sample_encoding_array
    ):
        """Test service handles multiple concurrent single-image requests."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            # Simulate 3 concurrent verification requests
            import asyncio

            tasks = [
                service.extract_single_encoding(mock_upload_file) for _ in range(3)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            for result in results:
                assert result["success"] is True
                assert len(result["facialEncoding"]) == 128

    @pytest.mark.asyncio
    async def test_multiple_concurrent_multiple_extractions(
        self, mock_upload_files_list, sample_encoding_array
    ):
        """Test service handles multiple concurrent registration requests."""
        service = ImageProcessingService()

        with (
            patch("face_recognition.face_locations") as mock_locations,
            patch("face_recognition.face_encodings") as mock_encodings,
        ):
            mock_locations.return_value = [(50, 590, 430, 50)]
            mock_encodings.return_value = [sample_encoding_array]

            # Simulate 2 concurrent registration requests
            import asyncio

            tasks = [
                service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )
                for _ in range(2)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 2
            for result in results:
                assert result["success"] is True
                assert result["metadata"]["images_processed"] == 5


# ERROR MESSAGE TESTS


class TestErrorMessages:
    """Test that error messages are user-friendly and don't expose internals."""

    @pytest.mark.asyncio
    async def test_generic_error_messages(self, mock_upload_file):
        """Test that internal errors are sanitized."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            # Simulate internal error
            mock_locations.side_effect = Exception(
                "Internal traceback with /var/www/app/service.py line 123"
            )

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_single_encoding(mock_upload_file)

            # Error message should be generic, not expose internal paths
            assert "/var/www" not in exc_info.value.detail
            assert "service.py" not in exc_info.value.detail
            assert (
                "Failed to process image" in exc_info.value.detail
                or "Could not process" in exc_info.value.detail
            )

    @pytest.mark.asyncio
    async def test_user_friendly_validation_messages(self, mock_upload_files_list):
        """Test that validation error messages are helpful."""
        service = ImageProcessingService()

        with patch("face_recognition.face_locations") as mock_locations:
            mock_locations.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                await service.extract_multiple_encodings(
                    mock_upload_files_list, required_count=5
                )

            # Should provide actionable feedback
            assert "No face detected" in exc_info.value.detail
            assert (
                "clearly visible" in exc_info.value.detail
                or "well-lit" in exc_info.value.detail
            )
