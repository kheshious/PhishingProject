from pathlib import Path
import os
import re
import shutil
import sys

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


WINDOWS_TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


ALLOWED_CONTENT_TYPES = {
    "general",
    "url",
    "sms",
    "email",
}


def configure_tesseract() -> None:
    """
    Locate the Tesseract executable on Windows,
    Linux or Raspberry Pi.
    """

    configured_path = os.getenv(
        "TESSERACT_CMD"
    )

    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = (
            configured_path
        )

        return

    system_path = shutil.which(
        "tesseract"
    )

    if system_path:
        pytesseract.pytesseract.tesseract_cmd = (
            system_path
        )

        return

    if WINDOWS_TESSERACT_PATH.exists():
        pytesseract.pytesseract.tesseract_cmd = str(
            WINDOWS_TESSERACT_PATH
        )

        return

    raise RuntimeError(
        "Tesseract OCR could not be located. "
        "Install Tesseract or set the TESSERACT_CMD "
        "environment variable."
    )


def load_image(
    image_path: str | Path,
) -> np.ndarray:
    """
    Load an image from a file path.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image was not found: {image_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            "The selected file is not a supported image."
        )

    return image


def decode_image(
    image_bytes: bytes,
) -> np.ndarray:
    """
    Convert uploaded image bytes into an OpenCV image.
    """

    if not image_bytes:
        raise ValueError(
            "The uploaded image is empty."
        )

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "The uploaded file could not be "
            "decoded as an image."
        )

    return image


def resize_for_ocr(
    image: np.ndarray,
) -> np.ndarray:
    """
    Enlarge images containing small text.

    Large images are kept close to their original
    size to avoid unnecessary processing.
    """

    height, width = image.shape[:2]

    target_width = 1800

    if width < target_width:
        scale = target_width / width
    elif width > 2600:
        scale = 2600 / width
    else:
        scale = 1.0

    if scale == 1.0:
        return image.copy()

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    return resized


def add_white_border(
    image: np.ndarray,
) -> np.ndarray:
    """
    Add a white border around the image.

    Tesseract performs better when text does not
    touch the edge of an image.
    """

    return cv2.copyMakeBorder(
        image,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def create_image_versions(
    image: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """
    Produce different versions of the same image.

    Uploaded images and webcam images can have very
    different lighting, contrast and sharpness. Each
    version is tested and the best OCR result is kept.
    """

    resized = resize_for_ocr(
        image
    )

    grayscale = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    # Improve local contrast without making the
    # entire image excessively dark or bright.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    contrast = clahe.apply(
        grayscale
    )

    # Remove camera noise while preserving text edges.
    denoised = cv2.bilateralFilter(
        contrast,
        7,
        50,
        50,
    )

    # Sharpen blurred webcam characters.
    blurred = cv2.GaussianBlur(
        contrast,
        (0, 0),
        sigmaX=2.0,
    )

    sharpened = cv2.addWeighted(
        contrast,
        1.8,
        blurred,
        -0.8,
        0,
    )

    # Global threshold works well on clean,
    # evenly lit backgrounds.
    otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    # Threshold the sharpened image.
    sharpened_otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    # Adaptive threshold helps when a photographed
    # screen has uneven brightness or glare.
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        15,
    )

    versions = [
        (
            "grayscale",
            add_white_border(
                grayscale
            ),
        ),
        (
            "contrast",
            add_white_border(
                contrast
            ),
        ),
        (
            "denoised",
            add_white_border(
                denoised
            ),
        ),
        (
            "sharpened",
            add_white_border(
                sharpened
            ),
        ),
        (
            "otsu",
            add_white_border(
                otsu
            ),
        ),
        (
            "sharpened_otsu",
            add_white_border(
                sharpened_otsu
            ),
        ),
        (
            "adaptive",
            add_white_border(
                adaptive
            ),
        ),
    ]

    return versions


def clean_ocr_text(
    text: str,
) -> str:
    """
    Clean Tesseract output while preserving useful
    line breaks for SMS and email content.
    """

    if not text:
        return ""

    text = text.replace(
        "\x0c",
        ""
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    cleaned_lines = []

    for line in text.split("\n"):
        cleaned_line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if cleaned_line:
            cleaned_lines.append(
                cleaned_line
            )

    return "\n".join(
        cleaned_lines
    ).strip()


def normalise_url_text(
    text: str,
) -> str:
    """
    Correct common spaces and OCR mistakes inside URLs.
    """

    normalised = text.strip()

    normalised = re.sub(
        r"https\s*:\s*/\s*/",
        "https://",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"http\s*:\s*/\s*/",
        "http://",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"www\s*\.\s*",
        "www.",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"\s*\.\s*com\b",
        ".com",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"\s*\.\s*co\s*\.\s*za\b",
        ".co.za",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"\s*\.\s*org\b",
        ".org",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"\s*\.\s*net\b",
        ".net",
        normalised,
        flags=re.IGNORECASE,
    )

    normalised = re.sub(
        r"\s*/\s*",
        "/",
        normalised,
    )

    return normalised


def extract_urls(
    text: str,
) -> list[str]:
    """
    Extract complete URLs from OCR text.
    """

    normalised_text = normalise_url_text(
        text
    )

    url_pattern = re.compile(
        r"""
        (?:
            https?://
            |
            www\.
        )
        [a-z0-9]
        [a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    detected_urls = []

    for match in url_pattern.findall(
        normalised_text
    ):
        cleaned_url = match.strip(
            " \t\r\n,;!?\"'()[]{}<>"
        )

        # Remove a final full stop only when it appears
        # to be sentence punctuation.
        cleaned_url = cleaned_url.rstrip(
            "."
        )

        if (
            cleaned_url
            and cleaned_url
            not in detected_urls
        ):
            detected_urls.append(
                cleaned_url
            )

    return detected_urls


def get_page_modes(
    content_type: str,
) -> list[int]:
    """
    Return suitable Tesseract page-segmentation modes.

    URL images usually contain a single line.
    SMS and email images normally contain multiple lines.
    """

    content_type = content_type.lower()

    if content_type == "url":
        return [
            7,
            8,
            6,
            11,
        ]

    if content_type in {
        "sms",
        "email",
    }:
        return [
            6,
            3,
            11,
            12,
        ]

    return [
        6,
        3,
        11,
    ]


def calculate_readability(
    text: str,
) -> float:
    """
    Estimate whether OCR output contains readable text
    rather than mostly random symbols.
    """

    visible_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    if not visible_characters:
        return 0.0

    readable_characters = sum(
        1
        for character in visible_characters
        if (
            character.isalnum()
            or character
            in ".,:;!?@#%&()[]{}'\"/_-+=\\"
        )
    )

    return (
        readable_characters
        / len(visible_characters)
    ) * 100


def calculate_candidate_score(
    text: str,
    confidence: float,
    content_type: str,
) -> float:
    """
    Combine OCR confidence with text readability and
    useful-content checks.

    This prevents a short but incorrect result from
    winning only because it has a high confidence.
    """

    if not text:
        return -1000.0

    readability = calculate_readability(
        text
    )

    words = re.findall(
        r"[A-Za-z0-9]{2,}",
        text,
    )

    alphabetic_words = [
        word
        for word in words
        if any(
            character.isalpha()
            for character in word
        )
    ]

    score = (
        confidence * 0.65
        + readability * 0.20
    )

    content_type = content_type.lower()

    if content_type == "url":
        detected_urls = extract_urls(
            text
        )

        if detected_urls:
            score += 40

            longest_url_length = max(
                len(url)
                for url in detected_urls
            )

            score += min(
                longest_url_length,
                80,
            ) * 0.10

        if "http://" in text.lower():
            score += 8

        if "https://" in text.lower():
            score += 8

        if "." in text:
            score += 3

    else:
        # Reward candidates that preserve more meaningful
        # words from an SMS or email.
        score += min(
            len(alphabetic_words),
            30,
        ) * 0.70

        score += min(
            len(text),
            300,
        ) * 0.025

        if len(alphabetic_words) < 3:
            score -= 20

    return round(
        score,
        4,
    )


def run_ocr_candidate(
    processed_image: np.ndarray,
    content_type: str,
    page_mode: int,
) -> dict:
    """
    Run Tesseract on one image version using one
    page-segmentation mode.
    """

    configuration = (
        f"--oem 3 --psm {page_mode} "
        "-c preserve_interword_spaces=1"
    )

    extracted_text = pytesseract.image_to_string(
        processed_image,
        lang="eng",
        config=configuration,
    )

    extracted_text = clean_ocr_text(
        extracted_text
    )

    ocr_data = pytesseract.image_to_data(
        processed_image,
        lang="eng",
        config=configuration,
        output_type=Output.DICT,
    )

    valid_confidences = []

    for index, raw_confidence in enumerate(
        ocr_data["conf"]
    ):
        word = str(
            ocr_data["text"][index]
        ).strip()

        try:
            confidence = float(
                raw_confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if word and confidence >= 0:
            valid_confidences.append(
                confidence
            )

    if valid_confidences:
        average_confidence = (
            sum(valid_confidences)
            / len(valid_confidences)
        )
    else:
        average_confidence = 0.0

    average_confidence = round(
        average_confidence,
        2,
    )

    score = calculate_candidate_score(
        extracted_text,
        average_confidence,
        content_type,
    )

    return {
        "text": extracted_text,
        "ocr_confidence": average_confidence,
        "score": score,
        "page_mode": page_mode,
    }


def perform_ocr(
    image: np.ndarray,
    content_type: str = "general",
) -> dict:
    """
    Run OCR using multiple preprocessing methods and
    multiple Tesseract page-segmentation modes.

    The most suitable result is returned.
    """

    configure_tesseract()

    content_type = str(
        content_type
    ).strip().lower()

    if content_type not in ALLOWED_CONTENT_TYPES:
        content_type = "general"

    image_versions = create_image_versions(
        image
    )

    page_modes = get_page_modes(
        content_type
    )

    candidates = []

    for version_name, processed_image in image_versions:
        for page_mode in page_modes:
            try:
                candidate = run_ocr_candidate(
                    processed_image,
                    content_type,
                    page_mode,
                )

                candidate["processing_method"] = (
                    version_name
                )

                if candidate["text"]:
                    candidates.append(
                        candidate
                    )

            except pytesseract.TesseractError:
                continue

    if not candidates:
        return {
            "content_type": content_type,
            "text": "",
            "ocr_confidence": 0.0,
            "urls": [],
            "processing_method": None,
            "page_mode": None,
        }

    best_candidate = max(
        candidates,
        key=lambda candidate: candidate["score"],
    )

    extracted_text = best_candidate["text"]

    detected_urls = []

    if content_type == "url":
        detected_urls = extract_urls(
            extracted_text
        )

        # When a URL was found, return the normalised URL
        # as the main extracted text as well.
        if detected_urls:
            extracted_text = detected_urls[0]

    result = {
        "content_type": content_type,
        "text": extracted_text,
        "ocr_confidence": best_candidate[
            "ocr_confidence"
        ],
        "urls": detected_urls,
        "processing_method": best_candidate[
            "processing_method"
        ],
        "page_mode": best_candidate[
            "page_mode"
        ],
    }

    return result


def perform_ocr_from_path(
    image_path: str | Path,
    content_type: str = "general",
) -> dict:
    """
    Run OCR on an image stored on disk.
    """

    image = load_image(
        image_path
    )

    return perform_ocr(
        image,
        content_type,
    )


def perform_ocr_from_bytes(
    image_bytes: bytes,
    content_type: str = "general",
) -> dict:
    """
    Run OCR on an uploaded or webcam-captured image.
    """

    image = decode_image(
        image_bytes
    )

    return perform_ocr(
        image,
        content_type,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python machine_learning/ocr_service.py "
            '"path_to_image" [url|sms|email]'
        )

        raise SystemExit(1)

    selected_image_path = (
        sys.argv[1]
    )

    selected_content_type = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else "general"
    )

    ocr_result = perform_ocr_from_path(
        selected_image_path,
        selected_content_type,
    )

    print(
        "\nExtracted text:"
    )

    print(
        ocr_result["text"]
    )

    print(
        "\nOCR confidence:",
        f'{ocr_result["ocr_confidence"]}%',
    )

    print(
        "\nProcessing method:",
        ocr_result.get(
            "processing_method"
        ),
    )

    print(
        "Tesseract page mode:",
        ocr_result.get(
            "page_mode"
        ),
    )

    if selected_content_type.lower() == "url":
        print(
            "\nDetected URLs:"
        )

        if ocr_result["urls"]:
            for detected_url in ocr_result["urls"]:
                print(
                    f"- {detected_url}"
                )
        else:
            print(
                "- No URL was detected."
            )