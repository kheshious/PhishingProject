from pathlib import Path

import os
import re
import shutil
import sys
import time

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
    Locate Tesseract on Windows, Linux or Raspberry Pi.
    """

    configured_path = os.getenv(
        "TESSERACT_CMD",
        "",
    ).strip()

    if configured_path:
        if not Path(configured_path).exists():
            raise RuntimeError(
                "The TESSERACT_CMD path does not exist: "
                f"{configured_path}"
            )

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
        "Install Tesseract or set TESSERACT_CMD."
    )


def load_image(
    image_path: str | Path,
) -> np.ndarray:
    """
    Load an image from disk.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image was not found: {image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
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
    Decode an uploaded or webcam-captured image.
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
            "The uploaded file could not be decoded "
            "as an image."
        )

    return image


def resize_for_ocr(
    image: np.ndarray,
) -> np.ndarray:
    """
    Resize an image without making it unnecessarily large.
    """

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        raise ValueError(
            "The image has invalid dimensions."
        )

    minimum_width = 1200
    maximum_width = 1600

    if width < minimum_width:
        scale = minimum_width / width

    elif width > maximum_width:
        scale = maximum_width / width

    else:
        return image.copy()

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )

    interpolation = (
        cv2.INTER_CUBIC
        if scale > 1
        else cv2.INTER_AREA
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=interpolation,
    )


def add_white_border(
    image: np.ndarray,
) -> np.ndarray:
    """
    Add a border so text does not touch image edges.
    """

    return cv2.copyMakeBorder(
        image,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def prepare_ocr_images(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Create one primary and one fallback image.

    Primary:
        Otsu thresholding, which produced the best
        result in the SMS test.

    Fallback:
        Sharpened grayscale for images where thresholding
        removes useful characters.
    """

    resized = resize_for_ocr(
        image
    )

    grayscale = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    contrast = clahe.apply(
        grayscale
    )

    blurred = cv2.GaussianBlur(
        contrast,
        (0, 0),
        sigmaX=1.5,
    )

    sharpened = cv2.addWeighted(
        contrast,
        1.6,
        blurred,
        -0.6,
        0,
    )

    otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    return {
        "otsu": add_white_border(
            otsu
        ),
        "sharpened": add_white_border(
            sharpened
        ),
    }


def clean_ocr_text(
    text: str,
) -> str:
    """
    Clean OCR output while keeping line breaks.
    """

    if not text:
        return ""

    text = text.replace(
        "\x0c",
        "",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
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
    Fix common OCR spacing errors inside URLs.
    """

    normalised = str(
        text or ""
    ).strip()

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
        r"\s*\.\s*co\s*\.\s*za\b",
        ".co.za",
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
    Extract URLs from OCR text.
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

    for match in url_pattern.finditer(
        normalised_text
    ):
        cleaned_url = match.group(0).strip(
            " \t\r\n,;!?\"'()[]{}<>"
        )

        cleaned_url = cleaned_url.rstrip(
            "."
        )

        if (
            cleaned_url
            and cleaned_url not in detected_urls
        ):
            detected_urls.append(
                cleaned_url
            )

    return detected_urls


def build_text_from_ocr_data(
    ocr_data: dict,
) -> str:
    """
    Build text from image_to_data results.

    This means Tesseract is called only once per attempt.
    """

    lines = {}

    words = ocr_data.get(
        "text",
        [],
    )

    for index, raw_word in enumerate(words):
        word = str(
            raw_word
        ).strip()

        if not word:
            continue

        line_key = (
            ocr_data["block_num"][index],
            ocr_data["par_num"][index],
            ocr_data["line_num"][index],
        )

        if line_key not in lines:
            lines[line_key] = []

        lines[line_key].append(
            word
        )

    extracted_lines = [
        " ".join(line_words)
        for line_words in lines.values()
        if line_words
    ]

    return clean_ocr_text(
        "\n".join(extracted_lines)
    )


def calculate_confidence(
    ocr_data: dict,
) -> float:
    """
    Calculate average OCR word confidence.
    """

    valid_confidences = []

    words = ocr_data.get(
        "text",
        [],
    )

    confidences = ocr_data.get(
        "conf",
        [],
    )

    for word, raw_confidence in zip(
        words,
        confidences,
    ):
        word = str(
            word
        ).strip()

        if not word:
            continue

        try:
            confidence = float(
                raw_confidence
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if confidence >= 0:
            valid_confidences.append(
                confidence
            )

    if not valid_confidences:
        return 0.0

    return round(
        sum(valid_confidences)
        / len(valid_confidences),
        2,
    )


def calculate_readability(
    text: str,
) -> float:
    """
    Calculate how much of the OCR text is readable.
    """

    visible_characters = [
        character
        for character in text
        if not character.isspace()
    ]

    if not visible_characters:
        return 0.0

    allowed_punctuation = (
        ".,:;!?@#%&()[]{}'\"/_-+=\\"
    )

    readable_characters = sum(
        1
        for character in visible_characters
        if (
            character.isalnum()
            or character in allowed_punctuation
        )
    )

    return round(
        (
            readable_characters
            / len(visible_characters)
        )
        * 100,
        2,
    )


def calculate_candidate_score(
    text: str,
    confidence: float,
    content_type: str,
) -> float:
    """
    Score an OCR result.
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

    score = (
        confidence * 0.70
        + readability * 0.20
    )

    if content_type == "url":
        detected_urls = extract_urls(
            text
        )

        if detected_urls:
            score += 45

        if "http://" in text.lower():
            score += 8

        if "https://" in text.lower():
            score += 8

        if "." in text:
            score += 3

    else:
        score += min(
            len(words),
            30,
        ) * 0.60

        score += min(
            len(text),
            300,
        ) * 0.02

        if len(words) < 3:
            score -= 20

    return round(
        score,
        4,
    )


def run_ocr_attempt(
    processed_image: np.ndarray,
    content_type: str,
    page_mode: int,
    processing_method: str,
) -> dict:
    """
    Run one Tesseract attempt.
    """

    configuration = (
        f"--oem 3 --psm {page_mode} "
        "-c preserve_interword_spaces=1"
    )

    ocr_data = pytesseract.image_to_data(
        processed_image,
        lang="eng",
        config=configuration,
        output_type=Output.DICT,
    )

    extracted_text = build_text_from_ocr_data(
        ocr_data
    )

    confidence = calculate_confidence(
        ocr_data
    )

    score = calculate_candidate_score(
        extracted_text,
        confidence,
        content_type,
    )

    return {
        "text": extracted_text,
        "ocr_confidence": confidence,
        "score": score,
        "processing_method": processing_method,
        "page_mode": page_mode,
    }


def result_is_acceptable(
    candidate: dict,
    content_type: str,
) -> bool:
    """
    Decide whether the first attempt is good enough.

    A fallback attempt only runs when this returns False.
    """

    text = candidate.get(
        "text",
        "",
    )

    confidence = float(
        candidate.get(
            "ocr_confidence",
            0,
        )
    )

    if not text:
        return False

    if content_type == "url":
        return bool(
            extract_urls(
                text
            )
        )

    words = re.findall(
        r"[A-Za-z0-9]{2,}",
        text,
    )

    readability = calculate_readability(
        text
    )

    return (
        confidence >= 65
        and readability >= 85
        and len(words) >= 4
        and len(text) >= 20
    )


def perform_ocr(
    image: np.ndarray,
    content_type: str = "general",
) -> dict:
    """
    Run balanced OCR.

    The primary attempt uses Otsu with page mode 6.
    A second attempt runs only if the first result is poor.
    """

    started_at = time.perf_counter()

    configure_tesseract()

    content_type = str(
        content_type
    ).strip().lower()

    if content_type not in ALLOWED_CONTENT_TYPES:
        content_type = "general"

    prepared_images = prepare_ocr_images(
        image
    )

    if content_type == "url":
        primary_method = "otsu"
        primary_page_mode = 7

        fallback_method = "sharpened"
        fallback_page_mode = 6

    else:
        primary_method = "otsu"
        primary_page_mode = 6

        fallback_method = "sharpened"
        fallback_page_mode = 11

    candidates = []

    try:
        primary_candidate = run_ocr_attempt(
            prepared_images[primary_method],
            content_type,
            primary_page_mode,
            primary_method,
        )

        if primary_candidate["text"]:
            candidates.append(
                primary_candidate
            )

    except pytesseract.TesseractError:
        primary_candidate = None

    if (
        primary_candidate is None
        or not result_is_acceptable(
            primary_candidate,
            content_type,
        )
    ):
        try:
            fallback_candidate = run_ocr_attempt(
                prepared_images[fallback_method],
                content_type,
                fallback_page_mode,
                fallback_method,
            )

            if fallback_candidate["text"]:
                candidates.append(
                    fallback_candidate
                )

        except pytesseract.TesseractError:
            pass

    processing_seconds = round(
        time.perf_counter() - started_at,
        2,
    )

    if not candidates:
        return {
            "content_type": content_type,
            "text": "",
            "ocr_confidence": 0.0,
            "urls": [],
            "processing_method": None,
            "page_mode": None,
            "processing_seconds": processing_seconds,
            "ocr_attempts": 0,
        }

    best_candidate = max(
        candidates,
        key=lambda candidate: candidate["score"],
    )

    extracted_text = best_candidate[
        "text"
    ]

    detected_urls = extract_urls(
        extracted_text
    )

    if (
        content_type == "url"
        and detected_urls
    ):
        extracted_text = detected_urls[0]

    return {
        "content_type": content_type,
        "text": extracted_text,
        "ocr_confidence": best_candidate[
            "ocr_confidence"
        ],
        "urls": (
            detected_urls
            if content_type == "url"
            else []
        ),
        "processing_method": best_candidate[
            "processing_method"
        ],
        "page_mode": best_candidate[
            "page_mode"
        ],
        "processing_seconds": processing_seconds,
        "ocr_attempts": len(
            candidates
        ),
    }


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
    Run OCR on an uploaded or webcam image.
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

    selected_image_path = sys.argv[1]

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
        "Processing time:",
        f'{ocr_result["processing_seconds"]} seconds',
    )

    print(
        "OCR attempts:",
        ocr_result.get(
            "ocr_attempts"
        ),
    )

    print(
        "Processing method:",
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