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


def configure_tesseract() -> None:
    configured_path = os.getenv("TESSERACT_CMD")

    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = configured_path
        return

    system_path = shutil.which("tesseract")

    if system_path:
        pytesseract.pytesseract.tesseract_cmd = system_path
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


def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image was not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "The selected file is not a supported image."
        )

    return image


def decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

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
            "The uploaded file could not be decoded as an image."
        )

    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    enlarged = cv2.resize(
        grayscale,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )

    denoised = cv2.bilateralFilter(
        enlarged,
        9,
        75,
        75,
    )

    thresholded = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    return thresholded


def calculate_ocr_confidence(
    processed_image: np.ndarray,
) -> float:
    ocr_data = pytesseract.image_to_data(
        processed_image,
        lang="eng",
        config="--oem 3 --psm 6",
        output_type=Output.DICT,
    )

    valid_confidences = []

    for confidence in ocr_data["conf"]:
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            continue

        if numeric_confidence >= 0:
            valid_confidences.append(numeric_confidence)

    if not valid_confidences:
        return 0.0

    return round(
        sum(valid_confidences) / len(valid_confidences),
        2,
    )


def clean_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        " ".join(line.split())
        for line in text.split("\n")
        if line.strip()
    ]

    return "\n".join(lines).strip()


def normalise_url_text(text: str) -> str:
    replacements = {
        "https : //": "https://",
        "https: //": "https://",
        "https ://": "https://",
        "http : //": "http://",
        "http: //": "http://",
        "http ://": "http://",
        "www .": "www.",
        ". com": ".com",
        ". co.za": ".co.za",
        ". org": ".org",
        ". net": ".net",
    }

    normalised = text

    for original, replacement in replacements.items():
        normalised = normalised.replace(
            original,
            replacement,
        )

    return normalised


def extract_urls(text: str) -> list[str]:
    normalised_text = normalise_url_text(text)

    url_pattern = re.compile(
        r"(?i)\b(?:https?://|www\.)"
        r"[a-z0-9][a-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*"
    )

    detected_urls = []

    for match in url_pattern.findall(normalised_text):
        cleaned_url = match.strip(
            " \t\r\n.,;:!?\"'()[]{}<>"
        )

        if cleaned_url and cleaned_url not in detected_urls:
            detected_urls.append(cleaned_url)

    return detected_urls


def perform_ocr(
    image: np.ndarray,
    content_type: str = "general",
) -> dict:
    configure_tesseract()

    processed_image = preprocess_image(image)

    extracted_text = pytesseract.image_to_string(
        processed_image,
        lang="eng",
        config="--oem 3 --psm 6",
    )

    extracted_text = clean_ocr_text(
        extracted_text
    )

    confidence = calculate_ocr_confidence(
        processed_image
    )

    result = {
        "content_type": content_type,
        "text": extracted_text,
        "ocr_confidence": confidence,
        "urls": [],
    }

    if content_type.lower() == "url":
        result["urls"] = extract_urls(
            extracted_text
        )

    return result


def perform_ocr_from_path(
    image_path: str | Path,
    content_type: str = "general",
) -> dict:
    image = load_image(image_path)

    return perform_ocr(
        image,
        content_type,
    )


def perform_ocr_from_bytes(
    image_bytes: bytes,
    content_type: str = "general",
) -> dict:
    image = decode_image(image_bytes)

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

    print("\nExtracted text:")
    print(ocr_result["text"])

    print(
        "\nOCR confidence:",
        f'{ocr_result["ocr_confidence"]}%',
    )

    if selected_content_type.lower() == "url":
        print("\nDetected URLs:")

        if ocr_result["urls"]:
            for detected_url in ocr_result["urls"]:
                print(f"- {detected_url}")
        else:
            print("- No URL was detected.")