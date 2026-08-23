from urllib.parse import urlsplit
import ipaddress
import math
import re

import numpy as np


def calculate_entropy(text):

    if not text:
        return 0.0

    probabilities = [
        text.count(character) / len(text)
        for character in set(text)
    ]

    return -sum(
        probability * math.log2(probability)
        for probability in probabilities
    )


def extract_url_features(url):

    value = str(url).strip()

    parsing_value = value

    if "://" not in parsing_value:
        parsing_value = "http://" + parsing_value

    try:
        parsed = urlsplit(parsing_value)
    except ValueError:
        parsed = None

    if parsed is None:
        hostname = ""
        path = ""
        query = ""
    else:
        hostname = (
            parsed.hostname or ""
        ).lower()

        path = parsed.path or ""
        query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        is_ip_address = 1
    except ValueError:
        is_ip_address = 0

    hostname_labels = [
        part
        for part in hostname.split(".")
        if part
    ]

    subdomain_count = max(
        len(hostname_labels) - 2,
        0
    )

    path_parts = [
        part
        for part in path.split("/")
        if part
    ]

    digit_count = sum(
        character.isdigit()
        for character in value
    )

    letter_count = sum(
        character.isalpha()
        for character in value
    )

    special_character_count = sum(
        not character.isalnum()
        for character in value
    )

    hyphen_count = value.count("-")
    dot_count = value.count(".")
    slash_count = value.count("/")
    question_mark_count = value.count("?")
    equals_count = value.count("=")
    ampersand_count = value.count("&")
    at_symbol_count = value.count("@")
    underscore_count = value.count("_")
    percent_count = value.count("%")

    url_length = len(value)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    digit_ratio = (
        digit_count / url_length
        if url_length
        else 0.0
    )

    letter_ratio = (
        letter_count / url_length
        if url_length
        else 0.0
    )

    special_character_ratio = (
        special_character_count / url_length
        if url_length
        else 0.0
    )

    hostname_entropy = calculate_entropy(
        hostname
    )

    url_entropy = calculate_entropy(
        value
    )

    features = [
        url_length,
        hostname_length,
        path_length,
        query_length,
        dot_count,
        hyphen_count,
        slash_count,
        question_mark_count,
        equals_count,
        ampersand_count,
        at_symbol_count,
        underscore_count,
        percent_count,
        digit_count,
        letter_count,
        special_character_count,
        digit_ratio,
        letter_ratio,
        special_character_ratio,
        subdomain_count,
        len(path_parts),
        is_ip_address,
        hostname_entropy,
        url_entropy,
    ]

    return features


def extract_feature_matrix(urls):

    return np.array(
        [
            extract_url_features(url)
            for url in urls
        ],
        dtype=float,
    )