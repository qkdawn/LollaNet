from research.phase1.value_investing.download_sec_mda import locate_legacy_item6, locate_mda, locate_mda_dom, normalize_html, promote_semantic_locator


def test_locate_10q_mda_ignores_short_table_of_contents_hit() -> None:
    raw = b"<html><body>Item 2 Item 3 " + b"x " * 40 + b"<h1>Item 2. Management's Discussion and Analysis</h1>" + b"The following discussion covers operations and liquidity " * 100 + b"<h1>Item 3. Quantitative Disclosures</h1></body></html>"
    text = normalize_html(raw)
    result = locate_mda(text, "10-Q")
    assert result["status"] == "located"
    assert "operations and liquidity" in text[result["start_char"]:result["end_char"]]


def test_locate_10k_mda_ends_before_item_7a() -> None:
    raw = b"<h1>Item 7. Management's Discussion and Analysis</h1>" + b"The following discussion covers management discussion " * 100 + b"<h1>Item 7A</h1>market risk"
    text = normalize_html(raw)
    result = locate_mda(text, "10-K")
    section = text[result["start_char"]:result["end_char"]]
    assert result["status"] == "located"
    assert "management discussion" in section
    assert "market risk" not in section


def test_unsupported_form_is_explicit() -> None:
    assert locate_mda("Item 2 text", "8-K")["status"] == "unsupported_form"


def test_formal_body_signal_wins_over_table_of_contents() -> None:
    text = (
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION 100 "
        + "contents " * 100
        + "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION. "
        + "The following discussion should be read with our financial statements. "
        + "operations and liquidity " * 100
        + "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES"
    )
    result = locate_mda(text, "10-K")
    assert result["status"] == "located"
    assert result["locator_quality"] == "high"
    assert text[result["start_char"]:result["end_char"]].startswith("ITEM 7. MANAGEMENT")


def test_dom_locator_follows_table_of_contents_anchor() -> None:
    raw = (
        b'<a href="#mda">Item 2. Management\'s Discussion and Analysis</a> 21 '
        b'<a id="mda"></a><h1>Item 2. Management\'s Discussion and Analysis</h1>'
        + b'operations liquidity cash flows ' * 100
        + b'<h1>Item 3. Quantitative Disclosures</h1>'
    )
    text = normalize_html(raw)
    result = locate_mda_dom(raw, text, "10-Q")
    assert result is not None
    assert result["locator_method"] == "toc_anchor"
    assert "operations liquidity" in text[result["start_char"]:result["end_char"]]


def test_legacy_10k_item6_mda_is_supported() -> None:
    text = "ITEM 6. MANAGEMENT'S DISCUSSION AND ANALYSIS " + "operations liquidity " * 100 + " ITEM 7A. MARKET RISK"
    result = locate_legacy_item6(text, "10-K")
    assert result is not None
    assert result["locator_method"] == "legacy_item6"


def test_semantic_locator_promotes_real_heading_without_fixed_opening() -> None:
    text = "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS " + "revenue liquidity cash flow " * 100 + " ITEM 3. MARKET RISK"
    locator = locate_mda(text, "10-Q")
    result = promote_semantic_locator(text, locator, "10-Q")
    assert result["status"] == "located"
    assert result["locator_method"] == "semantic_heading"


def test_semantic_locator_rejects_table_of_contents() -> None:
    text = "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS 21 Results of Operations 22 Liquidity 23 ITEM 3. MARKET RISK"
    locator = locate_mda(text, "10-Q")
    result = promote_semantic_locator(text, locator, "10-Q")
    assert result["status"] != "located"
