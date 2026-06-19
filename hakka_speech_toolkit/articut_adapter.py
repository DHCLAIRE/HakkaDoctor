"""Optional adapter for Droidtown's ArticutAPI_Hakka package.

This module does not vendor or reimplement Droidtown's dictionaries or parser.
When the external package is installed and credentials are provided, it imports
the real ``ArticutAPI_Hakka`` client and delegates parsing to it. The local
``HakkaRuleParser`` remains available as a transparent fallback for offline
research, tests, and grammar prototyping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class ArticutHakkaUnavailable(RuntimeError):
    """Raised when the optional ArticutAPI_Hakka backend cannot be loaded."""


def load_articut_hakka_class():
    """Import Droidtown's ArticutAPI_Hakka class without making it mandatory."""

    try:
        from ArticutAPI_Hakka import ArticutHKK

        return ArticutHKK
    except ImportError as package_error:
        try:
            from ArticutAPI_Hakka.ArticutAPI_Hakka import ArticutHKK

            return ArticutHKK
        except ImportError as module_error:
            raise ArticutHakkaUnavailable(
                "ArticutAPI_Hakka is not installed. Install Droidtown's package "
                "and provide Articut credentials, or use HakkaRuleParser offline."
            ) from module_error or package_error


class ArticutHakkaParser:
    """Parser facade that delegates to Droidtown ArticutAPI_Hakka when available."""

    def __init__(
        self,
        username: str = "",
        apikey: str = "",
        usernameENG: str = "",
        apikeyENG: str = "",
        articut_client: Any | None = None,
        allow_fallback: bool = True,
        fallback_parser: Any | None = None,
    ) -> None:
        """Create an Articut client or remember why fallback is needed."""

        self.allow_fallback = allow_fallback
        self.fallback_parser = fallback_parser
        self.client = articut_client
        self.unavailable_reason: str | None = None

        if self.client is None:
            try:
                articut_cls = load_articut_hakka_class()
                self.client = articut_cls(
                    username=username,
                    apikey=apikey,
                    usernameENG=usernameENG,
                    apikeyENG=apikeyENG,
                )
            except ArticutHakkaUnavailable as error:
                self.unavailable_reason = str(error)
                if not allow_fallback:
                    raise

    def parse(
        self,
        input_text: str,
        level: str = "lv2",
        userDefinedDictFILE: str | Path | None = None,
        convert: str | None = None,
        user_defined_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse text with ArticutAPI_Hakka, falling back offline if unavailable."""

        if self.client is not None:
            kwargs: Dict[str, Any] = {"level": level}
            if userDefinedDictFILE is not None:
                kwargs["userDefinedDictFILE"] = str(userDefinedDictFILE)
            if convert is not None:
                kwargs["convert"] = convert
            result = self.client.parse(input_text, **kwargs)
            return self._normalize_articut_result(result, input_text)

        if not self.allow_fallback:
            raise ArticutHakkaUnavailable(self.unavailable_reason or "ArticutAPI_Hakka is unavailable.")

        parser = self.fallback_parser
        if parser is None:
            from .parser import HakkaRuleParser

            parser = HakkaRuleParser(user_defined_dict=user_defined_dict)
            self.fallback_parser = parser
        result = parser.parse(
            input_text,
            level=level,
            userDefinedDictFILE=userDefinedDictFILE,
            user_defined_dict=user_defined_dict,
        )
        result["parser_backend"] = "offline_fallback"
        result["articut_unavailable_reason"] = self.unavailable_reason
        return result

    @staticmethod
    def _normalize_articut_result(result: Dict[str, Any], input_text: str) -> Dict[str, Any]:
        """Attach backend metadata to a Droidtown ArticutAPI_Hakka result."""

        normalized = dict(result)
        normalized.setdefault("input", input_text)
        normalized["parser"] = "ArticutHakkaParser"
        normalized["parser_backend"] = "Droidtown ArticutAPI_Hakka"
        normalized["parser_strategy"] = [
            "external_articutapi_hakka",
            "droidtown_hakka_lexicon",
            "droidtown_pos_shift",
            "articut_mandarin_backbone",
        ]
        return normalized


class HakkaParser:
    """Unified parser interface.

    ``backend='articut'`` requires Droidtown's package and credentials.
    ``backend='offline'`` uses this repository's rule parser.
    ``backend='auto'`` tries Articut first and falls back offline.
    """

    def __init__(
        self,
        backend: str = "auto",
        user_defined_dict: Optional[Dict[str, Any]] = None,
        **articut_kwargs: Any,
    ) -> None:
        """Create a unified parser using Articut, offline rules, or auto fallback."""

        self.backend = backend
        if backend not in {"auto", "articut", "offline"}:
            raise ValueError("backend must be one of: auto, articut, offline")

        if backend == "offline":
            from .parser import HakkaRuleParser

            self.parser = HakkaRuleParser(user_defined_dict=user_defined_dict)
        else:
            self.parser = ArticutHakkaParser(
                allow_fallback=backend == "auto",
                fallback_parser=None,
                **articut_kwargs,
            )
            self.user_defined_dict = user_defined_dict

    def parse(self, input_text: str, **kwargs: Any) -> Dict[str, Any]:
        """Parse text through the configured backend."""

        if self.backend in {"auto", "articut"}:
            kwargs.setdefault("user_defined_dict", getattr(self, "user_defined_dict", None))
        return self.parser.parse(input_text, **kwargs)
