"""Hakka speech processing toolkit."""

__all__ = [
    "AccentEvaluator",
    "HakkaAccentConverter",
    "HakkaSpeechModel",
    "HakkaSTT",
    "HakkaTTS",
    "HakkaRuleParser",
    "ArticutLikeHakkaParser",
    "build_user_defined_dict",
]


def __getattr__(name: str):
    if name == "AccentEvaluator":
        from .accent import AccentEvaluator

        return AccentEvaluator
    if name == "HakkaAccentConverter":
        from .conversion import HakkaAccentConverter

        return HakkaAccentConverter
    if name in {"HakkaSpeechModel", "HakkaSTT", "HakkaTTS"}:
        from .models import HakkaSpeechModel, HakkaSTT, HakkaTTS

        return {
            "HakkaSpeechModel": HakkaSpeechModel,
            "HakkaSTT": HakkaSTT,
            "HakkaTTS": HakkaTTS,
        }[name]
    if name in {"HakkaRuleParser", "ArticutLikeHakkaParser", "build_user_defined_dict"}:
        from .parser import ArticutLikeHakkaParser, HakkaRuleParser, build_user_defined_dict

        return {
            "HakkaRuleParser": HakkaRuleParser,
            "ArticutLikeHakkaParser": ArticutLikeHakkaParser,
            "build_user_defined_dict": build_user_defined_dict,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
