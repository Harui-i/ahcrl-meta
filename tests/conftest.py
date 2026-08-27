"""pytest 実行中だけ、ahcrl の jaxtyping 注釈を実行時検査する。"""

from jaxtyping import install_import_hook

# pytest はテストモジュールより先に conftest を import するため、このフックは
# ahcrl の各モジュールを import する際に jaxtyped + beartype を自動適用する。
_jaxtyping_import_hook = install_import_hook("ahcrl", "beartype.beartype")


def pytest_unconfigure() -> None:
    """pytest 終了時に import hook を解除する。"""
    _jaxtyping_import_hook.uninstall()
