import pytest

from scripts.eval_bureau_with_min15 import canonical_from_extract


@pytest.mark.parametrize(
    ("sous_direction", "bureau", "expected"),
    [
        ("SD2", "Bureau 2B", "SD2/2B"),
        ("SD SP", "Bureau SP3", "SDSP/SP3"),
        ("SD1 B", "REDACTEURS", "SD1B"),
        ("SDRH1", "Chef de bureau", "SDRH1"),
        ("SDAS1", "Zonage, régulation territoriale", "SDAS1"),
        ("SD1", "MCGRM", "MCGRM"),
        ("DACI", "REDACTEURS", "DACI"),
        ("DACI", "Chargée de mission", "DACI"),
        ("SDAS", "Sous-Direction", None),
        ("SD3", "CAB S/DIR", None),
        ("CAB", "Cabinet", None),
        ("Centralisateur", "MDI", "CENTRALISATEUR/MDI"),
    ],
)
def test_min15_key_is_the_bureau_not_the_role(sous_direction, bureau, expected):
    assert canonical_from_extract(sous_direction, bureau) == expected
