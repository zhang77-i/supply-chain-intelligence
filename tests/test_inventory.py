from src.inventory.newsvendor import critical_ratio


def test_critical_ratio_range():
    ratio = critical_ratio(10, 5)
    assert 0 < ratio < 1
