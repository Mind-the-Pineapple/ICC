"""Unit tests for confound_standardization/confound_module.py."""
import numpy as np
import pytest
from pytest import approx

from ICC import icc


def test_not_implemented_config():
    """Test raise error when using not implemented configuration."""
    ratings = np.array([4, 2])
    with pytest.raises(ValueError):
        icc(ratings, model='foo', type='bar', unit='baz')


def test_one_subject_input():
    """Test raise error when using an input with only one subject."""
    ratings = np.array([4, 2])
    with pytest.raises(ValueError):
        icc(ratings, model='oneway', type='agreement', unit='single')


def test_icc_1_1_with_shrout_values():
    """Test ICC(1,1) with values from [1] ([2] and [3] are example in R and SPSS).

    References
    ----------
    [1] - Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses in assessing rater reliability."
     Psychological bulletin 86.2 (1979): 420.
    [2] - http://www.clinimetrics.nl/images/upload/files/Chapter%205/chapter%205_5_Calculation%20of%20ICC%20in%20SPSS.pdf
    [3] - http://finzi.psych.upenn.edu/R/library/psych/html/ICC.html
    """
    ratings = np.array([[9., 2., 5., 8.],
                        [6., 1., 3., 2.],
                        [8., 4., 6., 8.],
                        [7., 1., 2., 6.],
                        [10., 5., 6., 9.],
                        [6., 2., 4., 7.]])

    # ICC(1,1)
    coeff, Fvalue, df1, df2, pvalue, lbound, ubound = icc(ratings, model='oneway', type='agreement', unit='single')
    assert 0.1657418 == approx(coeff, abs=1e-3)
    assert 1.794678 == approx(Fvalue, abs=1e-3)
    assert df1 == 5
    assert df2 == 18
    assert 0.1647688083 == approx(pvalue, abs=1e-3)
    assert -0.09672220 == approx(lbound, abs=1e-3)
    assert 0.6433983 == approx(ubound, abs=1e-3)


def test_icc_2_1_with_shrout_values():
    """Test ICC(2,1) with values from [1] ([2] and [3] are example in R and SPSS).

    References
    ----------
    [1] - Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses in assessing rater reliability."
     Psychological bulletin 86.2 (1979): 420.
    [2] - http://www.clinimetrics.nl/images/upload/files/Chapter%205/chapter%205_5_Calculation%20of%20ICC%20in%20SPSS.pdf
    [3] - http://finzi.psych.upenn.edu/R/library/psych/html/ICC.html
    """
    ratings = np.array([[9., 2., 5., 8.],
                        [6., 1., 3., 2.],
                        [8., 4., 6., 8.],
                        [7., 1., 2., 6.],
                        [10., 5., 6., 9.],
                        [6., 2., 4., 7.]])

    # ICC(2,1)
    coeff, Fvalue, df1, df2, pvalue, lbound, ubound = icc(ratings, model='twoway', type='agreement', unit='single')
    assert 0.2897638 == approx(coeff, abs=1e-3)
    assert 11.027248 == approx(Fvalue, abs=1e-3)
    assert df1 == 5
    assert df2 == 15
    assert 0.0001345665 == approx(pvalue, abs=1e-3)
    assert 0.04290119 == approx(lbound, abs=1e-3)
    assert 0.6910706 == approx(ubound, abs=1e-3)


def test_icc_3_1_with_shrout_values():
    """Test ICC(3,1) with values from [1] ([2] and [3] are example in R and SPSS).

    References
    ----------
    [1] - Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses in assessing rater reliability."
     Psychological bulletin 86.2 (1979): 420.
    [2] - http://www.clinimetrics.nl/images/upload/files/Chapter%205/chapter%205_5_Calculation%20of%20ICC%20in%20SPSS.pdf
    [3] - http://finzi.psych.upenn.edu/R/library/psych/html/ICC.html
    """
    ratings = np.array([[9., 2., 5., 8.],
                        [6., 1., 3., 2.],
                        [8., 4., 6., 8.],
                        [7., 1., 2., 6.],
                        [10., 5., 6., 9.],
                        [6., 2., 4., 7.]])

    # ICC(3,1)
    coeff, Fvalue, df1, df2, pvalue, lbound, ubound = icc(ratings, model='twoway', type='consistency', unit='single')
    assert 0.7148407 == approx(coeff, abs=1e-3)
    assert 11.0272 == approx(Fvalue, abs=1e-3)
    assert df1 == 5
    assert df2 == 15
    assert 0.0001345665 == approx(pvalue, abs=1e-3)
    assert 0.41183413 == approx(lbound, abs=1e-3)
    assert 0.9258328 == approx(ubound, abs=1e-3)


def test_icc_1_k_with_shrout_values():
    """Test ICC(1,k) with values from [1] ([2] and [3] are example in R and SPSS).

    References
    ----------
    [1] - Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses in assessing rater reliability."
     Psychological bulletin 86.2 (1979): 420.
    [2] - http://www.clinimetrics.nl/images/upload/files/Chapter%205/chapter%205_5_Calculation%20of%20ICC%20in%20SPSS.pdf
    [3] - http://finzi.psych.upenn.edu/R/library/psych/html/ICC.html
    """
    ratings = np.array([[9., 2., 5., 8.],
                        [6., 1., 3., 2.],
                        [8., 4., 6., 8.],
                        [7., 1., 2., 6.],
                        [10., 5., 6., 9.],
                        [6., 2., 4., 7.]])

    # ICC(1,k)
    coeff, Fvalue, df1, df2, pvalue, lbound, ubound = icc(ratings, model='oneway', type='agreement', unit='average')
    assert 0.4427971 == approx(coeff, abs=1e-3)
    assert 1.794678 == approx(Fvalue, abs=1e-3)
    assert df1 == 5
    assert df2 == 18
    assert 0.1647688083 == approx(pvalue, abs=1e-3)
    assert -0.54504172 == approx(lbound, abs=1e-3)
    assert 0.8783010 == approx(ubound, abs=1e-3)


def test_icc_2_k_with_shrout_values():
    """Test ICC(2,k) with values from [1] ([2] and [3] are example in R and SPSS).

    References
    ----------
    [1] - Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses in assessing rater reliability."
     Psychological bulletin 86.2 (1979): 420.
    [2] - http://www.clinimetrics.nl/images/upload/files/Chapter%205/chapter%205_5_Calculation%20of%20ICC%20in%20SPSS.pdf
    [3] - http://finzi.psych.upenn.edu/R/library/psych/html/ICC.html
    """
    ratings = np.array([[9., 2., 5., 8.],
                        [6., 1., 3., 2.],
                        [8., 4., 6., 8.],
                        [7., 1., 2., 6.],
                        [10., 5., 6., 9.],
                        [6., 2., 4., 7.]])

    # ICC(2,k)
    coeff, Fvalue, df1, df2, pvalue, lbound, ubound = icc(ratings, model='twoway', type='agreement', unit='average')
    assert 0.6200505 == approx(coeff, abs=1e-3)
    assert 11.027248 == approx(Fvalue, abs=1e-3)
    assert df1 == 5
    assert df2 == 15
    assert 0.0001345665 == approx(pvalue, abs=1e-3)
    assert 0.15203705 == approx(lbound, abs=1e-3)
    assert 0.8994767 == approx(ubound, abs=1e-3)


def test_icc_3_k_with_shrout_values():
    """Test ICC(3,k) with values from [1] ([2] and [3] are example in R and SPSS).

    References
    ----------
    [1] - Shrout, Patrick E., and Joseph L. Fleiss. "Intraclass correlations: uses in assessing rater reliability."
     Psychological bulletin 86.2 (1979): 420.
    [2] - http://www.clinimetrics.nl/images/upload/files/Chapter%205/chapter%205_5_Calculation%20of%20ICC%20in%20SPSS.pdf
    [3] - http://finzi.psych.upenn.edu/R/library/psych/html/ICC.html
    """
    ratings = np.array([[9., 2., 5., 8.],
                        [6., 1., 3., 2.],
                        [8., 4., 6., 8.],
                        [7., 1., 2., 6.],
                        [10., 5., 6., 9.],
                        [6., 2., 4., 7.]])

    # ICC(3,k)
    coeff, Fvalue, df1, df2, pvalue, lbound, ubound = icc(ratings, model='twoway', type='consistency', unit='average')
    assert 0.9093155 == approx(coeff, abs=1e-3)
    assert 11.027248 == approx(Fvalue, abs=1e-3)
    assert df1 == 5
    assert df2 == 15
    assert 0.0001345665 == approx(pvalue, abs=1e-3)
    assert 0.73689768 == approx(lbound, abs=1e-3)
    assert 0.9803661 == approx(ubound, abs=1e-3)


def test_with_koo_and_li_values_case_1():
    """Test icc method with values from fig 2 from [1].


    References
    ----------
    [1] - Koo, Terry K., and Mae Y. Li. "A guideline of selecting and reporting
     intraclass correlation coefficients for reliability research." Journal of
     chiropractic medicine 15.2 (2016): 155-163.
    """
    x = np.array([4, 8, 12, 16, 20])
    y = x
    ratings = np.vstack((x, y)).T

    # Single Measurement:
    # One-Way Random, absolute [ICC(1,1)] = 1.00
    coeff, _, _, _, _, _, _ = icc(ratings, model='oneway', type='agreement', unit='single')
    assert 1.00 == approx(coeff, abs=1e-2)
    # Two-Way Random, absolute [ICC(2,1)] = 1.00
    coeff, _, _, _, _, _, _ = icc(ratings, model='twoway', type='agreement', unit='single')
    assert 1.00 == approx(coeff, abs=1e-2)
    # Two-Way Mixed, consistency [ICC(3,1)] = 1.00
    coeff, _, _, _, _, _, _ = icc(ratings, model='twoway', type='consistency', unit='single')
    assert 1.00 == approx(coeff, abs=1e-2)

    # Mean Measurement:
    # One-Way Random, absolute [ICC(1,k)] = 1.00
    coeff, _, _, _, _, _, _ = icc(ratings, model='oneway', type='agreement', unit='average')
    assert 1.00 == approx(coeff, abs=1e-2)
    # Two-Way Random, absolute [ICC(2,k)] = 1.00
    coeff, _, _, _, _, _, _ = icc(ratings, model='twoway', type='agreement', unit='average')
    assert 1.00 == approx(coeff, abs=1e-2)
    # Two-Way Mixed, absolute [ICC(3,k)]= 1.00
    coeff, _, _, _, _, _, _ = icc(ratings, model='twoway', type='consistency', unit='average')
    assert 1.00 == approx(coeff, abs=1e-2)
