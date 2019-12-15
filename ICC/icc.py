""" Script to implement Intraclass correlation coefficient (ICC).

Based on:
    https://www.rdocumentation.org/packages/irr/versions/0.84.1/topics/icc
"""
import numpy as np
from scipy.stats import f


def icc(ratings, model='oneway', type='consistency', unit='single', r0=0, confidence_level=0.95):
    """Implement Intraclass correlation coefficient (ICC) for oneway and twoway models.

    Computes single score or average score ICCs as an index of interrater reliability
    of quantitative data. Additionally, F-test and confidence interval are computed.

    When considering which form of ICC is appropriate for an actual set of data, one has
    take several decisions (Shrout & Fleiss, 1979): 1. Should only the subjects be considered
    as random effects ('"oneway"' model) or are subjects and raters randomly chosen from
    a bigger pool of persons ('"twoway"' model). 2. If differences in judges' mean ratings
    are of interest, interrater '"agreement"' instead of '"consistency"' should be computed.
    3. If the unit of analysis is a mean of several ratings, unit should be changed to '"average"'.
    In most cases, however, single values (unit='"single"') are regarded.

    Parameters
    ----------
    ratings: array-like, shape (n_subjects, n_raters)
        Matrix with n subjects m raters
    model: string, optional (default='oneway')
        String specifying if a 'oneway' model with row effects random, or a
        'twoway' model with column and row effects random should be applied.
    type: string, optional (default='consistency')
        String specifying if 'consistency' or 'agreement' between raters
        should be estimated. If a 'oneway' model is used, only 'consistency'
         could be computed.
    unit: string, optional (default='single')
        String specifying the unit of analysis: Must be one of 'single' or
        'average'.
    r0: float, optional (default=0.0)
        Specification of the null hypothesis r = r0 (default=0.0).
        Note that a one sided test (H1: r > r0) is performed.
    confidence_level: float, optional (default=0.95)
        Confidence level of the interval.

    Returns
    -------
    coeff: float
        The intraclass correlation coefficient.
    Fvalue: float
        The value of the F-statistic.
    df1: int
        The numerator degrees of freedom.
    df2: int
        The denominator degrees of freedom.
    pvalue: float
        The two-tailed p-value.
    lbound: float
        The lower bound of the confidence interval.
    ubound: float
        The upper bound of the confidence interval.

    References
    ----------
        [1] - Bartko, J.J. (1966). The intraclass correlation coefficient as a measure of reliability.
        Psychological Reports, 19, 3-11. McGraw, K.O., & Wong, S.P. (1996), Forming inferences about
        some intraclass correlation coefficients. Psychological Methods, 1, 30-46. Shrout, P.E., &
        Fleiss, J.L. (1979), Intraclass correlation: uses in assessing rater reliability. Psychological
        Bulletin, 86, 420-428.
    """
    n_subjects, n_raters = ratings.shape
    SStotal = np.var(ratings) * (n_subjects * n_raters - 1)
    alpha = 1 - confidence_level

    MSr = np.mean(ratings, axis=0) * n_raters
    MSw = np.sum(np.var(ratings, axis=0) / n_subjects)
    MSc = np.var(np.mean(ratings, axis=1)) * n_subjects
    MSe = (SStotal - MSr * (n_subjects - 1) - MSc * (n_raters - 1)) / ((n_subjects - 1) * (n_raters - 1))

    # Single Score ICCs
    if unit == 'single':
        if model == 'oneway':
            # Asendorpf & Wallbott, S. 245, ICu
            # Bartko (1966), [3]
            # icc.name < - "ICC(1)"
            coeff = (MSr - MSw) / (MSr + (n_raters - 1) * MSw)
            Fvalue = MSr / MSw * ((1 - r0) / (1 + (n_raters - 1) * r0))
            df1 = n_subjects - 1
            df2 = n_subjects * (n_raters - 1)
            pvalue = 1 - f.cdf(Fvalue, df1, df2)

            # confidence interval
            FL = (MSr / MSw) / f.ppf(1 - alpha / 2, n_subjects - 1, n_subjects * (n_raters - 1))
            FU = (MSr / MSw) * f.ppf(1 - alpha / 2, n_subjects * (n_raters - 1), n_subjects - 1)
            lbound = (FL - 1) / (FL + (n_raters - 1))
            ubound = (FU - 1) / (FU + (n_raters - 1))

        elif model == 'twoway':
            if type == 'consistency':
                # Asendorpf & Wallbott, S. 245, ICa
                # Bartko (1966), [21]
                # Shrout & Fleiss (1979), ICC(3,1)
                # icc.name < - "ICC(C,1)"
                coeff = (MSr - MSe) / (MSr + (n_raters - 1) * MSe)
                Fvalue = MSr / MSe * ((1 - r0) / (1 + (n_raters - 1) * r0))
                df1 = n_subjects - 1
                df2 = (n_subjects - 1) * (n_raters - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval
                FL = (MSr / MSe) / f.ppf(1 - alpha / 2, n_subjects - 1, (n_subjects - 1) * (n_raters - 1))
                FU = (MSr / MSe) * f.ppf(1 - alpha / 2, (n_subjects - 1) * (n_raters - 1), n_subjects - 1)
                lbound = (FL - 1) / (FL + (n_raters - 1))
                ubound = (FU - 1) / (FU + (n_raters - 1))

            elif type == 'agreement':
                # Asendorpf & Wallbott, S. 246, ICa'
                # Bartko (1966), [15]
                # Shrout & Fleiss (1979), ICC(2,1)
                # icc.name < - "ICC(A,1)"
                coeff = (MSr - MSe) / (MSr + (n_raters - 1) * MSe + (n_raters / n_subjects) * (MSc - MSe))
                a = (n_raters * r0) / (n_subjects * (1 - r0))
                b = 1 + (n_raters * r0 * (n_subjects - 1)) / (n_subjects * (1 - r0))

                Fvalue = MSr / (a * MSc + b * MSe)
                a = (n_raters * coeff) / (n_subjects * (1 - coeff))
                b = 1 + (n_raters * coeff * (n_subjects - 1)) / (n_subjects * (1 - coeff))
                v = (a * MSc + b * MSe) ^ 2 / (
                            (a * MSc) ^ 2 / (n_raters - 1) + (b * MSe) ^ 2 / ((n_subjects - 1) * (n_raters - 1)))
                df1 = n_subjects - 1
                df2 = v
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval (McGraw & Wong, 1996)
                FL = f.ppf(1 - alpha / 2, n_subjects - 1, v)
                FU = f.ppf(1 - alpha / 2, v, n_subjects - 1)
                lbound = (n_subjects * (MSr - FL * MSe)) / (FL * (
                            n_raters * MSc + (n_raters * n_subjects - n_raters - n_subjects) * MSe) + n_subjects * MSr)
                ubound = (n_subjects * (FU * MSr - MSe)) / (n_raters * MSc + (
                            n_raters * n_subjects - n_raters - n_subjects) * MSe + n_subjects * FU * MSr)

    elif unit == 'average':
        if model == 'oneway':
            # Asendorpf & Wallbott, S. 245, Ru
            # icc.name < - paste("ICC(", n_raters, ")", sep="")

            coeff = (MSr - MSw) / MSr
            Fvalue = MSr / MSw * (1 - r0)
            df1 = n_subjects - 1
            df2 = n_subjects * (n_raters - 1)
            pvalue = 1 - f.cdf(Fvalue, df1, df2)

            # confidence interval
            FL = (MSr / MSw) / f.ppf(1 - alpha / 2, n_subjects - 1, n_subjects * (n_raters - 1))
            FU = (MSr / MSw) * f.ppf(1 - alpha / 2, n_subjects * (n_raters - 1), n_subjects - 1)
            lbound = 1 - 1 / FL
            ubound = 1 - 1 / FU

        elif model == 'twoway':
            if type == 'consistency':
                # Asendorpf & Wallbott, S. 246, Ra
                # icc.name < - paste("ICC(C,", n_raters, ")", sep="")

                coeff = (MSr - MSe) / MSr
                Fvalue = MSr / MSe * (1 - r0)
                df1 = n_subjects - 1
                df2 = (n_subjects - 1) * (n_raters - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval
                FL = (MSr / MSe) / f.ppf(1 - alpha / 2, n_subjects - 1, (n_subjects - 1) * (n_raters - 1))
                FU = (MSr / MSe) * f.ppf(1 - alpha / 2, (n_subjects - 1) * (n_raters - 1), n_subjects - 1)
                lbound = 1 - 1 / FL
                ubound = 1 - 1 / FU

            elif type == 'agreement':
                # icc.name <- paste("ICC(A,",n_raters,")",sep="")
                coeff = (MSr - MSe) / (MSr + (MSc - MSe) / n_subjects)
                a = r0 / (n_subjects * (1 - r0))
                b = 1 + (r0 * (n_subjects - 1)) / (n_subjects * (1 - r0))

                Fvalue = MSr / (a * MSc + b * MSe)
                a = (n_raters * coeff) / (n_subjects * (1 - coeff))
                b = 1 + (n_raters * coeff * (n_subjects - 1)) / (n_subjects * (1 - coeff))
                v = (a * MSc + b * MSe) ^ 2 / (
                            (a * MSc) ^ 2 / (n_raters - 1) + (b * MSe) ^ 2 / ((n_subjects - 1) * (n_raters - 1)))
                df1 = n_subjects - 1
                df2 = v
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval (McGraw & Wong, 1996)
                FL = f.ppf(1 - alpha / 2, n_subjects - 1, v)
                FU = f.ppf(1 - alpha / 2, v, n_subjects - 1)
                lbound = (n_subjects * (MSr - FL * MSe)) / (FL * (MSc - MSe) + n_subjects * MSr)
                ubound = (n_subjects * (FU * MSr - MSe)) / (MSc - MSe + n_subjects * FU * MSr)

    return coeff, Fvalue, df1, df2, pvalue, lbound, ubound
