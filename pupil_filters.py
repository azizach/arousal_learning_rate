'''
@author: raphfle
'''
import pandas as pd
import numpy as np

def noise_blinks(data, null_threshold=0, min_valid_block_size=0,
                 smoothing=False, win_type='hamming',
                 include_cutoff_blinks=False, min_length=0,
                 max_length=0.5):
    """
Eye blinks in pupillometry data are characterized by episodes of missing /
invalid data (such as 0 or negative dilations) flanked by a sudden drop from
and recovery to baseline. This function identifies episodes of missing data and
the characteristic drops by determining where the data stops increasing
monotonically when moving away from the onsetor offset. Returns a data frame
with rows containing indices for the last/first valid data point before and
after an episode of missing/invalid data and indices of data points where onset
and offset of anomalous data due to the eye blink are thought to be. Based on
an algorythm described in *Hershman, Ronen; Henik, Avishai; Cohen, Noga (2018):
A novel blink detection method based on pupillometry noise. In Behavior
research methods 50 (1), pp. 107–114. DOI: 10.3758/s13428-017-1008-1.*

Arguments:
data : pandas.Series or convertible.

null_threshold: float, default = 0.
    Data <= null_threshold will be considered missing. Missing data (NaN) will
    always be considered missing.

min_valid_block_size: int, default = 0.
    Can be used to consider closeby segments of invalid/missing data as one.
    Argument indicates how many valid need to seperate two missing values for
    them to be considered independent blinks. Values <=1 will disable
    this function.

smoothing: int or False, default = False.
    Allows to apply smoothing before identification of the point where values
    stop to increase monotonically. Argument indicates window size of the
    rolling mean. If <= 1 or False, data will remain unsmoothed. Useful if
    tracker has sufficiently high sample rate and noise characteristics that
    allow for single samples breaking the stereotypical pattern of
    monotonically increasing values, even though the general trend is still
    rising. Window size of 0.01 * sampling rate is proposed to yield a
    smoothing window of 10ms (Hershman, Henik, Cohen 2018).

win_type: str, default = 'hamming'.
    Window type to be used for smoothing. Will be ignored if *smoothed* <= 1 or
    **False**. If ``win_type=None`` all points are evenly weighted. To learn
    more about different window types see `scipy.signal window functions
    <https://docs.scipy.org/doc/scipy/reference/signal.html#window-functions>`.

include_cutoff_blinks: boolean, default = False.
    Whether or not to include episodes of missing/invalid data at the beginning
    or end of the series. If True, episodes of missing data at the beginning of
    the series will have their onset / last valid on the first data point, and
    respectively episodes at the end will have their offset/first valid on the
    last data point. If ´False´, only episodes surrounded by valid data will be
    considered.

´´´Returns´´´:
    pandas DataFrame where each row is a blink, described by indices of four
    data points:
    1. onset : last value before data monotonously drops until invalid / missing.
    2. last_valid : last valid value before episode of invalid / missing data.
    3. first_valid : first valid value after episode of invalid / missing data.
    4. offset : first value where data has stopped increasing monotonously after
    episode of invalid data.

    """
    assert isinstance(min_valid_block_size, int)
    try:
        data = pd.Series(data, copy=True)
    except Exception:
        raise TypeError(
            'Input must be of type pandas.Series or convertible to such.')
    # this line counts how many consecutive valid (or invalid)
    # data points we have.
    consecutive_valids = data.groupby((data.le(null_threshold) != data.shift(
            ).le(null_threshold)).cumsum()).transform('count')
    # in the following step, we replace all invalids with 0
    consecutive_valids = consecutive_valids.mask(data.le(null_threshold), 0)
    # this now gives us a series where each invalid data point is represented
    # by a 0, and each valid one by an integer indicating the number of valid
    # values around it (itself included)

    # which now allows us to replace all values, where we only have a few
    # valid values surrounded by invalids, which we deem not trustworthy
    data = data.mask(consecutive_valids < min_valid_block_size, null_threshold)

    # now we identify all data points directly before and after episodes of
    # invalid data, which are onsets & offsets for onset: find data points that
    # are the last valid values before invalids
    onsets = (data <= 0).astype(float).diff(-1)
    onsets = onsets[onsets == -1]

    # for offset: find data points that are the first valid values
    # after invalids
    offsets = (data <= 0).astype(float).diff(1)
    offsets = offsets[offsets == -1]

    # the next step is to determine where the data stops increasing
    # monotonically when moving away from the onset/offset

    # at this point, Hershman, Henik & Cohen suggest smoothing the data with a
    # 10ms window. Here, we'll just use number of samples as measure of window
    # width. If < 2, no smoothing takes place.
    if isinstance(smoothing, int) and (smoothing >= 2):
        smoothed = data.rolling(smoothing, center=True,
                                win_type=win_type).mean()
    elif isinstance(smoothing, int) or smoothing is False:
        smoothed = data
    else:
        raise ValueError('"smoothing must be int or False"')

    # compute a boolean data frame that tells us whether the next and the last
    # value are larger / smaller
    frame = pd.concat([smoothed.diff(1) < 0, smoothed.diff(-1) < 0], axis=1)
    frame.columns = ["last_one_bigger", "next_one_bigger"]

    onsets = onsets.index.to_series().apply(
        lambda x: frame.index[
                frame.next_one_bigger & (x-frame.index >= 0)].max())

    offsets = offsets.index.to_series().apply(
        lambda x: frame.index[
                frame.last_one_bigger & (x-frame.index <= 0)].min())

    # EDGE CASES
    # usually, we'd have as many onsets as offsets, as each onset is followed
    # by an offset. However, there are two edge cases we need to take care of:
    # blinks starting before the beginning of the time series (no onset) and
    # blinks being cut of at the end of the time series (no offset)

    # first: quick check whether all data conforms to our expectations about
    # the standard case
    try:
        # for each onset, there is an offset -> same number of onsets & offsets
        assert len(onsets) == len(offsets)
        # all onsets are earlier than their corresponding offset
        assert (onsets.index < offsets.index).all()
        # there is no offset that is earlier than the next onset
        assert not (onsets.index.to_series().shift(-1)
                < offsets.index).any()
        # there is no onset, for which the difference to its offset is larger
        # than its difference to the next onset
        assert not (offsets.index-onsets.index <
                onsets.index.to_series().diff(-1)).any()

    except AssertionError:
        print('Exceptional Data!')

    # FIX EDGE CASES

    # if there is no matching onset for the first offset, set first onset to
    # first data point:
    if not (onsets.index < offsets.index[0]).any():
        print('first onset missing')
        if data.iloc[0] <= null_threshold:
            print('reason: data begins with invalid / blink value')
            if include_cutoff_blinks:
                onsets = onsets.append(
                    pd.Series(index=[
                            data.index.min()], data=[data.index.min()]))
                onsets.sort_index(inplace=True)
                print('fixed by setting first data point as onset')
            else:
                offsets = offsets.iloc[1:]
                print('fixed by dropping first offset')

    # if there is no matching offset for the last onset:
    if not (offsets.index > onsets.index[-1]).any():
        print('last offset missing')
        if data.iloc[-1] <= null_threshold:
            print('reason: data ends on invalid / blink value')
            if include_cutoff_blinks:
                offsets = offsets.append(
                    pd.Series(index=[
                            data.index.max()], data=[data.index.max()]))
                offsets.sort_index(inplace=True)
                print('fixed by setting last data point as offset')
            else:
                onsets = onsets.iloc[:-1]
                print('fixed by dropping last onset')

    # re-check whether all data conforms to our expectations about the
    # standard case
    try:
        # for each onset, there is an offset -> same number of onsets & offsets
        assert len(onsets) == len(offsets)
        # all onsets are earlier than their corresponding offset
        assert (onsets.index < offsets.index).all()
        # there is no offset that is earlier than the next onset
        assert not (onsets.index.to_series().shift(-1)
                < offsets.index).any()
        # there is no onset, for which the difference to its offset is larger
        # than its difference to the next onset
        assert not (offsets.index-onsets.index <
                onsets.index.to_series().diff(-1)).any()

        print('all good')
    except AssertionError:
        print('Data remains exceptional !')

    blinks = pd.DataFrame([onsets.values, onsets.index, offsets.index,
                           offsets.values],
                          index=['onset', 'last_valid', 'first_valid',
                                 'offset']).T

    blinks['length'] = blinks.first_valid - blinks.last_valid

    blinks = blinks[blinks.length.le(
        max_length) & blinks.length.ge(min_length)]

    return blinks
