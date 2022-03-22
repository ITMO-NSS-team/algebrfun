import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


class FrequencyProcessor4TimeSeries:
    """
    Works with the spectrum of the studied series.
    Identifies the most significant harmonics to determine the frequencies of future tokens.
    """

    @staticmethod
    def fft(grid, x, wmin=0, wmax=None, c=10):
        step = np.mean(grid[1:] - grid[:-1])
        w = np.fft.fftfreq(c * len(grid), step)
        y = np.fft.fft(x, n=c * len(grid))
        if wmax is None:
            wmax = w.max()
        y = y[(w >= wmin) & (w <= wmax)]
        w = w[(w >= wmin) & (w <= wmax)]
        # not circle
        return w, y

    @staticmethod
    def findextrema(w, spec):
        spec = np.abs(spec)
        extremums_idxs = argrelextrema(spec, np.greater, mode='wrap')[0]
        kw = w[extremums_idxs]
        kspec = spec[extremums_idxs]
        # plt.plot(kw, kspec)
        return kw, kspec

    @staticmethod
    def sort_by_specter(w, spec):
        idxs_sorted = np.argsort(spec)[::-1]
        # sortspec = np.array([spec[i] for i in idxs])
        # sortw = np.array([w[i] for i in idxs])
        spec_sorted = spec[idxs_sorted]
        w_sorted = w[idxs_sorted]
        return w_sorted, spec_sorted

    @staticmethod
    def choice_freqs(w, spec, pow=1, number_selected=1, number_selecting=None):
        if number_selecting is None:
            number_selecting = len(w)
        assert number_selecting <= len(w), 'selecting number more than all freqs number'
        assert 0 <= number_selected <= number_selecting, 'selected freqs number more than' \
                                                         ' selecting number or less than zero'
        if number_selected == 0:
            return None
        spec = spec[:number_selecting]
        w = w[:number_selecting]
        spec_sum = (spec ** pow).sum()
        # probabilities = list(map(lambda x: x ** pow / spec_sum, spec))
        probabilities = (spec**pow)/spec_sum
        choice = np.random.choice(w, size=number_selected, replace=False, p=probabilities)

        # plt.figure('selected freqs: {}'.format(choice))
        # idxs = np.argsort(w)
        # plt.plot(w[idxs], spec[idxs])

        return choice

    @staticmethod
    def find_dif_in_freqs(w, known_freqs):
        if len(w) == 0:
            return []
        res = []
        if len(w) == 1:
            w1 = w[0]
            for w0 in known_freqs:
                if 2 * abs(w1 - w0)/abs(w1 + w0) < 0.05:
                    res.append((w0, 0, 2*w0, w0))
            return res
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                for w0 in known_freqs:
                    w1 = min(w[i], w[j])
                    w2 = max(w[i], w[j])
                    dw = w2 - w0
                    w1_expected = abs(w0 - dw)
                    if 2 * abs(w1 - w1_expected)/abs(w1 + w1_expected) < 0.05:
                        res.append((w0, w1, w2, dw))
        return res

    @staticmethod
    def find_freq_for_summand(grid, x, wmin=0, wmax=None, c=10, number_selecting=1, number_selected=1):
        w, s = FrequencyProcessor4TimeSeries.fft(grid, x, wmin, wmax, c)
        kw, ks = FrequencyProcessor4TimeSeries.findextrema(w, s)
        kw, ks = FrequencyProcessor4TimeSeries.findextrema(kw, ks)
        kw, ks = FrequencyProcessor4TimeSeries.sort_by_specter(kw, ks)
        out_freqs = FrequencyProcessor4TimeSeries.choice_freqs(kw, ks,
                                                               # pow=number_selecting ** 0.5,
                                                               pow=2,
                                                               number_selected=number_selected,
                                                               number_selecting=number_selecting)
        return out_freqs

    @staticmethod
    def find_freq_for_multiplier(grid, x, w0, wmin=0, wmax=None, c=10, max_len=1):
        w, s = FrequencyProcessor4TimeSeries.fft(grid, x, wmin, wmax, c)
        kw, ks = FrequencyProcessor4TimeSeries.findextrema(w, s)
        kw, ks = FrequencyProcessor4TimeSeries.findextrema(kw, ks)
        kw, ks = FrequencyProcessor4TimeSeries.sort_by_specter(kw, ks)
        return FrequencyProcessor4TimeSeries.find_dif_in_freqs(kw, w0) # todo refactor

    @staticmethod
    def choice_freq_for_summand(grid, x, wmin=0, wmax=None, c=10,
                                number_selecting=1, number_selected=1, token_type='seasonal', threshold=0.001):
        Wmax = np.fft.fftfreq(len(grid), np.mean(grid[1:]-grid[:-1])).max()
        if wmax == None:
            wmax = Wmax
        choice_freqs = FrequencyProcessor4TimeSeries.find_freq_for_summand(grid, x, wmin,
                                                                           wmax, c=c,
                                                                           number_selecting=number_selecting,
                                                                           number_selected=number_selected)
        if choice_freqs is None:
            return None
        for choice_freq in choice_freqs:
            if choice_freq < threshold*Wmax:
                if token_type == 'seasonal':
                    continue
                return choice_freq
            else:
                if token_type == 'trend':
                    continue
            return choice_freq