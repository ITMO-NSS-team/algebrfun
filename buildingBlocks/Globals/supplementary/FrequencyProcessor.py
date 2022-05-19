import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from buildingBlocks.Globals.GlobalEntities import set_constants, get_full_constant
from functools import reduce
from itertools import product

class FrequencyProcessor4TimeSeries:
    """
    Works with the spectrum of the studied series.
    Identifies the most significant harmonics to determine the frequencies of future tokens.
    """

    @staticmethod
    def fft(grid, x, wmin=0, wmax=None, c=10):
        # x = x.reshape()
        constants = get_full_constant()
        shp = constants['shape_grid']
        w = []
        mask = []
        for i in range(len(grid)):
            step = np.mean(grid[i][1:] - grid[i][:-1])
            w_c = np.fft.fftfreq(len(grid[i]), step)
            w_c = w_c.reshape(shp)
            w.append(w_c)
            c_max = w_c.max()
            if wmax == None or c_max > wmax:
                wmax = c_max
            current_mask = (w_c >= wmin) & (w_c <= c_max)
            if len(mask) == 0:
                mask = current_mask
            else:
                mask *= current_mask
        print("msk", mask.shape, mask)
        x = x.reshape((constants['shape_grid']))
        y = np.fft.fftn(x, s=shp)
        y = np.abs(y)
        y[~mask] = y.min()
        for wi, we in enumerate(w):
            w[wi][~mask] = wmin
        return w, y, wmax

    @staticmethod
    def findextrema_prev(w, spec):
        # print(spec.shape)
        # spec = np.abs(spec)
        # print(spec.shape)
        extremums_idxs = argrelextrema(spec, np.greater, mode='wrap')[0]
        kw = w[extremums_idxs]
        kspec = spec[extremums_idxs]
        # plt.plot(kw, kspec)
        return kw, kspec
    
    @staticmethod
    def findextrema(w, spec, n):
        # spec = np.abs(spec)
        sorted_spec = sorted(list(zip(np.ndindex(spec.shape), spec.reshape(-1))), key=lambda zp: zp[1], reverse=True)
        n_indexs = np.array([list(sorted_spec_elem[0]) for sorted_spec_elem in sorted_spec[:n]])
        n_indexs = [n_indexs[:, it] for it in range(len(spec.shape))]
        kw = []
        for wi, we in enumerate(w):
            kw.append(w[wi][n_indexs])
        kspec = spec[n_indexs]
        
        return kw, kspec 
        


    @staticmethod
    def sort_by_specter(w, spec):
        idxs_sorted = np.argsort(spec)[::-1]
        # sortspec = np.array([spec[i] for i in idxs])
        # sortw = np.array([w[i] for i in idxs])
        # print("spec + idxs", spec.shape, idxs_sorted)
        spec_sorted = spec[idxs_sorted]
        w_sorted = []
        for wi, we in enumerate(w):
            w_sorted.append(w[wi][idxs_sorted])
        return w_sorted, spec_sorted

    @staticmethod
    def choice_freqs(w, spec, pow=1, number_selected=1, number_selecting=None):
        if number_selecting is None:
            number_selecting = len(w)
        # print("number selection and len w", number_selected, len(w))
        # print("content in w", w)    
        # assert number_selecting <= len(w), 'selecting number more than all freqs number'
        # assert 0 <= number_selected <= number_selecting, 'selected freqs number more than' \
                                                        #  ' selecting number or less than zero'
        if number_selected == 0:
            return None
        # spec = spec[:number_selecting]
        # w = w[:number_selecting]
        spec_sum = (spec ** pow).sum()
        # probabilities = list(map(lambda x: x ** pow / spec_sum, spec))
        probabilities = (spec**pow)/spec_sum
        idxs = np.arange(len(w[0]))
        choice_i = np.random.choice(idxs, size=number_selected, replace=False, p=probabilities)
        choice = np.array(w)[:, choice_i]

        # plt.figure('selected freqs: {}'.format(choice))
        # idxs = np.argsort(w)
        # plt.plot(w[idxs], spec[idxs])

        return choice.T

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
        # print("shapes for start", grid.shape, x.shape)
        w, s, wmax = FrequencyProcessor4TimeSeries.fft(grid, x, wmin, wmax, c)
        # print("amplitudes", w)
        # print("we found fft", w.shape, s.shape)
        kw, ks = FrequencyProcessor4TimeSeries.findextrema(w, s, number_selecting)
        # print("we found extrema", kw)
        # kw, ks = FrequencyProcessor4TimeSeries.findextrema(kw, ks)
        # print("we found extrema", ks.shape)
        kw, ks = FrequencyProcessor4TimeSeries.sort_by_specter(kw, ks)
        out_freqs = FrequencyProcessor4TimeSeries.choice_freqs(kw, ks,
                                                            # pow=number_selecting ** 0.5,
                                                            pow=2,
                                                            number_selected=number_selected,
                                                            number_selecting=number_selecting)
        # print("also aort", out_freqs)
        return out_freqs, wmax

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
        # Wmax = np.fft.fftfreq(len(grid), np.mean(grid[1:]-grid[:-1])).max()
        # if wmax == None:
        #     wmax = Wmax
        choice_freqs, Wmax = FrequencyProcessor4TimeSeries.find_freq_for_summand(grid, x, wmin,
                                                                           wmax, c=c,
                                                                           number_selecting=number_selecting,
                                                                           number_selected=number_selected)
        if choice_freqs is None:
            return None
        ending_freqs = []
        # print("choice freqs shape", choice_freqs)
        # print("fkg", threshold*Wmax, token_type)
        for choice_freq in choice_freqs:
            # print(choice_freq)
            if len(choice_freq[choice_freq < threshold*Wmax]) == len(choice_freq):
                if token_type == 'seasonal':
                    continue
                ending_freqs.append(choice_freq)
                break
            else:
                if token_type == 'trend':
                    continue
            ending_freqs.append(choice_freq)
            break
        # print("ending freqs shape", ending_freqs)
        return ending_freqs