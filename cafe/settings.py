import numpy as np
from scipy.signal import argrelextrema
from functools import reduce

class SingletonClass(type):
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


class FrequencyProcessor4TimeSeries:
    """
    Works with the spectrum of the studied series.
    Identifies the most significant harmonics to determine the frequencies of future tokens.
    """

    @staticmethod
    def fft(grid, x, shp, wmin=0, wmax=None, c=10):
        # x = x.reshape()
        w = []
        mask = []
        wmax = []
        freq_steps = [] 
        for i in range(len(grid)):
            current_grid = grid[i].reshape(shp)
            if len(np.unique(current_grid[0])) == 1:
                current_grid = current_grid.T
            current_grid = current_grid[0]
            step = np.mean(current_grid[1:] - current_grid[:-1])
            # w_c = np.fft.fftfreq(len(grid[i]), step)
            w_c = np.fft.fftfreq(len(current_grid), step)
            w_c = FrequencyProcessor4TimeSeries.get_actual_freq(grid[i], current_grid, w_c)
            freq_steps.append(np.mean(w_c[1:] - w_c[:-1]))
            w_c = w_c.reshape(shp)
            w.append(w_c)
            c_max = w_c.max()
            wmax.append(c_max)
            current_mask = (w_c >= wmin) & (w_c <= c_max)
            # mask.append(current_mask)
            if len(mask) == 0:
                mask = current_mask
            else:
                mask *= current_mask
        # w = np.array(list(product(*w)))
        # w = np.array([cur_w.reshape(shp) for cur_w in w.T])
        # mask = np.array([np.prod(tpl) for tpl in product(*mask)])
        # mask = mask.reshape(shp)
        x = x.reshape(shp)
        # print("our x", x)
        y = np.fft.fftn(x, s=shp)
        y = np.abs(y)
        y[~mask] = y.min()
        for wi, we in enumerate(w):
            w[wi][~mask] = wmin
        # print("resulting fft procedure", y)
        return w, y, wmax, freq_steps

    @staticmethod
    def get_actual_freq(gen_grid, current_coord_grid, freqs):
        coord_freq = dict(np.array([current_coord_grid, freqs]).T)
        end_freqs = []
        for coord_value in gen_grid:
            end_freqs.append(coord_freq.get(coord_value))
        
        return np.array(end_freqs)

    @staticmethod
    def findextrema_prev(w, spec):
        # spec = np.abs(spec)
        extremums_idxs = argrelextrema(spec, np.greater, mode='wrap')[0]
        kw = w[extremums_idxs]
        kspec = spec[extremums_idxs]
        # plt.plot(kw, kspec)
        return kw, kspec
    

    @staticmethod
    def findextrema(w, spec, n):
        spec_line = np.array(spec).reshape(-1)
        extremums_idxs = argrelextrema(spec_line, np.greater, mode='wrap')[:n]
        # print("why zero", extremums_idxs)
        kw = []
        for w_iter in w:
            w_line = np.array(w_iter).reshape(-1)
            kw.append(w_line[extremums_idxs])
        kspec = spec_line[extremums_idxs]

        return kw, kspec

    @staticmethod
    def find_intersection(lst_of_ampl):
        ss = [x[0][0] for x in lst_of_ampl]
        general_w = list(reduce(lambda a, b: set(a) & set(b), ss))
        current_elem = list(filter(lambda x: x[2] == 1, lst_of_ampl))[0]
        s = []
        for i, elem in enumerate(current_elem[0][0]):
            if elem in general_w:
                s.append(current_elem[1][i])
        # w = list(filter(lambda x: x!= None, map(lambda x: x[0] if x[1] in general_s else None, current_elem)))

        return np.array([general_w]), np.array(s)

    @staticmethod
    def sort_by_specter(w, spec):
        idxs_sorted = np.argsort(spec)[::-1]
        # sortspec = np.array([spec[i] for i in idxs])
        # sortw = np.array([w[i] for i in idxs])
        spec_sorted = spec[idxs_sorted]
        w_sorted = []
        for wi, we in enumerate(w):
            w_sorted.append(w[wi][idxs_sorted])
        return w_sorted, spec_sorted

    @staticmethod
    def choice_freqs(w, spec, pow=1, number_selected=1, number_selecting=None):
        if number_selecting is None:
            number_selecting = len(w)
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
        # print(idxs.shape, number_selected)
        if idxs.shape[0] < number_selected:
            choice = np.array(w)[:, idxs]
            return choice.T

        choice_i = np.random.choice(idxs, size=number_selected, replace=False, p=probabilities)
        choice = np.array(w)[:, choice_i]

        # plt.figure('selected freqs: {}'.format(choice))
        # idxs = np.argsort(w)
        # plt.plot(w[idxs], spec[idxs])

        return choice.T


    @staticmethod
    def find_freq_for_summand(grid, x, shp, wmin=0, wmax=None, c=10, number_selecting=1, number_selected=1):
        # print("check freqsss")
        w, s, wmax, freq_steps = FrequencyProcessor4TimeSeries.fft(grid, x, shp, wmin, wmax, c)
        # print("after fft", w, "apec", s)
        # kw, ks = FrequencyProcessor4TimeSeries.findextrema(w, s, number_selecting)
        kw, ks = FrequencyProcessor4TimeSeries.findextrema(w, s, number_selecting)
        # kw, ks = FrequencyProcessor4TimeSeries.findextrema(kw, ks)
        kw, ks = FrequencyProcessor4TimeSeries.sort_by_specter(kw, ks)
        # print("after sorting", kw, "apec", ks)
        out_freqs = FrequencyProcessor4TimeSeries.choice_freqs(kw, ks,
                                                            # pow=number_selecting ** 0.5,
                                                            pow=2,
                                                            number_selected=number_selected,
                                                            number_selecting=number_selecting)
        # print("after choose", out_freqs)
        if len(out_freqs) == 0:
            return (None, None)
        return out_freqs, wmax, freq_steps


    @staticmethod
    def choice_freq_for_summand(grid, x, shp, wmin=0, wmax=None, c=10,
                                number_selecting=1, number_selected=1, token_type='seasonal', threshold=0.001):
        # Wmax = np.fft.fftfreq(len(grid), np.mean(grid[1:]-grid[:-1])).max()
        # if wmax == None:
        #     wmax = Wmax
        choice_freqs,  Wmax, freq_steps = FrequencyProcessor4TimeSeries.find_freq_for_summand(grid, x, shp, wmin,
                                                                           wmax, c=c,
                                                                           number_selecting=number_selecting,
                                                                           number_selected=number_selected)

        # print(choice_freqs)
        # print("freqsss", choice_freqs, Wmax)
        if choice_freqs is None:
            return None
        ending_freqs = []
        try:
            # print(len(Wmax))
            Wmax = np.array(Wmax)
        except:
            Wmax = np.array([Wmax]) 
        for choice_freq in choice_freqs:
            # if len(choice_freq[choice_freq < threshold*Wmax]) == len(choice_freq):
            if np.all(choice_freq < threshold*Wmax):
                if token_type == 'seasonal':
                    continue
                ending_freqs.append(choice_freq)
                break
            else:
                if token_type == 'trend':
                    continue
                ending_freqs.append(choice_freq)
                break
        if len(ending_freqs) == 0:
            return None
        return ending_freqs, freq_steps
