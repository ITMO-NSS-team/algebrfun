"""Классы, представляющие интерфейс для синтезирвоания"""

from functools import reduce

import buildingBlocks.default.Tokens as Tokens
import buildingBlocks.Globals.GlobalEntities as Ge
import buildingBlocks.Synthesis.Chain as Chain

import numpy as np


class Synthesizer:
    def __init__(self, individ, grid, coder, markov_chain, instructions_high: dict = None, split_imps: dict = None,
                 residuals: np.ndarray = None):
        self.individ = individ
        self.grid = grid
        self.sinthesizer_complex = SynthesizerComplex(individ, coder, markov_chain)
        self.sinthesizer_simple = SynthesizerSimple(individ, instructions_high)
        if split_imps is None:
            split_imps = {}

        self.split_imps = {
            'make': True,
            'min_ampl': 0.05,
            'max_freq': float('inf')
        }
        self.split_imps.update(split_imps)
        self._split_imps()
        self.residuals = residuals

    def _split_imps(self):
        for idx, token in enumerate(self.individ.structure):
            if self.split_imps['make'] and isinstance(token, Tokens.Imp) and \
               abs(token.param('Amplitude')) >= self.split_imps['min_ampl'] and \
               abs(token.param('Frequency')) <= self.split_imps['max_freq']:
                self.individ.structure[idx] = Tokens.ImpComplex(pattern=self.individ.structure[idx])
                self.individ.structure[idx].init_structure_from_pattern(self.grid)
                self.individ.structure[idx].fitness = self.individ.structure[idx].pattern.fitness

    def _add_residuals(self, grid):
        if self.residuals is None:
            return 0
        return np.random.choice(self.residuals, size=grid.shape, replace=True)

    def fit(self):
        self.sinthesizer_simple.fit()
        self.sinthesizer_complex.fit()

    def predict(self, grid):
        return self.sinthesizer_simple.generate(grid) + self.sinthesizer_complex.generate(grid)\
               + self._add_residuals(grid)


class SynthesizerComplex:
    def __init__(self, individ, coder, markov_chain):
        self.individ = individ
        self.coder = coder
        self.markov_chain = markov_chain

    def fit(self):
        if self.coder.complex_imps:
            labels = self.coder.encode()
            self.markov_chain.fit(labels)

    def create_pulses(self, grid, super_state=None):
        n_samples = 50
        new_labels, super_state = self.markov_chain.generate(super_state=super_state, n_samples=n_samples)
        last_pulse_start = 0
        grid_max = grid.max()
        new_imps = []
        while last_pulse_start < grid_max:
            self.coder.decode(labels=new_labels, grid=grid, init_pulse_start=last_pulse_start, generated_imps=new_imps)
            # new_imps.extend(generated_imps)
            last_pulse_start = new_imps[-1].param(name='Pulse start')
            # if len(generated_imps) < n_samples:
            #     break
            new_labels, super_state = self.markov_chain.generate(super_state=super_state, init_state=new_labels[-1],
                                                                 n_samples=n_samples)
            # print(last_pulse_start)
        return new_imps

    def generate(self, grid):
        if not self.coder.complex_imps:
            return np.zeros(grid.shape)
        new_imps = self.create_pulses(grid)
        return reduce(lambda x, y: x + y,
                      list(map(lambda token: token.value(grid), new_imps)))


class SynthesizerSimple:
    def __init__(self, individ, instructions_high=None):
        self.individ = individ
        self.simple_tokens = self._get_simple_tokens()

        if instructions_high is None:
            instructions_high = {
                'Sin': dict(Amplitude=lambda param, grid: param + 0*np.random.normal(0, 10*param, grid.shape),
                            Phase=lambda param, grid: param + 0*np.sin(0.1*grid)),
                # 'Power': dict(Amplitude=lambda param, grid: np.random.normal(param, 0.0*param, grid.shape))
            }
        self.instructions_high = instructions_high

    def _get_simple_tokens(self):
        simple_tokens = list(filter(lambda token: not isinstance(token, Tokens.ImpComplex) and token.mandatory == 0,
                                    self.individ.structure))
        return simple_tokens

    def _add_high_frequency_noise(self):
        for token in self.simple_tokens:
            if token.name_ in self.instructions_high.keys():
                try:
                    for key, value in self.instructions_high[token.name_].items():
                        key_ = token.get_key_use_params_description('name', key)
                        token.set_descriptor(key_, 'func', value)
                except:
                    raise
            token.fixator['val'] = False

    def fit(self):
        pass

    def generate(self, grid):
        self._add_high_frequency_noise()
        if not self.simple_tokens:
            return np.zeros(grid.shape)
        return reduce(lambda x, y: x+y,
                      list(map(lambda token: token.value(grid), self.simple_tokens)))
