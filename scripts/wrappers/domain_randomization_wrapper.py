#!/usr/bin/env python3
"""
domain_randomization_wrapper.py  --  Domain Randomisation
Per-episode randomisation of CNN noise, cloud bias, and slew cost
for fault-tolerant policies (Nagano & Schaub 2025).
"""
import logging, numpy as np, gymnasium as gym
logger = logging.getLogger(__name__)


class DomainRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, noise_range=(0.02, 0.15),
                 bias_range=(-0.05, 0.05), slew_range=(0.7, 1.3),
                 enabled=True, seed=None):
        super().__init__(env)
        self._rng   = np.random.default_rng(seed)
        self._noise = noise_range
        self._bias  = bias_range
        self._slew  = slew_range
        self._on    = enabled

    def reset(self, **kwargs):
        if self._on:
            noise = float(self._rng.uniform(*self._noise))
            bias  = float(self._rng.uniform(*self._bias))
            slew  = float(self._rng.uniform(*self._slew))
            self._set_cnn(noise, bias)
            self._set_slew(slew)
            logger.debug(f"DR: noise={noise:.3f} bias={bias:+.3f} slew={slew:.2f}")
        return self.env.reset(**kwargs)

    def _find_cm(self):
        obj = self
        while hasattr(obj, "env"):
            for a in ("_cloud_model", "cloud_model"):
                cm = getattr(obj, a, None)
                if cm: return cm
            obj = obj.env
        base = getattr(obj, "unwrapped", obj)
        for a in ("_cloud_model", "cloud_model"):
            cm = getattr(base, a, None)
            if cm: return cm
        return None

    def _set_cnn(self, noise, bias):
        cm = self._find_cm()
        if not cm: return
        for a in ("_noise_std", "noise_std"):
            if hasattr(cm, a): setattr(cm, a, noise); break
        for a in ("_bias", "bias"):
            if hasattr(cm, a): setattr(cm, a, bias); break

    def _set_slew(self, mult):
        try:
            obj = self
            while hasattr(obj, "env"): obj = obj.env
            getattr(obj, "unwrapped", obj).satellites[0]._slew_energy_multiplier = mult
        except Exception: pass