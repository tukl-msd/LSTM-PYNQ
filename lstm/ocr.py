#   Copyright (c) 2018, TU Kaiserslautern
#   Copyright (c) 2018, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import cv2 as cv2

from lstm import PynqLSTM, RUNTIME_HW, LSTM_DATA_DIR

NETWORK_FRAKTUR_OCR = "lstm-fraktur-ocr-pynq"
FRAKTUR_DATA_DIR = os.path.join(LSTM_DATA_DIR, 'fraktur')
FRAKTUR_ALPHABET = os.path.join(FRAKTUR_DATA_DIR, 'alphabet.txt')

MAX_OCR_LENGTH = 1024
PADDING = 16
MIN_CLIP = -1.75
MAX_CLIP = 3.75


class PynqOCR(PynqLSTM):
    __metaclass__ = ABCMeta

    def __init__(self, runtime, network, load_overlay, alphabet_path):
        super(PynqOCR, self).__init__(runtime, network, load_overlay)
        self.alphabet_path = alphabet_path

    @property
    def ops_per_seq_element(self):
        return self.lstm_ops_per_seq_element + self.fc_ops_per_seq_element

    @property
    def fc_ops_per_seq_element(self):
        return 2 * self.hidden_size * 2 * self.alphabet_size if self.peepholes_enabled else 2 * self.hidden_size * self.alphabet_size

    def inference(self, input_data):
        input_data = self.preprocess(input_data)
        input_data_post_process_width = int(len(input_data) / self.input_size)
        input_data_f = self._ffi.cast("float *", input_data.ctypes.data)
        keepalive = []
        out_buffer = self._ffi.new("char[]", MAX_OCR_LENGTH)
        ms_compute_time = self._ffi.new("float *")
        keepalive.append(out_buffer)
        self.interface.lstm_ocr_wrapper(input_data_f, len(input_data), out_buffer, bytes(self.alphabet_path, encoding='ascii'), ms_compute_time)
        mops_per_s = 0.001 * self.ops_per_seq_element * input_data_post_process_width / ms_compute_time[0]
        return mops_per_s, ms_compute_time[0], self._ffi.string(out_buffer).decode('utf8')

    @abstractproperty
    def alphabet_size(self):
        pass


class PynqFrakturOCR(PynqOCR):

    def __init__(self, runtime=RUNTIME_HW, load_overlay=True):
        super(PynqFrakturOCR, self).__init__(runtime, NETWORK_FRAKTUR_OCR, load_overlay, FRAKTUR_ALPHABET)
        self.fraktur_mean = np.loadtxt(os.path.join(FRAKTUR_DATA_DIR, 'mean.txt'))
        self.fraktur_std_deviation = np.loadtxt(os.path.join(FRAKTUR_DATA_DIR, 'std_deviation.txt'))

    @property
    def ffi_interface(self):
        return """
            void lstm_ocr_wrapper(float* input_data, int flat_length, char* out_buffer, char* alphabet_path, float* ms_compute_time);
            """

    @property
    def alphabet_size(self):
        return 110

    @property
    def input_size(self):
        return 25

    @property
    def hidden_size(self):
        return 100

    @property
    def peepholes_enabled(self):
        return True

    @property
    def bias_enabled(self):
        return True

    @property
    def bidirectional_enabled(self):
        return True

    def preprocess(self, input_data):
        input_data = input_data * 1.0 / np.amax(input_data)
        input_data = np.amax(input_data) - input_data
        input_data = input_data.T   
        w = input_data.shape[1]
        input_data = np.vstack([np.zeros((PADDING, w)),input_data,np.zeros((PADDING, w))])
        input_data = (input_data - self.fraktur_mean) / self.fraktur_std_deviation
        input_data = input_data.reshape(-1,1)
        input_data = np.round(input_data * 4)/4
        input_data = np.clip(input_data, a_min=float(MIN_CLIP), a_max=float(MAX_CLIP))
        input_data = input_data.astype(np.float32)    
        return input_data

