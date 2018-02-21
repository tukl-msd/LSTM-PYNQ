# LSTM-PYNQ Pip Installable Package

This repo contains the pip install package for Quantized LSTM on PYNQ. 
Currently one overlay is included, that performs Optical Character Recognition (OCR) of [Fraktur](http://www.deutschestextarchiv.de/) text.

If you find it useful, we would appreciate a citation to:

**Hardware Architecture of Bidirectional Long Short-Term Memory Neural Network for Optical Character Recognition**,
V. Rybalkin, M. R. Yousefi, N. Wehn, and D. Stricker,
*In Proceedings of the Conference on Design, Automation & Test in Europe (DATE)*. European Design and Automation Association, 2017, pp. 1390–1395, [DOI](http://ieeexplore.ieee.org/document/7927210/)

BibTeX:

``` bibtex
@inproceedings{rybalkin2017hardware,
  title={Hardware architecture of bidirectional long short-term memory neural network for optical character recognition},
  author={Rybalkin, Vladimir and Wehn, Norbert and Yousefi, Mohammad Reza and Stricker, Didier},
  booktitle={Proceedings of the Conference on Design, Automation \& Test in Europe},
  pages={1394--1399},
  year={2017},
  organization={European Design and Automation Association}
}
```
This design became a winner of [Xilinx Open Hardware Design Contest 2016](http://www.openhw.eu/2016-finalists.html) in *PhD Embedded Category*.

This repo is a joint release of University of Kaiserslautern, [Microelectronic Systems Design Research Group](https://ems.eit.uni-kl.de/en/start/): Vladimir Rybalkin, Muhammad Mohsin Ghaffar, Norbert Wehn in cooperation with [Xilinx, Inc.:](https://www.xilinx.com/) Alessandro Pappalardo, Giulio Gambardella, Michael Gross, Michaela Blott.

## Quick Start

In order to install it to your PYNQ (on PYNQ v2.0), connect to the board, open a terminal and type:

```
sudo pip3.6 install git+https://github.com/tukl-msd/LSTM-PYNQ.git 
```

This will install the LSTM-PYNQ package to your board, and create a **lstm** directory in the Jupyter home area. You will find the Jupyter notebooks to test the LSTM in this directory. 
 
## Repo organization 

The repo is organized as follows:
-   *lstm*: contains the pip installed package.
    -	*lstm.py*: contains the PynqLSTM abstract class description.
    -   *ocr.py*: contains the PynqOCR abstract class and PynqFrakturOCR class descriptions.
    -	*bitstreams*: bitstream for the Fraktur OCR overlay.
    -	*libraries*: pre-compiled shared objects for low-level driver of the overlays.
    -	*datasets*: contains support files for working with a given dataset.
-	*notebooks*: lists a set of python notebooks examples, that during installation will be moved in `/home/xilinx/jupyter_notebooks/lstm/` folder.
