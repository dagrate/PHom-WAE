# PHom-WAE: Persistent Homology for Wasserstein Auto-Encoders

Persistent Homology for Wasserstein Auto-Encoders

<p align="middle">
  <img src="https://github.com/dagrate/PHom-WAE/blob/master/images/originSamples_barcode.png" width="100"/>
  <img src="https://github.com/dagrate/PHom-WAE/blob/master/images/WAE.png" width="100"/>
  <img src="https://github.com/dagrate/PHom-WAE/blob/master/images/VAE.png" width="100"/>
</p>

PHom-WAE is a Python and R library that proposes to evaluate the quality of the encoding-decoding process of Wasserstein auto-encoders in comparison to Variational auto-encoders. For some real-world applications, different than computer vision, we cannot assess visually the quality of the encoding-decoding. Therefore, we have to use other metrics. Here, we rely on persistent homology because it is capable to acknowledge the shape of the data points, by opposition to traditional distance measures such the Euclidean distance.

The auto-encoders are trained with Python to produce reconstructed samples saved in csv files.

The persistent homology features and the bottleneck distance are evaluated with the TDA package of R. 


----------------------------

## Dependencies

The library uses **Python 3** and **R** with the following modules:
- numpy (Python 3)
- scipy (Python 3)
- matplotlib (Python 3)
- pandas (Python 3)
- pylab (Python 3)
- sklearn (Python 3)
- keras (Python 3)
- functools (Python 3)
- TDA (R)
- TDAmapper (R) -> only if you want to play with the mapper algorithm

It is advised to install BLAS/LAPACK to increase the efficiency of the computations:  
sudo apt-get install libblas-dev liblapack-dev gfortran

----------------------------

## Citing

If you use the repository, please cite: to be updated
