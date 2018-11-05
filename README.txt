In addition to the libraries that are included with the standard installation of Anaconda, I found it was necessary to install these additional libraries (installed using conda install):

tensorflow-gpu==1.10.0
keras==2.2.2

----------------------------TA-Lib------------------------------------

In order to preprocess the Technical Indicator features, the python wrapper for TA-Lib was used.

A standard install of this is a little more difficult, and for me it was required to incorporate this library by downloading the file from this repository of python libraries: 

https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

I have included this library file in the Preprocessing sub-folder:
"TA_Lib-0.4.17-cp36-cp36m-win_amd64.whl"

