# NGPKS
Copyright (C) 2022 Haitao Zou(zht@glut.edu.cn)

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see http://www.gnu.org/licenses/.

Haitao Zou(zht@glut.edu.cn) Guilin University of Technology,Guilin, China


NGPKS
NGPKS:  Non-linear Gaussian profile kernel Similarity and Convolutional Networks for miRNA-disease association prediction. 

Requirements
Pytorch (tested on version 1.10.1+cu113)
numpy (tested on version 1.22.3)
sklearn (tested on version 0.20.3)

Quick start
To reproduce our results:
Unzip NGPKS.zip in ./NGPKS.
Run train.py to RUN NGPKS.

Data description
d-d.csv:disease-disease similarity matrix.
m-m.csv: miRNA-miRNA similarity matrix.
disease name.csv: list of disease names.
miRNA name.csv: list of miRNA names
m-d.csv: miRNA-disease association matrix



