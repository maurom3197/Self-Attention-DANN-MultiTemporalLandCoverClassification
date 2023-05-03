import sys
import argparse
import breizhcrops
import torch
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

import pandas as pd
import os
import sklearn.metrics


def get_datasets(datapath, mode, batchsize, preload_ram=False, level="L2A"):
    print(f"Setting up datasets in {os.path.abspath(datapath)}, level {level}")
    datapath = os.path.abspath(datapath)

    frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath,
                                    preload_ram=preload_ram, level=level)
    frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath,
                                    preload_ram=preload_ram, level=level)
    frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath,
                                    preload_ram=preload_ram, level=level)
    if not "validation_only" in mode:
            frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath,
                                            preload_ram=preload_ram, level=level)

    if mode == "evaluation" or mode == "evaluation1":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02, frh03])
        testdataset = frh04
    elif mode == "validation_only":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02])
        validationdataset = frh03
    elif mode == "validation_test":
        traindatasets = torch.utils.data.ConcatDataset([frh01, frh02])
        validationdataset = frh03
        testdataset = frh04
        
    elif mode == 'all_zones':
        traindatasets = frh01
        testdataset1 = frh02
        testdataset2 = frh03        
        testdataset3 = frh04
        
    else:
        raise ValueError("only --mode 'validation' or 'evaluation' allowed")
    meta = dict(
        ndims=13 if level=="L1C" else 10,
        num_classes=len(frh01.classes),
        sequencelength=45
    )

    return traindatasets, testdataset1, testdataset2, testdataset3, meta   

def get_dataloader(traindatasets,testdataset, validationdataset=None, batchsize=32, workers=0):
    traindataloader = DataLoader(traindatasets, batch_size=batchsize, shuffle=True, num_workers=workers)
    validationdataloader = DataLoader(validationdataset, batch_size=batchsize, shuffle=True, num_workers=workers)
    testdataloader = DataLoader(testdataset, batch_size=batchsize, shuffle=False, num_workers=workers)

    return traindataloader, validationdataloader, testdataloader

def get_dataloader2(traindatasets,testdataset1, testdataset2, testdataset3, batchsize=32, workers=0):
    traindataloader = DataLoader(traindatasets, batch_size=batchsize, shuffle=True, num_workers=workers)
    testdataloader1 = DataLoader(testdataset1, batch_size=batchsize, shuffle=True, num_workers=workers)
    testdataloader2 = DataLoader(testdataset2, batch_size=batchsize, shuffle=True, num_workers=workers)
    testdataloader3 = DataLoader(testdataset3, batch_size=batchsize, shuffle=True, num_workers=workers)

    return traindataloader, testdataloader1, testdataloader2, testdataloader3

def reduce_random_dataset(full_dataset, new_size = 10000):

    residual = len(full_dataset) - new_size
    reduced_dataset, _ = torch.utils.data.random_split(full_dataset, [new_size,residual], generator=torch.Generator().manual_seed(42))
    
    return reduced_dataset