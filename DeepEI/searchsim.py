# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 08:03:27 2019

@author: hcji
"""

import re
import json
import time
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

def searchsim(smi, thres=80, maxcid=9999, maxsmi=100, timeout=999):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/smiles/"
    url += smi
    url += "/XML?Threshold="
    url += str(thres)
    url += "&MaxRecords="
    url += str(maxcid)
    req = requests.get(url, timeout=timeout)
    res = str(BeautifulSoup(req.text, "html.parser"))
    res = res.split('\n')
    for txt in res:
        if "<listkey>" in txt:
            key = re.findall(r"\d+\.?\d*", txt)[0]
    url2 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/"
    cids = []
    start_time = time.time()
    for i in range(100):
        url3 = url2 + key
        url3 += "/cids/XML"
        try:
            req = requests.get(url3, timeout=timeout)
        except:
            continue
        res = str(BeautifulSoup(req.content, "html.parser"))
        res = res.split('\n')
        if '</waiting>' in res:
            time.sleep(5)
            end_time = time.time()
            if end_time - start_time > 300:
                break
            continue
        else:
            for txt in res:
                if "<cid>" in txt:
                    cid = re.findall(r"\d+\.?\d*", txt)[0]
                    cids.append(cid)
            break
    return cids


def get_compounds(cids, timeout = 999):
    idstring = ''
    smiles = []
    inchikey = []
    all_cids = []
    for i, cid in enumerate(cids):
        idstring += ',' + str(cid)
        if ((i%100==99) or (i==len(cids)-1)):
            url_i = "http://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + idstring[1:(len(idstring))] + "/property/InChIKey,CanonicalSMILES/JSON"
            for l in range(100):
                try:
                    res_i = requests.get(url_i, timeout=timeout)
                    break
                except:
                    time.sleep(3)
                    continue
            soup_i = BeautifulSoup(res_i.content, "html.parser")
            str_i = str(soup_i)
            properties_i = json.loads(str_i)['PropertyTable']['Properties']
            idstring = ''
            for properties_ij in properties_i:
                smiles_ij = properties_ij['CanonicalSMILES']
                if '.' in smiles_ij:
                    continue
                if smiles_ij not in smiles:
                    smiles.append(smiles_ij)
                    inchikey.append(properties_ij['InChIKey'])
                    all_cids.append(str(properties_ij['CID']))
                else:
                    wh = np.where(np.array(smiles)==smiles_ij)[0][0]
                    all_cids[wh] = all_cids[wh] + ', ' + str(properties_ij['CID'])
    
    result = pd.DataFrame({'InChIKey': inchikey, 'SMILES': smiles, 'PubChem': all_cids})
    return result    


def get_simcomps(smi, thres=80, maxcid=9999, maxsmi=100, timeout=999):
    cids = searchsim(smi, thres, maxcid, maxsmi, timeout)
    output = get_compounds(cids, timeout)
    return output
    