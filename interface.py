import numpy as np
import importlib
import sys
import os


class Indicators:


    @property
    def taxonomy(self):
        return np.array(list(Indicators._taxonomy.keys()))


    def __init__(self): 
        if not hasattr(Indicators, "_taxonomy"): self._load()


    def __getattr__(self, taxa):
        if taxa not in Indicators._taxonomy: raise AttributeError
        elif taxa in self.__dict__: return self.__dict__[taxa]
        statement = f"indicators.taxonomy.{taxa}.interface"
        module    = importlib.import_module(statement)
        interface = Indicators._taxonomy[taxa]
        if hasattr(module, interface): interface = getattr(module, interface)
        else: return "Not Implemented"
        instance  = interface()
        setattr(self, taxa, instance)
        return instance


    def __repr__(self):
        titleborder = " :::::::::::::::::::::::::::::::::::::: "
        title       = " :::: Adaptive Indicator Framework :::: "

        subtitle = "  Phylogenetic Taxonomy "
        border   = "  --------------------- "

        taxonomy = ""
        for i, taxa in enumerate(self.taxonomy):
            taxonomy += f"    {i+1}) {taxa}\n"
        
        note     = "  See >>> help(object) for details."

        header = f"\n{titleborder}\n{title}\n{titleborder}\n"
        body   = f"\n{subtitle}\n{border}\n{taxonomy}{border}"
        footer = f"\n{note}"

        return f"{header}{body}{footer}"


    def _load(self):
        Indicators._taxonomy = dict()
        fpath   = os.path.abspath(__file__)
        path    = os.sep.join(fpath.split(os.sep)[:-1])
        path    = os.path.join(path, "taxonomy")
        classes = os.listdir(path)
        for taxa in classes:
            Indicators._taxonomy[taxa] = taxa.title()
