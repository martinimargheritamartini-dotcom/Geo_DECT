"""


:author: Margherita Martini
:contact: 
:email: 
:organization: Laboratoire de Géologie de Lyon - Terre, Planètes, Environnement (LGL-TPE)
:address: 
:copyright: MIT License
:date: Jul. 13 2026 13:46
:dragonflyVersion: 2025.1.0.2063
:UUID: 7e90f2c77eb011f18afab07b2518f399
"""

__version__ = '1.0.0'

from ORSServiceClass.OrsPlugin.orsPlugin import OrsPlugin
from ORSServiceClass.OrsPlugin.uidescriptor import UIDescriptor


class Geo_DECT_7e90f2c77eb011f18afab07b2518f399(OrsPlugin):

    # Plugin definition
    multiple = False
    savable = False
    keepAlive = True
    canBeGenericallyOpened = True

    # UIs
    UIDescriptors = [UIDescriptor(name='MainFormGeo_dect',
                                  title='Geo_DECT',
                                  dock='Left',
                                  tab='Main',
                                  modal=False,
                                  collapsible=True,
                                  floatable=True)]

    def __init__(self, varname=None):
        super().__init__(varname)

    @classmethod
    def getMainFormName(cls):
        return 'MainFormGeo_dect'

    @classmethod
    def getMainFormClass(cls):
        from .mainformgeo_dect import MainFormGeo_dect
        return MainFormGeo_dect

