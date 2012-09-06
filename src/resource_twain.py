#!/usr/bin/env python
"""Resource module for aquiring images using TWAIN (Windows)"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.997"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"


#
# DEPENDENCIES
#

from PIL import Image, ImageWin

import os, os.path, sys
import time
import types

#
# GLOBALS
#

_TRANSFER_METHOD = 'Natively'

tmpfilename = "tmp.bmp"
OverridXferFileName = 'c:/twainxfer.jpg'

#
# CLASSES
#

class Twain_Base:
    """Simple Base Class for twain functionality. This class should
    work with all the windows librarys, i.e. wxPython, pyGTK and Tk.
    """

    SM=None                        # Source Manager
    SD=None                        # Data Source
    ProductName='Scannomatic'      # Name of this product
    XferMethod = _TRANSFER_METHOD      # Transfer method currently in use
    AcquirePending = False         # Flag to indicate that there is an acquire pending
    mainWindow = None              # Window handle for the application window
    next_file_name = ""
    owner = None
    


    def Initialise(self):
        """Set up the variables used by this class"""
        (self.SD, self.SM) = (None, None)
        self.ProductName= None #'Scannomatic'
        self.XferMethod = _TRANSFER_METHOD
        self.AcquirePending = False
        self.mainWindow = None

    def Terminate(self):
        """Destroy the data source and source manager objects."""

        retSourceName = None

        if self.SD: 
            retSourceName = self.SD.GetSourceName()
            self.SD.destroy()
        print "***Releasing scanner", retSourceName
        if self.SM: 
            self.SM.destroy()

        self.SD = None
        self.SM = None

        return retSourceName

    def OpenScanner(self, mainWindow=None, ProductName=None, UseCallback=False):
        """Connect to the scanner"""
        if ProductName:
            self.ProductName = ProductName
        if mainWindow:
            self.mainWindow = mainWindow

        if not self.SM:
            if self.owner and self.owner._scanner_id:
                self.SM = twain.SourceManager(self.mainWindow, ProductName=self.owner._scanner_id)
            else:
                self.SM = twain.SourceManager(self.mainWindow)

        if not self.SM:
            return

        if self.SD == None:
            if self.owner and self.owner._scanner_id:
                self.SD = self.SM.OpenSource(self.owner._scanner_id)
            else:
                self.SD = self.SM.OpenSource()

        if self.SD:
            print "***", self.ProductName+' has been opened: ' + self.SD.GetSourceName()

        if UseCallback:
            self.SM.SetCallback(self.OnTwainEvent)

    #def getClipboard(handle=None, dtype=win32con.CF_DIB): 
     #NOT IN USE...
     #   wclip.OpenClipboard()
      #  if handle:
       #     d=wclip.GetClipboardDataHandle(handle)
        #else:
         #   d=wclip.GetClipboardData(dtype) 
     #   wclip.CloseClipboard() 
      #  return d

    def _Acquire(self, handle):
        """Begin the acquisition process. The actual acquisition will be notified by 
        either polling or a callback function."""
        if not self.SD:
            self.OpenScanner(handle, ProductName="Scannomatic", UseCallback=USE_CALLBACK) #0, ProductName=scanner)
        if not self.SD: return
        try:
            self.SD.SetCapability(twain.ICAP_YRESOLUTION, twain.TWTY_FIX32, 72.0) 
        except:
            pass

        marginX = 0.0
        marginY = 0.0
        sizeY = 11.45875 #11.69
        sizeX = 8.32768 #8.27
        resXY = 600.0 #72.0

        #V700 SETTINGS Set/Get        
        self.SD.SetCapability(twain.ICAP_LIGHTPATH,twain.TWTY_UINT16, twain.TWLP_TRANSMISSIVE)
        self.SD.SetCapability(twain.ICAP_LIGHTSOURCE, twain.TWTY_UINT16, 0)
        self.SD.SetCapability(twain.ICAP_UNITS, twain.TWTY_UINT16, twain.TWUN_INCHES)
        self.SD.SetCapability(twain.ICAP_PIXELTYPE, twain.TWTY_UINT16, twain.TWPT_GRAY)
        self.SD.SetCapability(32829, 4, 1)          #This sets so it scans the full area 
        #self.SD.SetCapability(32805, 6, 0)         #Unknown fucntion
        #self.SD.SetCapability(32793, 6, 1)         #Unknown function        
        #self.SD.SetCapability(twain.ICAP_ORIENTATION, twain.TWTY_UINT16, twain.TWOR_ROT90)
        self.SD.SetCapability(twain.ICAP_XRESOLUTION,twain.TWTY_FIX32, resXY)
        self.SD.SetCapability(twain.ICAP_YRESOLUTION,twain.TWTY_FIX32, resXY)
        
        #self.SD.SetCapability(twain.ICAP_CONTRAST,twain.TWTY_FIX32,25.0) #0-1000
        #self.SD.SetCapability(twain.ICAP_BRIGHTNESS,twain.TWTY_FIX32,125.0) #0-1000
        self.SD.SetImageLayout((marginX, marginY, marginX + sizeX, marginY + sizeY), 1, 1, 1)
        
        #self.SD.SetCapability(twain.ICAP_BITDEPTH, twain.TWTY_UINT16, 16)

        #self.SD.SetCapability(twain.ICAP_AUTOMATICDESKEW,xxxxx)
        self.SD.SetCapability(twain.ICAP_BITORDER, twain.TWTY_UINT16, 1)
        
        settings_dict = {}
        #PRINTS ALL CAPABILITIES
        caps = self.SD.GetCapability(twain.CAP_SUPPORTEDCAPS)
        for i in range(len(caps[1][2])):
            try:
                settings_dict[caps[1][2][i]] = self.SD.GetCapabilityCurrent(caps[1][2][i])
            except:
                pass
        
        self.SD.RequestAcquire(0, 0)  # 1,1 to show scanner user interface
        
        for key, item in settings_dict.items():
            if item != self.SD.GetCapabilityCurrent(key):
                print key, "has changed from", item, "to", self.SD.GetCapabilityCurrent(key)
            
        self.AcquirePending=True
        print "*** Waiting for scanner"

    def AcquireNatively(self, scanner=None, handle=None):
        """Acquire Natively - this is a memory based transfer"""
        self.XferMethod = _TRANSFER_METHOD
        return self._Acquire(handle)

    def Process_Transfer(self, Info=None):
        """An image is ready at the scanner - fetch and display it"""
        more_to_come = False
        try:
            if self.XferMethod == _TRANSFER_METHOD:
                gotFile = True
                try:
                    (handle, more_to_come) = self.SD.XferImageNatively()
                except:
                    print "*** Warning: Scan was cancelled or failed"
                    gotFile = False
                if gotFile:
                    global sys
                                            
                    print "Image is:", Info['ImageWidth'],"x",Info['ImageLength'],"with resolution", Info['XResolution'], "x", Info['YResolution'], "and pixel-depth", Info['BitsPerPixel']

                    header_size = 1064

                    file_size_in_bytes = Info['ImageWidth'] * Info['ImageLength']+header_size
                    image_data = twain.GlobalHandleGetBytes(handle, 0, file_size_in_bytes)                    

                    im = Image.fromstring("L", (Info['ImageWidth'], Info['ImageLength']), str(image_data)[header_size:],"raw","L;I",0,1)
                    im.save(self.next_file_name, "TIFF")

                    twain.GlobalHandleFree(handle)
                    print "*** Image aquired"

                    if self.owner:
                        self.owner.log_file({'File':self.next_file_name,'ImageWidth': Info['ImageWidth'],'ImageLength':Info['ImageLength'],"XResolution":Info['XResolution'], 'YResolution': Info['YResolution'], 'BitsPerPixel':Info['BitsPerPixel'],'Histogram':im.histogram(),'Scanned Time':time.time()})
                        self.owner.check_quality()
                    else:
                        print im.histogram()
            if more_to_come: 
                self.AcquirePending = True
        except:
            # Display information about the exception
            import sys, traceback
            ei = sys.exc_info()
            traceback.print_exception(ei[0], ei[1], ei[2])

    def OnTwainEvent(self, event):
        """This is an event handler for the twain event. It is called 
        by the thread that set up the callback in the first place.

        It is only reliable on wxPython. Otherwise use the Polling mechanism above.
        
        """
        try:
            if event == twain.MSG_XFERREADY:
                self.AcquirePending = False
                self.Process_Transfer(self.SD.GetImageInfo())
                if not self.AcquirePending:
                    if self.owner:
                        self.owner._scanner_id = self.SD.GetSourceName()
                    self.Terminate()
            elif event == twain.MSG_CLOSEDSREQ:
                self.SD = None
        except:
            # Display information about the exception
            import sys, traceback
            ei = sys.exc_info()
            traceback.print_exception(ei[0], ei[1], ei[2])

    def VerifyCanWrite(self, filepath):
        """The scanner can have a configuration with a transfer file that cannot
        be created. This method raises an exception for this case."""
        parts = os.path.split(filepath)
        if parts[0]:
            dirpart=parts[0]
        else:
            dirpart='.'
        if not os.access(dirpart, os.W_OK):
            raise CannotWriteTransferFile, filepath


