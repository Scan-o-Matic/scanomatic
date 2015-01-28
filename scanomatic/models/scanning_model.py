
import scanomatic.generics.model as model

class ScanningModel(model.Model):

    def __init__(self, numberOfScans=217, timeBetweenScans=20,
                 projectName="", directoryContainingProject="",
                 projectTag="", scannerTag="",
                 description="", email="", pinningFormats=tuple(),
                 fixture="", scanner=""):

        super(ScanningModel, self).__init__(
            numberOfScans=numberOfScans, timeBetweenScans=timeBetweenScans,                                                         
            projectName=projectName,
            directoryContainingProject=directoryContainingProject,
            projectTag=projectTag, scannerTag=scannerTag,
            description=description, email=email,
            pinningFormats=pinningFormats,                                                     
            fixture=fixture, scanner=scanner)

        
