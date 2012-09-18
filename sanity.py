execfile('Utils.py')
execfile('Feature_Extractor.py')
execfile('citprov.py')
import citprov
c = citprov.citprov()
print c.predict(c.model, 'sample-parscit.xml', 'sample-parscit-section.xml', 'sample-parscit.xml', 'sample-parscit-section.xml')
