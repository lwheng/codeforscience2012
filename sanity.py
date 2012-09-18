execfile('Utils.py')
execfile('Feature_Extractor.py')
execfile('citprov.py')
import citprov
c = citprov.citprov()
prediction = c.predict(c.model, 'sample-parscit.xml', 'sample-parscit-section.xml', 'sample-parscit.xml', 'sample-parscit-section.xml')
if prediction == -1:
  print "Specific"
else:
  print "General"
