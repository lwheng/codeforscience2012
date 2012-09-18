execfile('citprov.py')
import citprov
c = citprov.citprov()
c.predict("asd", 'sample-parscit.xml', 'sample-parscit-section.xml', 'sample-parscit.xml', 'sample-parscit-section.xml')
