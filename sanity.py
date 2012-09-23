execfile('Utils.py')
execfile('Feature_Extractor.py')
execfile('citprov.py')
import citprov
import sys
c = citprov.citprov()

# Open and read files in memory
# Citing
citing_parscit_file = 'sample-parscit.xml'
citing_parscit_section_file = 'sample-parscit-section.xml'
data1 = open(citing_parscit_file).read()
data2 = open(citing_parscit_section_file).read()
dom_citing_parscit = c.tools.parseXML(c.tools.normalize(data1))
dom_citing_parscit_section = c.tools.parseXML(c.tools.normalize(data2))

# Cited
cited_parscit_file = 'sample-parscit.xml'
cited_parscit_section_file = 'sample-parscit-section.xml'
data1 = open(cited_parscit_file).read()
data2 = open(cited_parscit_section_file).read()
dom_cited_parscit = c.tools.parseXML(c.tools.normalize(data1))
dom_cited_parscit_section = c.tools.parseXML(c.tools.normalize(data2))

# Version 1: Does not require the cited paper; Does not output a 'region' prediction
entries = c.predict(c.model, dom_citing_parscit, dom_citing_parscit_section)

# Version 2: Requires the cited paper; prediction a best region for citation's context
prediction_list = c.predict_v2(c.model_v2, dom_citing_parscit, dom_citing_parscit_section, dom_cited_parscit, dom_cited_parscit_section)
print prediction_list
