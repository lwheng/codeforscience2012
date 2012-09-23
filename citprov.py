import Utils
import Feature_Extractor
import sys
import json

class citprov:
  def __init__(self):
    self.dist = Utils.dist()
    self.nltk_Tools = Utils.nltk_tools()
    self.pickler = Utils.pickler()
    self.tools = Utils.tools()
    self.weight = Utils.weight()
    self.dataset_tools = Utils.dataset_tools(self.dist, self.nltk_Tools, self.pickler, self.tools)
    self.extractor = Feature_Extractor.extractor(self.dist, self.nltk_Tools, self.pickler, self.tools, self.weight, "authors", "titles")
    # Load model for prediction
    self.model = self.pickler.loadPickle('ModelCFS.pickle')
    self.model_v2 = self.pickler.loadPickle('ModelCFS_v2.pickle')

  def interpret_predictions_v2(self, model, feature_vectors):
    b_index = 0
    r_index = 0
    prob = -1
    for i in range(len(feature_vectors)):
      fv = feature_vectors[i]
      res = model.predict_proba(fv)[0]
      for j in range(len(res)):
        if res[j] > prob:
          b_index = i
          r_index = j
          prob = res[j]
    return (b_index, r_index)

  def section_finder(self, b):
    section_header_node = None
    target = b.previousSibling
    while target:
      if target.nodeType == Node.ELEMENT_NODE:
        if target.nodeName == 'sectionHeader':
          section_header_node = target
          break
      target = target.previousSibling
    if target == None:
      return 'none'
    if section_header_node.attributes.has_key('genericHeader'):
      header = section_header_node.attributes['genericHeader'].value
    elif section_header_node.attributes.has_key('genericheader'):
      header = section_header_node.attributes['genericheader'].value
    return header

  def predict_v2(self, model_v2, dom_citing_parscit, dom_citing_parscit_section, dom_cited_parscit, dom_cited_parscit_section):
    entries = []

    # Citing
    title_citing = dom_citing_parscit_section.getElementsByTagName('title')[0].firstChild.wholeText
    dom_authors_citing = dom_citing_parscit_section.getElementsByTagName('authors')
    authors_citing = []
    for dom_author in dom_authors_citing[0].getElementsByTagName('fullname'):
      authors_citing.append(dom_author.firstChild.wholeText)

    # Get all citations
    citations = dom_citing_parscit.getElementsByTagName('citation')
    for citation in citations:
      # Get title
      dom_title_cited = citation.getElementsByTagName('title')
      if dom_title_cited:
        title_cited = dom_title_cited[0].firstChild.wholeText

      # Get authors
      dom_authors_cited = citation.getElementsByTagName('author')
      authors_cited = []
      for a in dom_authors_cited:
        authors_cited.append(a.firstChild.wholeText)

      # Get contexts
      dom_contexts_citing = citation.getElementsByTagName('context')
      context_list = []
      for c in dom_contexts_citing:
        value = c.firstChild.wholeText.lower()
        context_list.append(self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value)))
      citing_col = self.nltk_Tools.nltkTextCollection(context_list)

      # Get body_texts from dom_parscit_section_cited for output
      body_texts = dom_cited_parscit_section.getElementsByTagName('bodyTexts')

      # For each context, need to predict which bodyText is the prov
      # With the prediction, return the section, and the bodytext itself
      # General - 0, Specific (Yes) - 1, Specific (No) - 2, Undetermined - 3
      for c in dom_contexts_citing:
        cite_context = c.firstChild.wholeText
        feature_vectors = self.extractor.extractFeaturesCFS_v2(c, citing_col, dom_citing_parscit_section, dom_cited_parscit_section, title_citing, title_cited, authors_citing, authors_cited)
        prediction = self.interpret_predictions_v2(model_v2, feature_vectors)
        if prediction[1] != 1:
          entry = {'cite-context':cite_context, 'prov-type':'general', 'prov-section':'none', 'prov-snippet':'none'}
        else:
          # Find the section the body_text belong to
          b = body_texts[prediction[0]]
          header = section_finder(b)
          entry = {'cite-context':cite_context, 'prov-type':'specific', 'prov-section':header, 'prov-snippet':b.firstChild.wholeText}
        entries.append(entry)
    return json.dumps(entries)

  def predict(self, model, dom_citing_parscit, dom_citing_parscit_section):
    # Uses only DOMs from the current (citing) paper
    # We can return provenance predictions for all contexts found in this current paper
    # Open and read files into memory

    entries = []

    # Citing
    title_citing = dom_citing_parscit_section.getElementsByTagName('title')[0].firstChild.wholeText
    dom_authors_citing = dom_citing_parscit_section.getElementsByTagName('authors')
    authors_citing = []
    for dom_author in dom_authors_citing[0].getElementsByTagName('fullname'):
      authors_citing.append(dom_author.firstChild.wholeText)

    # Get all citations
    citations = dom_citing_parscit.getElementsByTagName('citation')
    for citation in citations:
      # Get title
      dom_title_cited = citation.getElementsByTagName('title')
      if dom_title_cited:
        title_cited = dom_title_cited[0].firstChild.wholeText

      # Get authors
      dom_authors_cited = citation.getElementsByTagName('author')
      authors_cited = []
      for a in dom_authors_cited:
        authors_cited.append(a.firstChild.wholeText)

      # Get contexts
      dom_contexts_cited = citation.getElementsByTagName('context')
      context_list = []
      for c in dom_contexts_cited:
        value = c.firstChild.wholeText.lower()
        context_list.append(self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value)))
      citing_col = self.nltk_Tools.nltkTextCollection(context_list)

      for c in dom_contexts_cited:
        cite_context = c.firstChild.wholeText
        feature_vector = self.extractor.extractFeaturesCFS_v1(c, citing_col, dom_citing_parscit_section, title_citing, title_cited, authors_citing, authors_cited)
        prediction = model.predict(feature_vector)
        if prediction == -1:
          prediction = "specific"
        else:
          prediction = "general"
        entry = {'cite-context':cite_context, 'prov-type':prediction}
        entries.append(entry)
    return json.dumps(entries)
