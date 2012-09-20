import Utils
import Feature_Extractor
import sys

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

  def predict_v2(self, model_v2, dom_citing_parscit, dom_citing_parscit_section, dom_cited_parscit, dom_cited_parscit_section):
    data = open(citing_parscit,'r').read()
    dom_citing_parscit = self.tools.parseXML(self.tools.normalize(data))
    data = open(citing_parscit_section,'r').read()
    dom_citing_parscit_section = self.tools.parseXML(self.tools.normalize(data))
    data = open(cited_parscit,'r').read()
    dom_cited_parscit = self.tools.parseXML(self.tools.normalize(data))
    data = open(cited_parscit_section,'r').read()
    dom_cited_parscit_section = self.tools.parseXML(self.tools.normalize(data))

    # Get Titles
    title_citing = dom_citing_parscit_section.getElementsByTagName('title')[0].firstChild.wholeText
    title_cited = dom_cited_parscit_section.getElementsByTagName('title')[0].firstChild.wholeText

    # Get Authors
    dom_authors_citing = dom_citing_parscit_section.getElementsByTagName('authors')
    dom_authors_cited = dom_cited_parscit_section.getElementsByTagName('authors')

    authors_citing = []
    authors_cited = []
    for dom_author in dom_authors_citing[0].getElementsByTagName('fullname'):
      authors_citing.append(dom_author.firstChild.wholeText)
    for dom_author in dom_authors_cited[0].getElementsByTagName('fullname'):
      authors_cited.append(dom_author.firstChild.wholeText)

    # Extract all chunks in cited paper
    #bodyTexts = dom_cited_parscit_section.getElementsByTagName('bodyText')

    # Extract contexts
    citation = self.dataset_tools.prepContextsCFS(self.dist, self.tools, title_citing, title_cited, dom_citing_parscit)
    contexts = citation.getElementsByTagName('context')

    # Set up citing_col for TF-IDF
    context_list = []
    for c in contexts:
      value = c.firstChild.wholeText.lower()
      context_list.append(self.nltk_Tools.nltkText(self.nltk_Tools.nltkWordTokenize(value)))
    citing_col = self.nltk_Tools.nltkTextCollection(context_list)

    for c in contexts:
      feature_vector = self.extractor.extractFeaturesCFS_v2(c, citing_col, dom_citing_parscit_section, title_citing, title_cited, authors_citing, authors_cited)

    # Predict query using model
    return model.predict(feature_vector)

  def predict(self, model, dom_citing_parscit, dom_citing_parscit_section):
    # Uses only DOMs from the current (citing) paper
    # We can return provenance predictions for all contexts found in this current paper
    # Open and read files into memory

    # Citing
    title_citing = dom_citing_parscit_section.getElementsByTagName('title')[0].firstChild.wholeText
    dom_authors_citing = dom_citing_parscit_section.getElementsByTagName('authors')
    authors_citing = []
    for dom_author in dom_authors_citing[0].getElementsByTagName('fullname'):
      authors_citing.append(dom_author.firstChild.wholeText)

    # Get all contexts
    citations = dom_citing_parscit.getElementsByTagName('citation')
    prediction_list = []
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
        feature_vector = self.extractor.extractFeaturesCFS_v1(c, citing_col, dom_citing_parscit_section, title_citing, title_cited, authors_citing, authors_cited)
        prediction = model.predict(feature_vector)
        prediction_list.append(prediction)
    return prediction_list
