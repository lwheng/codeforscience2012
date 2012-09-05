import Utils

class citprov:
  def __init__(self):
    self.pickler = Utils.pickler()
    self.model = self.pickler.loadPickle(self.pickler.pathModel)
    print

  def predict(self, model, query):
    # Process query
    queryProcessed = ""
    return model.predict(queryProcessed)
